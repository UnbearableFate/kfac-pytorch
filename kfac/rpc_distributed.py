import datetime
import torch
import torch.distributed.rpc as rpc
import threading
from typing import Dict
import logging

import torch

def normalized_l2_similarity(tensor1, tensor2):
    """
    计算两个任意维度 Tensor 之间的归一化 L2 范数相似度。

    Args:
        tensor1 (torch.Tensor): 第一个输入 Tensor
        tensor2 (torch.Tensor): 第二个输入 Tensor

    Returns:
        float: 两个 Tensor 之间的归一化 L2 范数相似度，值范围在 [0, 1] 之间
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError("Both tensors must have the same shape")

    l2_norm = torch.norm(tensor1 - tensor2)
    max_norm = torch.norm(tensor1) + torch.norm(tensor2)
    similarity = l2_norm / max_norm
    return round(similarity.item(),6)

def rpc_work_name(rank:int) -> str:
    return f"rpc_{rank}"

class KfacRPCLayer:
    def __init__(self,a_handler=None,g_handler=None ,name=None):
        self.factor : Dict[str:torch.Tensor] = {"A" : None, "G": None }
        self.factor_recv_ct : Dict[str:Dict[int, int]] = {"A" : {}, "G": {} } # {t: count}
        self.assigned_worker :Dict[str : int]= {}
        self.assigned_worker['A'] = a_handler
        self.qa = None
        self.da = None
        self.recv_handled_a_iter = -2 # if inverse a belongs to this rank, =-1
        self.assigned_worker['G'] = g_handler
        self.qg = None
        self.dg = None
        self.dgda = None
        self.recv_handled_g_iter = -2
        self.outdated_weight_param = 1
        self.ahead_weight_param = 1
        self.name = name

    def update_local_factor(self, recv_factor, local_t, recv_t, factor_type):
        self.factor_recv_ct[factor_type][recv_t] = self.factor_recv_ct[factor_type].get(recv_t, 0) + 1
        if self.factor[factor_type] is None:
            self.factor[factor_type] = recv_factor
            return
        recv_world_weight = 1.0 / self.factor_recv_ct[factor_type][recv_t]
        if local_t > recv_t:
            recv_world_weight *= self.outdated_weight_param
        if local_t < recv_t:
            recv_world_weight *= self.ahead_weight_param
        temp = (1 - recv_world_weight) * self.factor[factor_type] + recv_world_weight * recv_factor
        self.factor[factor_type] = temp

    def update_local_eigen_a(self, qa, da, t):
        self.qa = qa
        self.da = da
        self.recv_handled_a_iter = t

    def update_local_eigen_g(self, qg, dg, dgda, t):
        self.qg = qg
        self.dg = dg
        self.dgda = dgda
        self.recv_handled_g_iter = t

    def clear_count_dict(self,local_t):
        n = 3
        for factor_type in self.factor_recv_ct.keys():
            self.factor_recv_ct[factor_type] = {k: v for k, v in self.factor_recv_ct[factor_type].items() if k >= local_t - n}


class KFacRPCCommunicator:
    def __init__(self, world_size, rank, preconditioner):
        rpc.init_rpc(name=f"rpc_{rank}", rank=rank, world_size=world_size)
        self.world_size = world_size
        self.rank = rank
        self.rpc_layers: Dict[str:KfacRPCLayer] = {} # {layer_name: KfacRPCLayer}
        self.computer_type = "" #eigen / inverse

        for name, kfac_layer in preconditioner._layers.values():
            a_handler = preconditioner._assignment.inv_worker(name, 'A')
            g_handler = preconditioner._assignment.inv_worker(name, 'G')
            self.rpc_layers[name] = KfacRPCLayer(a_handler,g_handler ,name)

        self.iter_of_rank = [-1] * world_size # the latest iteration of each rank received
        self.lock = threading.Lock()

        # hyperparameters
        self.necessary_ct = world_size - 1
        if rpc.is_available():
            print(f"RPC Communicator initialized for rank {rank}")


        # 创建日志记录器
        self.logger = logging.getLogger('my_logger')
        self.logger.setLevel(logging.DEBUG)  # 设置日志级别
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        # 创建一个 FileHandler，并设置级别为 DEBUG
        file_handler = logging.FileHandler(f'app_{self.rank}_{timestamp}.log')
        file_handler.setLevel(logging.DEBUG)

        # 创建一个日志格式器，并将其添加到 FileHandler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 将 FileHandler 添加到日志记录器
        self.logger.addHandler(file_handler)

        # 移除默认的 StreamHandler（终端输出）
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            self.logger.addHandler(file_handler)

        global global_communicator
        global_communicator = self

    def __repr__(self):
        log = f"Rank {self.rank} : iter {self.current_t()}\n"
        for name, layer in self.rpc_layers.items():
            log += f"Layer {name}:\n"
            for factor_type, factor in layer.factor.items():
                log += f"\t{factor_type} factor: {factor}\n"
                log += f"\t{factor_type} factor recv ct: {layer.factor_recv_ct[factor_type]}\n"
            log += f"\tA eigen: {layer.qa}, {layer.da}\n"
            log += f"\tG eigen: {layer.qg}, {layer.dg}, {layer.dgda}\n"
            log += f"\thandled A recv iter: {layer.recv_handled_a_iter}\n"
            log += f"\thandled G recv iter: {layer.recv_handled_g_iter}\n"
            log += f"\tA handler: {layer.assigned_worker['A']}\n"
            log += f"\tG handler: {layer.assigned_worker['G']}\n"
        return log

    def clear_count_dict(self):
        for layer in self.rpc_layers.values():
            layer.clear_count_dict(self.current_t())
    def shutdown(self):
        rpc.shutdown()
    def update_self_t(self):
        self.iter_of_rank[self.rank] += 1

    def current_t(self):
        return self.iter_of_rank[self.rank]

    def update_other_rank_iter(self, from_rank,t):
        self.iter_of_rank[from_rank] = max(self.iter_of_rank[from_rank], t)

    def is_factor_ready(self, layer_name, factor_type):
        if self.rpc_layers[layer_name].assigned_worker[factor_type] != self.rank:
            return True
        with self.lock:
            current_t = self.current_t()
            return self.rpc_layers[layer_name].factor_recv_ct[factor_type].get(current_t,0) >= self.necessary_ct
    
    def assigned_worker(self, layer_name, factor_type):
        return self.rpc_layers[layer_name].assigned_worker[factor_type]
    
    def send_kfac_factor(self,layer_name:str,factor_tensor :torch.Tensor, factor_type:str):
        target = 0
        if factor_type == "A":
            target = self.rpc_layers[layer_name].assigned_worker['A']
        elif factor_type == "G":
            target = self.rpc_layers[layer_name].assigned_worker['G']
        t = self.current_t()

        fut = None
        try:
            fut =  rpc.rpc_async(
                to=rpc_work_name(target),
                func=receive_kfac_factor,
                args=(self.rank, layer_name, factor_tensor, t, factor_type)
            )
        except Exception as e:
            print(f"Failed to send factor to {target}: {e}")
        if fut is not None and target == self.rank:
            fut.wait()

    def send_kfac_eigen_tensor(self, layer_name,q,d,dd, factor_type):
        t = self.current_t()
        for target_rank in range(self.world_size):
            if target_rank == self.rank:
                continue
            if factor_type == "A":
                rpc.rpc_async(
                    to=rpc_work_name(target_rank),
                    func=receive_eigen_tensor_a,
                    args=(self.rank, layer_name, q, d, t)
                )
            if factor_type == "G":
                rpc.rpc_async(
                    to=rpc_work_name(target_rank),
                    func=receive_eigen_tensor_g,
                    args=(self.rank, layer_name, q, d,dd, t)
                )

global_communicator: KFacRPCCommunicator
def receive_kfac_factor(from_rank,layer_name, factor, t,factor_type):
    global global_communicator
    if global_communicator is not None:
        self = global_communicator
        with self.lock:
            current_t = self.current_t()
            self.rpc_layers[layer_name].update_local_factor(factor, current_t, t, factor_type)
            self.update_other_rank_iter(from_rank,t)

def receive_eigen_tensor_a(from_rank, layer_name, q, d, t):
    global global_communicator
    with global_communicator.lock:
        global_communicator.rpc_layers[layer_name].update_local_eigen_a(q, d, t)
        global_communicator.update_other_rank_iter(from_rank, t)

def receive_eigen_tensor_g(from_rank, layer_name, q, d, dd, t):
    global global_communicator
    with global_communicator.lock:
        global_communicator.rpc_layers[layer_name].update_local_eigen_g(q, d, dd, t)
        global_communicator.update_other_rank_iter(from_rank,t)