import datetime
import math
import random
import time

import torch
import torch.distributed.rpc as rpc
import threading
from typing import Dict, Optional
import logging
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from kfac.layers.eigen import KFACEigenLayer,KFACBaseLayer

import torch

import kfac.model_param_avg_rpc as model_param_avg_rpc

# 创建日志记录器
#logger = logging.getLogger('my_logger')
#ogger.setLevel(logging.DEBUG)  # 设置日志级别

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
    return round(similarity.item(),5)

def rpc_work_name(rank:int) -> str:
    return f"rpc_{rank}"

class KfacRPCLayer:
    def __init__(self,a_handler,g_handler ,name , prediv_eigenvalues):
        self.factor : Dict[str:Optional['torch.Tensor']] = {"A" : None, "G": None }
        self.factor_recv_ct : Dict[str:Dict[int, int]] = {"A" : {}, "G": {} } # {t: count}
        self.assigned_worker :Dict[str : int]= {'A': a_handler, 'G': g_handler}
        self.qa = None
        self.da = None
        self.recv_handled_a_version = -2
        self.last_load_handled_tensor_version = -2
        self.qg = None
        self.dg = None
        self.dgda = None
        self.prediv_eigenvalues = prediv_eigenvalues # False：da, dg must be provided; True: dgda must be provided
        self.recv_handled_g_version = -2
        self.outdated_weight_param = 1
        self.ahead_weight_param = 1
        self.name = name

    def is_handled_a_updated(self):
        return self.qa is not None and self.recv_handled_a_version > self.last_load_handled_tensor_version

    def is_handled_g_updated(self):
        return self.qg is not None and self.recv_handled_g_version > self.last_load_handled_tensor_version

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
        self.factor[factor_type] = (1 - recv_world_weight) * self.factor[factor_type] + recv_world_weight * recv_factor

    def update_local_eigen_a(self, qa, da, t):
        if t <= self.recv_handled_a_version :
            return # outdated
        self.qa = qa
        self.da = da
        self.recv_handled_a_version = t

    def update_local_eigen_g(self, qg, dg, dgda, t):
        if t <= self.recv_handled_g_version :
            return # outdated
        self.qg = qg
        self.dg = dg
        self.dgda = dgda
        self.recv_handled_g_version = t

    def clear_count_dict(self,local_t):
        n = 3
        for factor_type in self.factor_recv_ct.keys():
            self.factor_recv_ct[factor_type] = {k: v for k, v in self.factor_recv_ct[factor_type].items() if k >= local_t - n}

    def load_eigen_tensor(self, kfac_layer: 'KFACEigenLayer', t):
        assert self.name == kfac_layer.name
        while self.qa is None or self.qg is None:
            time.sleep(0.1)
            #logger.debug(f"Waiting for eigen tensor of {self.name} to be ready")
        kfac_layer.qa = self.qa.clone()
        kfac_layer.qg = self.qg.clone()
        if self.prediv_eigenvalues:
            kfac_layer.dgda = self.dgda.clone()
        else:
            kfac_layer.dg = self.dg.clone()
            kfac_layer.da = self.da.clone()
        self.last_load_handled_tensor_version = t

class KFacRPCCommunicator:
    def __init__(self, world_size, rank, preconditioner ,model):
        self.io_layers = None
        self.skip_inverse_computation_flag = 0
        rpc.init_rpc(name=f"rpc_{rank}", rank=rank, world_size=world_size)
        self.world_size = world_size
        self.rank = rank
        self.rpc_layers: Dict[str:KfacRPCLayer] = {} # {layer_name: KfacRPCLayer}
        self.computer_type = "" #eigen / inverse
        self.factor_computer_list = []
        for name, kfac_layer in preconditioner._layers.values():
            a_handler = preconditioner._assignment.inv_worker(name, 'A')
            g_handler = preconditioner._assignment.inv_worker(name, 'G')
            self.rpc_layers[name] = KfacRPCLayer(a_handler,g_handler ,name ,kfac_layer.prediv_eigenvalues)
            self.factor_computer_list.append(name)

        self.iter_of_rank = [-1] * world_size # the latest iteration of each rank received
        self.lock = threading.Lock()
        self.param_lock = threading.Lock()

        # hyperparameters
        self.necessary_ct = world_size - 2
        self.load_inverse_max_loop = 3
        if rpc.is_available():
            print(f"RPC Communicator initialized for rank {rank}")

        #self.init_logger(rank)
        self.model_avg_rpc = model_param_avg_rpc.ModelAvgRPCCommunicator(world_size, rank, model ,self.current_t)

        global global_communicator
        global_communicator = self

    def init_logger(self,rank):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        # 创建一个 FileHandler，并设置级别为 DEBUG
        file_handler = logging.FileHandler(f'log_{rank}_{timestamp}.log')
        file_handler.setLevel(logging.DEBUG)

        # 创建一个日志格式器，并将其添加到 FileHandler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        global logger
        # 将 FileHandler 添加到日志记录器
        logger.addHandler(file_handler)

        # 移除默认的 StreamHandler（终端输出）
        if logger.hasHandlers():
            logger.handlers.clear()
            logger.addHandler(file_handler)

    def __repr__(self):
        log = f"Rank {self.rank} : iter {self.current_t()}\n"
        for name, layer in self.rpc_layers.items():
            log += f"Layer {name}:\n"
            for factor_type, factor in layer.factor.items():
                log += f"\t{factor_type} factor: {factor}\n"
                log += f"\t{factor_type} factor recv ct: {layer.factor_recv_ct[factor_type]}\n"
            log += f"\tA eigen: {layer.qa}, {layer.da}\n"
            log += f"\tG eigen: {layer.qg}, {layer.dg}, {layer.dgda}\n"
            log += f"\thandled A recv iter: {layer.recv_handled_a_version}\n"
            log += f"\thandled G recv iter: {layer.recv_handled_g_version}\n"
            log += f"\tA handler: {layer.assigned_worker['A']}\n"
            log += f"\tG handler: {layer.assigned_worker['G']}\n"
        return log

    def load_factor(self,kfac_layer: 'KFACBaseLayer', factor_type):
        if self.assigned_worker(kfac_layer.name, factor_type) == self.rank:
            while not self.is_factor_ready(kfac_layer.name, factor_type):
                time.sleep(0.1)
                #logger.debug(f"Waiting for factor of {kfac_layer.name} to be ready")
            assert self.rpc_layers[kfac_layer.name].factor[factor_type] is not None
            kfac_layer.a_factor = self.rpc_layers[kfac_layer.name].factor["A"].clone().detach()
        return True

    def load_eigen_tensor(self,kfac_layer, loop):
        layer_name = kfac_layer.name
        current_t = self.current_t()

        if self.rpc_layers[layer_name].last_load_handled_tensor_version == current_t:
            return False # already loaded

        if (self.rpc_layers[layer_name].recv_handled_g_version >= self.current_t() -loop
                or self.rpc_layers[layer_name].recv_handled_a_version == self.current_t() - loop
                or loop >= self.load_inverse_max_loop -1):
            self.rpc_layers[layer_name].load_eigen_tensor(kfac_layer,current_t)
            return True
        return False

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
        current_t = self.current_t()
        return self.rpc_layers[layer_name].factor_recv_ct[factor_type].get(current_t,0) >= self.necessary_ct
    
    def assigned_worker(self, layer_name, factor_type):
        return self.rpc_layers[layer_name].assigned_worker[factor_type]
    
    def send_kfac_factor(self,layer_name:str,factor_tensor :torch.Tensor, factor_type:str):
        if layer_name not in self.factor_computer_list:
            return True
        target = 0
        if factor_type == "A":
            target = self.rpc_layers[layer_name].assigned_worker['A']
        elif factor_type == "G":
            target = self.rpc_layers[layer_name].assigned_worker['G']
        t = self.current_t()
        if target == self.rank:
            self.rpc_layers[layer_name].update_local_factor(factor_tensor.clone(), t, t, factor_type)
            return
        try:
            rpc.rpc_async(
                to=rpc_work_name(target),
                func=receive_kfac_factor,
                args=(self.rank, layer_name, factor_tensor, t, factor_type)
            )
        except Exception as e:
            print(f"Failed to send factor to {target}: {e}")

        return True

    def send_kfac_eigen_tensor(self, layer_name,q:torch.Tensor,d:torch.Tensor,dd :torch.Tensor, factor_type):
        if self.assigned_worker(layer_name, factor_type) != self.rank:
            return True

        if q is None:
            raise RuntimeError(
                f'Attempt to broadcast {factor_type} inv from src={self.rank} but this rank '
                'has not computed inv yet.',
            )

        with self.lock:
            t = self.current_t()
            d_clone,dd_clone = None, None
            if d is not None and isinstance(d, torch.Tensor):
                d_clone = d.clone()
            if dd is not None and isinstance(dd, torch.Tensor):
                dd_clone = dd.clone()
            if factor_type == "A":
                self.rpc_layers[layer_name].update_local_eigen_a(q.clone(), d_clone, t)
            if factor_type == "G":
                self.rpc_layers[layer_name].update_local_eigen_g(q.clone(), d_clone, dd_clone, t)

        for i in range(self.world_size-1):
            target_rank = (self.rank + i + 1) % self.world_size
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
    def facotr_comput_lazy_wl_rebal(self):
        forward_than_local = sum(t > self.current_t() for t in self.iter_of_rank)
        late_than_local = sum(t > self.current_t() for t in self.iter_of_rank)
        if forward_than_local >= math.ceil(self.world_size * 0.7): # local is too slow, work less
            if len(self.factor_computer_list) > 1:
                layer_name = random.choice(self.factor_computer_list)
                while (self.rpc_layers[layer_name].assigned_worker['A'] == self.rank or 
                       self.rpc_layers[layer_name].assigned_worker['G'] == self.rank):
                    layer_name = random.choice(self.factor_computer_list)
                self.factor_computer_list.remove(layer_name)
                #logger.info(f"Skip computing factor for {layer_name} in rank {self.rank}")
                return
            else:
                self.skip_inverse_computation_flag = 4
        if late_than_local >= 1: #math.ceil(self.world_size * 0.3): # local is quick, work more
            if len(self.factor_computer_list) < len(self.rpc_layers):
                for layer_name in self.rpc_layers.keys():
                    if layer_name not in self.factor_computer_list:
                        self.factor_computer_list.append(layer_name)
                        #logger.info(f"Add computing factor for {layer_name} in rank {self.rank}")

    def send_model_param(self):
        self.model_avg_rpc.send_all_model_param()

global_communicator: KFacRPCCommunicator

def receive_kfac_factor(from_rank,layer_name, factor, t,factor_type):
    global global_communicator
    if global_communicator is not None:
        self = global_communicator
        with self.lock:
            current_t = self.current_t()
            self.rpc_layers[layer_name].update_local_factor(factor, current_t, t, factor_type)
            self.update_other_rank_iter(from_rank,t)

def receive_eigen_tensor_a(from_rank, layer_name, qa, da, t):
    global global_communicator
    with global_communicator.lock:
        if t < global_communicator.rpc_layers[layer_name].recv_handled_a_version:
            return
        global_communicator.rpc_layers[layer_name].update_local_eigen_a(qa, da, t)
        global_communicator.update_other_rank_iter(from_rank, t)

def receive_eigen_tensor_g(from_rank, layer_name, qg, dg, dadg, t):
    global global_communicator
    with global_communicator.lock:
        if t < global_communicator.rpc_layers[layer_name].recv_handled_g_version:
            return
        global_communicator.rpc_layers[layer_name].update_local_eigen_g(qg, dg, dadg, t)
        global_communicator.update_other_rank_iter(from_rank,t)