import torch
import torch.distributed.rpc as rpc
import threading
from typing import Dict

def rpc_work_name(rank:int) -> str:
    return f"rpc_{rank}"

class KfacRPCLayer:
    def __init__(self,a_handler=None,g_handler=None):
        self.factor : Dict[str:torch.Tensor] = {"A" : None, "G": None }
        self.factor_recv_ct : Dict[str:Dict[int, int]] = {"A" : {}, "G": {} } # {t: count}
        self.a_handler = a_handler
        self.qa = None
        self.da = None
        self.recv_handled_a_iter = -2 # if inverse a belongs to this rank, =-1
        self.g_handler = g_handler
        self.qg = None
        self.dg = None
        self.dgda = None
        self.recv_handled_g_iter = -2
        self.outdated_weight_param = 1.1
        self.ahead_weight_param = 0.9

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
        self.kfac_layers = {} # {layer_name: { factor type name: factor A /G}}
        self.rpc_layers = {} # {layer_name: KfacRPCLayer}
        self.computer_type = "" #eigen / inverse

        for name, kfac_layer in preconditioner._layers.values():
            self.kfac_layers[name] = kfac_layer
            a_handler = preconditioner._assignment.inv_worker(name, 'A')
            g_handler = preconditioner._assignment.inv_worker(name, 'G')
            self.rpc_layers[name] = KfacRPCLayer(a_handler,g_handler)

        self.iter_of_rank = [-1] * world_size # the latest iteration of each rank received
        self.lock = threading.Lock()

        # hyperparameters
        self.necessary_ct = world_size - 1
        if rpc.is_available():
            print(f"RPC Communicator initialized for rank {rank}")

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
            log += f"\tA handler: {layer.a_handler}\n"
            log += f"\tG handler: {layer.g_handler}\n"
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

    def send_kfac_factor(self,layer_name:str, factor_type:str):
        factor = None
        target = 0
        if factor_type == "A":
            factor = self.kfac_layers[layer_name].a_factor
            target = rpc_work_name(self.rpc_layers[layer_name].a_handler)
        elif factor_type == "G":
            factor = self.kfac_layers[layer_name].g_factor
            target = rpc_work_name(self.rpc_layers[layer_name].g_handler)
        t = self.current_t()
        try:
            rpc.rpc_async(
                to=target,
                func=receive_kfac_factor,
                args=(self.rank, layer_name, factor, t, factor_type)
            )
        except Exception as e:
            print(f"Failed to send factor to {target}: {e}")

    def update_other_rank_iter(self, from_rank,t):
        self.iter_of_rank[from_rank] = max(self.iter_of_rank[from_rank], t)

    def is_factor_ready(self, layer_name,factor_type):
        with self.lock:
            current_t = self.current_t()
            return self.rpc_layers[layer_name].factor_recv_ct[factor_type].get(current_t,0) >= self.necessary_ct
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