import torch
from sympy.core.random import random
from torch.distributed import rpc
from typing import TYPE_CHECKING
from mpi4py import MPI
from kfac.rpc_util.GraphConstruct import exponential_topology_sources,exponential_topology_targets
from kfac.rpc_util.common_util import flatten_tensor2model, compute_l2_norm
import random
from kfac.adsgds.layer_type_common import rpc_work_name,RootModelAvgRPCCommunicator ,LayerwiseModelStore
import torch.distributed as dist
import time
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from typing import Dict

if TYPE_CHECKING:
    from kfac.rpc_distributed import KFacRPCCommunicator

class LWExpTopoSwiftManager(RootModelAvgRPCCommunicator):
    def __init__(self, rank: int, model: torch.nn.Module, rpc_communicator: 'KFacRPCCommunicator'):
        super().__init__(rank, model, rpc_communicator)
        self.targets = exponential_topology_targets(self.origin_world_size, rank)
        self.sources = exponential_topology_sources(self.origin_world_size, rank)
        self.local_model_store.aggration_weight = 1/(len(self.sources)+1)
        self.neighbor_model_buffers : Dict[int,LayerwiseModelStore] = {}
        for index,neighbor in enumerate(self.sources):
            self.neighbor_model_buffers[neighbor] = LayerwiseModelStore()
            self.neighbor_model_buffers[neighbor].clone_model_store(self.local_model_store)
            self.neighbor_model_buffers[neighbor].aggration_weight = 1/(len(self.sources)+1)
        
        self.index = 0
        global model_avg_rpc_communicator
        model_avg_rpc_communicator = self

    def process(self):
        async_result = []
        for neighbor in self.targets:
            work = rpc.rpc_async(
                to=rpc_work_name(neighbor),
                func= recv_model_param,
                args=(*self.local_model_store.getData(),self.rank)
            )
            async_result.append(work)
        torch.futures.wait_all(async_result)
        
        for neighbor_store in self.neighbor_model_buffers.values():
            if neighbor_store.loss_value == 0:
                return
        
        with self.local_model_store.lock and torch.no_grad(): 
            for layer_name in self.local_model_store.layer_parameters.keys():
                self.local_model_store.layer_parameters[layer_name].mul_(self.local_model_store.aggration_weight)
                for neighbor, neighbor_store in self.neighbor_model_buffers.items():
                    with neighbor_store.lock:
                        self.local_model_store.layer_parameters[layer_name].add_(neighbor_store.layer_parameters[layer_name] ,alpha=neighbor_store.aggration_weight)
        

model_avg_rpc_communicator: LWExpTopoSwiftManager

def recv_model_param(data, term, loss_value,from_rank):
    global model_avg_rpc_communicator
    if from_rank not in model_avg_rpc_communicator.neighbor_model_buffers:
        print(f"Error: from_rank {from_rank} not in neighbor_model_buffers")
        return None
    model_avg_rpc_communicator.neighbor_model_buffers[from_rank].setDataWithLock(data, term, loss_value)