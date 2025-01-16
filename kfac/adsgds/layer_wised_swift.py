import torch
from sympy.core.random import random
from torch.distributed import rpc
from typing import TYPE_CHECKING, Dict
from mpi4py import MPI
import kfac.rpc_util.GraphConstruct as GraphConstruct
import random
from kfac.adsgds.layer_type_common import LayerwiseModelStore, RootModelAvgRPCCommunicator, rpc_work_name
if TYPE_CHECKING:
    from kfac.rpc_distributed import KFacRPCCommunicator

class LWSwiftManager(RootModelAvgRPCCommunicator):
    def __init__(self, rank, model: torch.nn.Module, rpc_communicator: 'KFacRPCCommunicator'):
        super().__init__(rank, model, rpc_communicator)
        self.graph = GraphConstruct.GraphConstruct(rank,self.origin_world_size, MPI.COMM_WORLD, 'clique-ring', 'swift', p = 0.15, num_c=3)
        self.sw = 1 - sum(self.graph.neighbor_weights)
        self.neighbor_model_buffers : Dict[int, LayerwiseModelStore] = {}
        for index,neighbor in enumerate(self.graph.neighbor_list):
            self.neighbor_model_buffers[neighbor] = LayerwiseModelStore()
            self.neighbor_model_buffers[neighbor].clone_model_store(self.local_model_store)
            self.neighbor_model_buffers[neighbor].aggration_weight = self.graph.neighbor_weights[index]

        global model_avg_rpc_communicator
        model_avg_rpc_communicator = self

    """
    call this func after backward propagation
    """
    def process(self): # this function is ok
        for neighbor in self.graph.neighbor_list:
            rpc.rpc_async(
                to=rpc_work_name(neighbor),
                func= recv_model_param,
                args=(*self.local_model_store.getData(),self.rank)
            )
        for neighbor_store in self.neighbor_model_buffers.values():
            if neighbor_store.loss_value == 0 or self.local_model_store.loss_value == 0:
                return
        
        temp_model_store = LayerwiseModelStore()
        temp_model_store.clone_model_store(self.local_model_store)

        for layer_name in self.local_model_store.layer_parameters.keys():
            temp_model_store.layer_parameters[layer_name].mul_(self.sw)
            for neighbor, neighbor_store in self.neighbor_model_buffers.items():
                temp_model_store.layer_parameters[layer_name].add_(neighbor_store.layer_parameters[layer_name] ,alpha=neighbor_store.aggration_weight)
        
        with self.local_model_store.lock and torch.no_grad():
            for layer_name in self.local_model_store.layer_parameters.keys():
                self.local_model_store.layer_parameters[layer_name].copy_(temp_model_store.layer_parameters[layer_name])

model_avg_rpc_communicator: LWSwiftManager

def recv_model_param(data, from_term, from_loss, from_rank):
    global model_avg_rpc_communicator
    if from_rank not in model_avg_rpc_communicator.graph.neighbor_list:
        return
    model_avg_rpc_communicator.neighbor_model_buffers[from_rank].setData(data, from_term, from_loss)