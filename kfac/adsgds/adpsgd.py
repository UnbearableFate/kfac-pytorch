import torch
from sympy.core.random import random
from torch.distributed import rpc
from typing import TYPE_CHECKING
from mpi4py import MPI
import kfac.rpc_util.GraphConstruct as GraphConstruct
import random
from kfac.adsgds.common import ModelStore, rpc_work_name,RootModelAvgRPCCommunicator , compute_recv_weight_by_loss
import torch.distributed as dist
import time
from torch.nn.utils import parameters_to_vector, vector_to_parameters
if TYPE_CHECKING:
    from kfac.rpc_distributed import KFacRPCCommunicator

class AdpsgdManager(RootModelAvgRPCCommunicator):
    def __init__(self, rank: int, model: torch.nn.Module, rpc_communicator: 'KFacRPCCommunicator'):
        super().__init__(rank, model, rpc_communicator)
        self.graph = GraphConstruct.GraphConstruct(rank,self.origin_world_size, MPI.COMM_WORLD, 'clique-ring', 'swift', p = 0.15, num_c=3)
        self.sw = 1 - sum(self.graph.neighbor_weights)
        print(f"Rank {rank} has neighbors {self.graph.neighbor_list}")
        print(f"Rank {rank} has weights {self.graph.neighbor_weights}")
        print(f"Rank {rank} has sw {self.sw}")
        self.neighbor_model_buffer = ModelStore(self.local_model_store.flatten_tensor)

        global model_avg_rpc_communicator
        model_avg_rpc_communicator = self
    
    """
    call this func after backward propagation
    """
    def adsgd_exchange_model(self):
        self.update_local_flat_model()

        target_neighbor = random.choice(self.graph.neighbor_list)
        res = rpc.rpc_sync(
            to=rpc_work_name(target_neighbor),
            func= exchange_model_param,
            args=(*self.local_model_store.getData(), self.rank, self.get_local_node_speed())
        )
        if res is not None:
            self.neighbor_model_buffer.setDataWithLock(*res)
        else:
            print(f"Rank {self.rank} get None from {target_neighbor}")

        # average local flat tensor with the target neighbor
        recv_weight = compute_recv_weight_by_loss(self.local_model_store.loss_value, self.neighbor_model_buffer.loss_value)
        with self.local_model_store.lock:
            self.local_model_store.flatten_tensor.mul_(1-recv_weight).add_(self.neighbor_model_buffer.flatten_tensor, alpha=recv_weight)
            vector_to_parameters(self.local_model_store.flatten_tensor, self.model.parameters())
    
    def process(self):
        self.update_local_flat_model()
        target_neighbor = random.choice(self.graph.neighbor_list)
        res = rpc.rpc_async(
            to=rpc_work_name(target_neighbor),
            func= recv_model_param,
            args=(),
        )
        if res is not None:
        # average local flat tensor with the target neighbor
            with self.local_model_store.lock and torch.no_grad():
                self.local_model_store.flatten_tensor.add_(res[0]).mul_(0.5)
                vector_to_parameters(self.local_model_store.flatten_tensor, self.model.parameters())


model_avg_rpc_communicator: AdpsgdManager

def recv_model_param():
    global model_avg_rpc_communicator
    if model_avg_rpc_communicator.local_model_store.term < model_avg_rpc_communicator.current_t_cb():
        model_avg_rpc_communicator.update_local_flat_model()
    return model_avg_rpc_communicator.local_model_store.flatten_tensor, model_avg_rpc_communicator.local_model_store.term, model_avg_rpc_communicator.local_model_store.loss_value

def exchange_model_param(data, from_term, from_loss, from_rank ,from_speed = 0):
    global model_avg_rpc_communicator
    model_avg_rpc_communicator.rpc_communicator.update_node_iter(from_rank, from_term,from_speed)
    if model_avg_rpc_communicator.local_model_store.term < model_avg_rpc_communicator.current_t_cb():
       model_avg_rpc_communicator.update_local_flat_model()
    recv_weight = compute_recv_weight_by_loss(model_avg_rpc_communicator.local_model_store.loss_value, from_loss)
    with model_avg_rpc_communicator.local_model_store.lock:
        vector_to_parameters(model_avg_rpc_communicator.local_model_store.flatten_tensor * (1-recv_weight) + data*recv_weight, model_avg_rpc_communicator.model.parameters())
    return model_avg_rpc_communicator.local_model_store.getData()