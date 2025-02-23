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
        self.recv_buf = ModelStore(self.local_model_store.flatten_tensor)
        #self.graph = GraphConstruct.GraphConstruct(rank,self.origin_world_size, MPI.COMM_WORLD, 'ring', 'adpsgd', p = 0.15, num_c=8)
        global model_avg_rpc_communicator
        model_avg_rpc_communicator = self
    
    """
    call this func after backward propagation
    """
    def process(self):
        self.update_local_flat_model()
        # randomly select a neighbor to exchange model from wolrd_size-1 neighbors
        rank_list = list(range(self.origin_world_size))
        rank_list.remove(self.rank)
        target_neighbor = random.choice(rank_list)
        with self.rpc_communicator.node_state_lock:
            node_states = self.rpc_communicator.node_states.copy()
        rpc.rpc_async(
            to=rpc_work_name(target_neighbor),
            func=recv_model_param,
            args=(*self.local_model_store.getData(), node_states)
        )
        if self.recv_buf.loss_value <= 0 and self.recv_buf.term <= 0:
            return

        with self.local_model_store.lock:
            self.local_model_store.flatten_tensor.add_(self.recv_buf.flatten_tensor,alpha=0.5)
            vector_to_parameters(self.local_model_store.flatten_tensor, self.model.parameters())

model_avg_rpc_communicator: AdpsgdManager

def exchange_model_param(data):
    global model_avg_rpc_communicator
    if model_avg_rpc_communicator.local_model_store.term < model_avg_rpc_communicator.current_t_cb():
       model_avg_rpc_communicator.update_local_flat_model()
    recv_weight = 0.5
    with model_avg_rpc_communicator.local_model_store.lock:
        model_avg_rpc_communicator.local_model_store.flatten_tensor = model_avg_rpc_communicator.local_model_store.flatten_tensor * (1-recv_weight) + data*recv_weight
        vector_to_parameters(model_avg_rpc_communicator.local_model_store.flatten_tensor , model_avg_rpc_communicator.model.parameters())
    return model_avg_rpc_communicator.local_model_store.getData()

def recv_model_param(from_flat_tensor, from_term, from_loss_value,node_states):
    global model_avg_rpc_communicator
    self = model_avg_rpc_communicator
    self.rpc_communicator.update_node_states(node_states)
    if self.recv_buf.loss_value > from_loss_value:
        self.recv_buf.setDataWithLock(from_flat_tensor, from_term, from_loss_value)