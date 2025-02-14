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
        res = rpc.rpc_sync(
            to=rpc_work_name(target_neighbor),
            func=exchange_model_param,
            args=(*self.local_model_store.getData(), self.rank, self.get_local_node_speed())
        )
        if res is not None and len(res) == 3:
            self.local_model_store.setDataWithLock(res[0], res[1], res[2])
        else:
            print(f"Rank {self.rank} get None from {target_neighbor}")
            return

        with self.local_model_store.lock:
            vector_to_parameters(self.local_model_store.flatten_tensor, self.model.parameters())

model_avg_rpc_communicator: AdpsgdManager

def exchange_model_param(data, from_term, from_loss, from_rank ,from_speed = 0):
    global model_avg_rpc_communicator
    model_avg_rpc_communicator.rpc_communicator.update_node_iter(from_rank, from_term,from_speed)
    if model_avg_rpc_communicator.local_model_store.term < model_avg_rpc_communicator.current_t_cb():
       model_avg_rpc_communicator.update_local_flat_model()
    recv_weight = 0.5
    with model_avg_rpc_communicator.local_model_store.lock:
        model_avg_rpc_communicator.local_model_store.flatten_tensor = model_avg_rpc_communicator.local_model_store.flatten_tensor * (1-recv_weight) + data*recv_weight
        vector_to_parameters(model_avg_rpc_communicator.local_model_store.flatten_tensor , model_avg_rpc_communicator.model.parameters())
    return model_avg_rpc_communicator.local_model_store.getData()