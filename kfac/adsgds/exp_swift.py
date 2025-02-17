import torch
from sympy.core.random import random
from torch.distributed import rpc
from typing import TYPE_CHECKING
from mpi4py import MPI
from kfac.rpc_util.GraphConstruct import exponential_topology_sources,exponential_topology_targets
from kfac.rpc_util.common_util import flatten_tensor2model, compute_l2_norm
import random
from kfac.adsgds.common import ModelStore, rpc_work_name,RootModelAvgRPCCommunicator , compute_recv_weight_by_loss
import torch.distributed as dist
import time
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from typing import Dict

if TYPE_CHECKING:
    from kfac.rpc_distributed import KFacRPCCommunicator

class ExpTopoSwiftManager(RootModelAvgRPCCommunicator):
    def __init__(self, rank: int, model: torch.nn.Module, rpc_communicator: 'KFacRPCCommunicator'):
        super().__init__(rank, model, rpc_communicator)
        self.targets = exponential_topology_targets(self.origin_world_size, rank)
        self.sources = exponential_topology_sources(self.origin_world_size, rank)
        self.local_model_store.weight = 1/(len(self.sources)+1)
        self.neighbor_model_buffers : Dict[int, ModelStore] = {}
        for index,neighbor in enumerate(self.sources):
            self.neighbor_model_buffers[neighbor] = ModelStore(self.local_model_store.flatten_tensor)
            self.neighbor_model_buffers[neighbor].weight = 1/(len(self.sources)+1)
        
        self.index = 0
        global model_avg_rpc_communicator
        model_avg_rpc_communicator = self

    def process(self):
        self.update_local_flat_model()
        result = self.local_model_store.flatten_tensor * self.local_model_store.weight
        for neighbor in self.targets:
            rpc.rpc_async(
                to=rpc_work_name(neighbor),
                func= recv_model_param,
                args=(*self.local_model_store.getData(),self.rank)
            )
        
        for neighbor_store in self.neighbor_model_buffers.values():
            if neighbor_store.loss_value == 0:
                return

        for neighbor_store in self.neighbor_model_buffers.values():
            with neighbor_store.lock:
                result.add_(neighbor_store.flatten_tensor, alpha=neighbor_store.weight)

        with torch.no_grad() and self.local_model_store.lock:
            vector_to_parameters(result, self.model.parameters())
    

model_avg_rpc_communicator: ExpTopoSwiftManager

def recv_model_param(data, term, loss_value,from_rank):
    global model_avg_rpc_communicator
    if from_rank not in model_avg_rpc_communicator.neighbor_model_buffers:
        print(f"Error: from_rank {from_rank} not in neighbor_model_buffers")
        return None
    model_avg_rpc_communicator.neighbor_model_buffers[from_rank].setDataWithLock(data, term, loss_value)