import torch
from sympy.core.random import random
from torch.distributed import rpc
from typing import TYPE_CHECKING
from mpi4py import MPI
import kfac.rpc_util.GraphConstruct as GraphConstruct
from kfac.rpc_util.common_util import flatten_tensor2model, compute_l2_norm
import random
from kfac.adsgds.common import ModelStore, rpc_work_name,RootModelAvgRPCCommunicator , compute_recv_weight_by_loss
import torch.distributed as dist
import time
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from typing import Dict

if TYPE_CHECKING:
    from kfac.rpc_distributed import KFacRPCCommunicator

class SwiftManager(RootModelAvgRPCCommunicator):
    def __init__(self, rank: int, model: torch.nn.Module, rpc_communicator: 'KFacRPCCommunicator'):
        super().__init__(rank, model, rpc_communicator)
        self.graph = GraphConstruct.GraphConstruct(rank,self.origin_world_size, MPI.COMM_WORLD, 'ring', 'swift', p = 0.15, num_c=8)
        self.local_model_store.weight = 1 - sum(self.graph.neighbor_weights)
        self.neighbor_model_buffers : Dict[int, ModelStore] = {}
        for index,neighbor in enumerate(self.graph.neighbor_list):
            self.neighbor_model_buffers[neighbor] = ModelStore(self.local_model_store.flatten_tensor)
            self.neighbor_model_buffers[neighbor].weight = self.graph.neighbor_weights[index]
        
        self.index = 0
        global model_avg_rpc_communicator
        model_avg_rpc_communicator = self

    """
    call this func after backward propagation
    """
    def process_sync(self): # OK?
        self.index += 1
        send_tensor = parameters_to_vector(self.model.parameters())
        recv_tensor = send_tensor.clone()
        if self.index % 2 == 0:
            if self.rank % 2 == 0:
                dist.send(tensor=send_tensor, dst=(self.rank+1)%self.origin_world_size)
            else:
                dist.recv(tensor=recv_tensor, src=(self.rank-1)%self.origin_world_size)
        else:
            if self.rank % 2 == 0:
                dist.recv(tensor=recv_tensor, src=(self.rank+1)%self.origin_world_size)
            else:
                dist.send(tensor=send_tensor, dst=(self.rank-1)%self.origin_world_size)

        vector_to_parameters(recv_tensor.add_(send_tensor).mul_(0.5), self.model.parameters())
    
    def process(self):
        self.update_local_flat_model()
        send_work_list = []
        result = self.local_model_store.flatten_tensor * self.local_model_store.weight
        for neighbor in self.graph.neighbor_list:
            work = rpc.rpc_async(
                to=rpc_work_name(neighbor),
                func= recv_model_param,
                args=(*self.local_model_store.getData(),self.rank)
            )
            send_work_list.append(work)

        for neighbor_store in self.neighbor_model_buffers.values():
            if neighbor_store.loss_value == 0:
                return

        for neighbor_store in self.neighbor_model_buffers.values():
            with neighbor_store.lock:
                result.add_(neighbor_store.flatten_tensor, alpha=neighbor_store.weight)

        with torch.no_grad() and self.local_model_store.lock:
            vector_to_parameters(result, self.model.parameters())

        self.rpc_communicator.debug_print(f"model avg process done ,memeory usage is {self.rpc_communicator.get_memory_usage_percent()}")
        self.rpc_communicator.com_statistic.add_send_stat("model_param", len(self.graph.neighbor_list))


model_avg_rpc_communicator: SwiftManager

def recv_model_param(data, term, loss_value,from_rank):
    global model_avg_rpc_communicator
    if from_rank not in model_avg_rpc_communicator.graph.neighbor_list:
        return None
    model_avg_rpc_communicator.neighbor_model_buffers[from_rank].setDataWithLock(data, term, loss_value)