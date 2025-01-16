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
        self.graph = GraphConstruct.GraphConstruct(rank,self.origin_world_size, MPI.COMM_WORLD, 'clique-ring', 'swift', p = 0.15, num_c=3)
        self.sw = 1 - sum(self.graph.neighbor_weights)
        self.neighbor_model_buffers : Dict[str, ModelStore] = {}
        for index,neighbor in enumerate(self.graph.neighbor_list):
            self.neighbor_model_buffers[neighbor] = ModelStore(self.local_model_store.flatten_tensor)
            self.neighbor_model_buffers[neighbor].weight = self.graph.neighbor_weights[index]
        
        self.index = 0
        global model_avg_rpc_communicator
        model_avg_rpc_communicator = self

    """
    call this func after backward propagation
    """
    def process(self): # OK?
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
    
    def process1(self): # OK !
        self.index += 1
        send_tensor = parameters_to_vector(self.model.parameters())
        
        for send_rank, recv_rank in self.graph.graph:
            if self.rank == send_rank and recv_rank in self.graph.neighbor_list:
                dist.send(tensor=send_tensor, dst=recv_rank)
            elif self.rank == recv_rank and send_rank in self.graph.neighbor_list:
                dist.recv(tensor=self.neighbor_model_buffers[send_rank].flatten_tensor, src=send_rank)
    
        for recv_rank, send_rank in self.graph.graph:
            if self.rank == send_rank and recv_rank in self.graph.neighbor_list:
                dist.send(tensor=send_tensor, dst=recv_rank)
            elif self.rank == recv_rank and send_rank in self.graph.neighbor_list:
                dist.recv(tensor=self.neighbor_model_buffers[send_rank].flatten_tensor, src=send_rank)
        
        send_tensor.mul_(self.sw)
        for neighbor_store in self.neighbor_model_buffers.values():
            send_tensor.add_(neighbor_store.flatten_tensor, alpha=neighbor_store.weight)

        with torch.no_grad():
            vector_to_parameters(send_tensor, self.model.parameters())
        
    def process2(self):
        send_tensor = parameters_to_vector(self.model.parameters())
        term = self.current_t_cb()
        loss = self.local_model_store.loss_value
        for neighbor in self.graph.neighbor_list:
            rpc.rpc_async(
                to=rpc_work_name(neighbor),
                func= recv_model_param,
                args=(send_tensor,term,loss,self.rank)
            )
        for neighbor_store in self.neighbor_model_buffers.values():
            if neighbor_store.loss_value == 0 or self.local_model_store.loss_value == 0:
                return
        
        send_tensor.mul_(self.sw)
        for neighbor_store in self.neighbor_model_buffers.values():
            send_tensor.add_(neighbor_store.flatten_tensor, alpha=neighbor_store.weight)

        with torch.no_grad():
            vector_to_parameters(send_tensor, self.model.parameters())

model_avg_rpc_communicator: SwiftManager

def recv_model_param(data, term, loss_value,from_rank):
    global model_avg_rpc_communicator
    if from_rank not in model_avg_rpc_communicator.graph.neighbor_list:
        return None
    model_avg_rpc_communicator.neighbor_model_buffers[from_rank].setData(data, term, loss_value)
    