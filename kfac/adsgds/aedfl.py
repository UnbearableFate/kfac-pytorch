import torch
from sympy.core.random import random
from torch.distributed import rpc
from typing import TYPE_CHECKING
from mpi4py import MPI
from kfac.rpc_util.GraphConstruct import exponential_topology_sources,exponential_topology_targets
from kfac.adsgds.common import ModelStore, rpc_work_name,RootModelAvgRPCCommunicator , compute_recv_weight_by_loss
import torch.distributed as dist
import time
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from typing import Dict

if TYPE_CHECKING:
    from kfac.rpc_distributed import KFacRPCCommunicator

class NewModelStore(ModelStore):
    lambda_lr = 0.2
    def __init__(self, flatten_tensor: torch.Tensor,rank):
        super().__init__(flatten_tensor)
        self.dynamic_weight = 1
        self.lamda_weight = 1
        self.grad_of_lamda = 0
        self.rank = rank
        self.flatten_grad = None
    
    def update_dynamic_weight(self):
        if self.loss_value == 0:
            return
        self.dynamic_weight = self.lamda_weight /  self.loss_value

    def update_lamda_weight(self):
        self.lamda_weight -= NewModelStore.lambda_lr * self.grad_of_lamda

class AedflManager(RootModelAvgRPCCommunicator):
    def __init__(self, rank: int, model: torch.nn.Module, rpc_communicator: 'KFacRPCCommunicator'):
        super().__init__(rank, model, rpc_communicator)
        self.local_model_store = NewModelStore(parameters_to_vector(self.model.parameters()),rank)
        self.targets = exponential_topology_targets(self.origin_world_size, rank)
        self.sources = exponential_topology_sources(self.origin_world_size, rank)
        self.local_model_store.weight = 1/(len(self.sources)+1)
        self.neighbor_model_buffers : Dict[int, NewModelStore] = {}
        for index,neighbor in enumerate(self.sources):
            self.neighbor_model_buffers[neighbor] = NewModelStore(self.local_model_store.flatten_tensor,neighbor)
            self.neighbor_model_buffers[neighbor].weight = 1/(len(self.sources)+1)
        
        self.index = 0
        global model_avg_rpc_communicator
        model_avg_rpc_communicator = self
    
    def formula_4_6(self):
        self.local_model_store.update_dynamic_weight()
        for rank,neighbor in self.neighbor_model_buffers.items():
            neighbor.update_dynamic_weight()
     
    def formula_4_7(self):
        sum_of_dynamic_weight = self.sum_of_dynamic_weight()
        self.local_model_store.weight = self.local_model_store.dynamic_weight / sum_of_dynamic_weight
        for neighbor in self.neighbor_model_buffers.values():
            neighbor.weight = neighbor.dynamic_weight / sum_of_dynamic_weight
    
    def formula_4_8(self):
        self.formula_4_10()
        self.local_model_store.update_lamda_weight()
        for neighbor in self.neighbor_model_buffers.values():
            neighbor.update_lamda_weight()

    def sum_of_dynamic_weight(self):
        s = self.local_model_store.dynamic_weight
        for neighbor in self.neighbor_model_buffers.values():
            s += neighbor.dynamic_weight
        return s
    
    def formula_4_10(self):
        sum_of_dynamic_weight = self.sum_of_dynamic_weight()
        
        left = (sum_of_dynamic_weight - self.local_model_store.dynamic_weight) / sum_of_dynamic_weight**2
        right = float(torch.dot(self.local_model_store.flatten_grad, self.local_model_store.flatten_tensor) / self.local_model_store.loss_value)
        self.local_model_store.grad_of_lamda = left * right

        for neighbor in self.neighbor_model_buffers.values():
            left = (sum_of_dynamic_weight - neighbor.dynamic_weight) / sum_of_dynamic_weight**2
            right = float(torch.dot(self.local_model_store.flatten_grad, neighbor.flatten_tensor) / neighbor.loss_value)
            neighbor.grad_of_lamda = left * right
    
    @torch.no_grad()
    def update_local_flat_model(self):
        with self.local_model_store.lock:
            self.local_model_store.flatten_tensor = parameters_to_vector(self.model.parameters())
            self.local_model_store.term = self.current_t_cb()
            self.local_model_store.flatten_grad = torch.cat([
                p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1)
                for p in self.model.parameters()
            ])
    
    # call after backward before optimizer.step()
    def process(self):
        self.update_local_flat_model()
        node_states = self.rpc_communicator.get_node_states()
        for neighbor in self.targets:
            rpc.rpc_async(
                to=rpc_work_name(neighbor),
                func= recv_model_param,
                args=(*self.local_model_store.getData(),self.rank, node_states)
            )
        if self.local_model_store.loss_value == 0:
            return
        for neighbor_store in self.neighbor_model_buffers.values():
            if neighbor_store.loss_value == 0:
                return
        
        self.formula_4_8() # update lamda_weight
        self.formula_4_6() # update dynamic_weight
        self.formula_4_7() # update real weight
        
        result = self.local_model_store.flatten_tensor * self.local_model_store.weight
        for neighbor in self.neighbor_model_buffers.values():
            with neighbor.lock:
                result += neighbor.flatten_tensor * neighbor.weight

        with torch.no_grad() , self.local_model_store.lock:
            vector_to_parameters(result, self.model.parameters())
    
model_avg_rpc_communicator: AedflManager

def recv_model_param(data, term, loss_value,from_rank,node_states):
    global model_avg_rpc_communicator
    if from_rank not in model_avg_rpc_communicator.neighbor_model_buffers:
        print(f"Error: from_rank {from_rank} not in neighbor_model_buffers")
        return None
    model_avg_rpc_communicator.rpc_communicator.update_node_states(node_states) 
    model_avg_rpc_communicator.neighbor_model_buffers[from_rank].setDataWithLock(data, term, loss_value)