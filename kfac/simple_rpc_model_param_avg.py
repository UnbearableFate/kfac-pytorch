import threading
import torch
from sympy.core.random import random
from torch.distributed import rpc
from typing import TYPE_CHECKING
from mpi4py import MPI
import kfac.rpc_util.GraphConstruct as GraphConstruct
from kfac.rpc_util.common_util import model2flatten_tensor, flatten_tensor2model
import random
if TYPE_CHECKING:
    from kfac.rpc_distributed import KFacRPCCommunicator

def rpc_work_name(rank:int) -> str:
    return f"rpc_{rank}"

class ModelStore:
    def __init__(self ,flatten_tensor: torch.Tensor):
        self.term = 0
        self.loss_value = 0
        self.lock = threading.Lock()
        self.flatten_tensor = flatten_tensor.clone()

    def getData(self):
        return self.flatten_tensor,self.term,self.loss_value

    def setData(self,data,term,loss_value):
        self.flatten_tensor = data
        self.term = term
        self.loss_value = loss_value
    def setDataWithLock(self,data,term,loss_value):
        with self.lock:
            self.flatten_tensor = data
            self.term = term
            self.loss_value = loss_value

class SimpleModelAvgRPCCommunicator:
    def __init__(self, rank, model: torch.nn.Module ,rpc_communicator: 'KFacRPCCommunicator'):
        self.rpc_communicator: 'KFacRPCCommunicator' = rpc_communicator
        self.world_size_cb = rpc_communicator.get_world_size
        self.current_t_cb = self.rpc_communicator.current_t
        self.origin_world_size = rpc_communicator.origin_world_size
        self.rank = rank
        self.model = model

        self.local_avg_flat_model = ModelStore(model2flatten_tensor(self.model))

        self.graph = GraphConstruct.GraphConstruct(rank,self.origin_world_size, MPI.COMM_WORLD, 'clique-ring', 'swift', p = 0.15, num_c=4) 
        self.sw = 1 - sum(self.graph.neighbor_weights)
        print(f"Rank {rank} has neighbors {self.graph.neighbor_list}")
        print(f"Rank {rank} has weights {self.graph.neighbor_weights}")
        print(f"Rank {rank} has sw {self.sw}")
        self.neighbors_model_buffer = {}
        for n in self.graph.neighbor_list:
            self.neighbors_model_buffer[n] = ModelStore(self.local_avg_flat_model.flatten_tensor)

        global model_avg_rpc_communicator
        model_avg_rpc_communicator = self

    def set_loss(self, loss_value):
        self.local_avg_flat_model.loss_value = loss_value

    def get_local_node_speed(self):
        if self.rpc_communicator.node_states[self.rank].speed is not None and self.rpc_communicator.node_states[self.rank].speed != 0:
            return self.rpc_communicator.node_states[self.rank].speed
        elif self.rpc_communicator.time_cost_accumulation != 0:
            self.rpc_communicator.node_states[self.rank].speed = int(self.rpc_communicator.computation_volume_accumulation / self.rpc_communicator.time_cost_accumulation)
            return self.rpc_communicator.node_states[self.rank].speed
        return None

    def broadcast_model(self):
        current_local_flat_model = model2flatten_tensor(self.model)
        self.local_avg_flat_model.flatten_tensor = current_local_flat_model
        self.local_avg_flat_model.term = self.current_t_cb()

        for neighbor in self.graph.neighbor_list:
            rpc.rpc_async(
                to=rpc_work_name(neighbor),
                func=receive_model_param,
                args=(self.rank,current_local_flat_model, self.rpc_communicator.current_t(), self.local_avg_flat_model.loss_value,
                      self.get_local_node_speed())
            )

    def update_model(self):
        self.local_avg_flat_model.flatten_tensor.mul_(self.sw)
        for index, neighbor in enumerate(self.graph.neighbor_list):
            with self.neighbors_model_buffer[neighbor].lock:
                self.local_avg_flat_model.flatten_tensor.add_(self.neighbors_model_buffer[neighbor].flatten_tensor * self.graph.neighbor_weights[index])

        flatten_tensor2model(self.local_avg_flat_model.flatten_tensor, self.model)

    def avg_model_with_neighbors(self):
        if self.current_t_cb() % 5 == 0:
            self.broadcast_model()
            self.update_model()

    def update_local_flat_model(self):
        with self.local_avg_flat_model.lock:
            self.local_avg_flat_model.flatten_tensor = model2flatten_tensor(self.model)
            self.local_avg_flat_model.term = self.current_t_cb()

    def adsgd_exchange_model(self):
        self.update_local_flat_model()

        target_neighbor = random.choice(self.graph.neighbor_list)
        res = rpc.rpc_sync(
            to=rpc_work_name(target_neighbor),
            func= exchange_model_param,
            args=(*self.local_avg_flat_model.getData(), self.get_local_node_speed())
        )
        if res is not None and len(res) == 3:
            self.neighbors_model_buffer[target_neighbor].setDataWithLock(*res)

        # average local flat tensor with the target neighbor
        self.local_avg_flat_model.flatten_tensor.mul_(0.5).add_(self.neighbors_model_buffer[target_neighbor].flatten_tensor * 0.5)
        flatten_tensor2model(self.local_avg_flat_model.flatten_tensor, self.model)

model_avg_rpc_communicator: SimpleModelAvgRPCCommunicator

def receive_model_param(from_rank,data,from_rank_term,from_loss, from_speed = 0):
    global model_avg_rpc_communicator

    if from_rank not in model_avg_rpc_communicator.graph.neighbor_list:
        return
    with model_avg_rpc_communicator.neighbors_model_buffer[from_rank].lock:
        model_avg_rpc_communicator.neighbors_model_buffer[from_rank].flatten_tensor = data
        model_avg_rpc_communicator.neighbors_model_buffer[from_rank].term = from_rank_term
        model_avg_rpc_communicator.neighbors_model_buffer[from_rank].loss_value = from_loss

def exchange_model_param(data, from_rank, from_term, from_loss, from_speed = 0):
    global model_avg_rpc_communicator
    if (from_rank not in model_avg_rpc_communicator.graph.neighbor_list or
        from_term <= model_avg_rpc_communicator.neighbors_model_buffer[from_rank].term):
        return
    model_avg_rpc_communicator.rpc_communicator.update_node_iter(from_rank,from_term, from_speed)
    model_avg_rpc_communicator.local_avg_flat_model.setDataWithLock(data, from_term, from_loss)
    if model_avg_rpc_communicator.local_avg_flat_model.term < model_avg_rpc_communicator.current_t_cb():
        model_avg_rpc_communicator.update_local_flat_model()
    return model_avg_rpc_communicator.local_avg_flat_model.getData()