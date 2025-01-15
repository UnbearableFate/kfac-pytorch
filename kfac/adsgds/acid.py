import threading
import time

import torch
from sympy.core.random import random
from torch.distributed import rpc
from typing import TYPE_CHECKING
from mpi4py import MPI
from torch.onnx.symbolic_opset9 import tensor

import kfac.rpc_util.GraphConstruct as GraphConstruct
from kfac.rpc_util.common_util import model2flatten_tensor, flatten_tensor2model ,gradient2flatten_tensor
import random
from kfac.adsgds.common import ModelStore, rpc_work_name, RootModelAvgRPCCommunicator , compute_recv_weight_by_loss

if TYPE_CHECKING:
    from kfac.rpc_distributed import KFacRPCCommunicator

class AcidModelAvgRPCCommunicator:
    def __init__(self,model: torch.nn.Module ,rpc_communicator: 'KFacRPCCommunicator'):
        self.rpc_communicator: 'KFacRPCCommunicator' = rpc_communicator
        self.world_size_cb = rpc_communicator.get_world_size
        self.current_t_cb = self.rpc_communicator.current_t
        self.origin_world_size = rpc_communicator.origin_world_size
        self.rank = rpc_communicator.rank
        self.model = model
        flat_model = model2flatten_tensor(self.model)
        self.local_flat_models = torch.stack([flat_model, flat_model.clone()], dim = 0)
        self.t = time.time()
        self.lr = 0.001
        self.exp_mat = torch.tensor([[-self.lr,  self.lr],
                   [ self.lr, -self.lr]], dtype=torch.float32)

        self.graph = GraphConstruct.GraphConstruct(self.rank ,self.origin_world_size, MPI.COMM_WORLD, 'clique-ring', 'swift', p = 0.15, num_c=4)
        global model_avg_rpc_communicator
        model_avg_rpc_communicator = self

    # did forward and backward pass before calling this function
    def process(self,current_loop_start_time):
        E = torch.matrix_exp((current_loop_start_time- self.t) * self.exp_mat)
        self.local_flat_models = torch.matmul(E, self.local_flat_models)
        flatten_gradient = gradient2flatten_tensor(self.model)
        self.local_flat_models[0].add_(-self.lr* flatten_gradient)
        self.local_flat_models[1].add_(-self.lr* flatten_gradient)
        flatten_tensor2model(self.local_flat_models[0], self.model)
        self.t = current_loop_start_time

    def process2(self,current_loop_start_time):
        E = torch.matrix_exp((current_loop_start_time- self.t) * self.exp_mat)
        self.local_flat_models = torch.matmul(E, self.local_flat_models)
        flatten_gradient = gradient2flatten_tensor(self.model)
        self.local_flat_models[0].add_(-self.lr* flatten_gradient)
        self.local_flat_models[1].add_(-self.lr* flatten_gradient)
        flatten_tensor2model(self.local_flat_models[0], self.model)
        self.t = current_loop_start_time

model_avg_rpc_communicator: AcidModelAvgRPCCommunicator

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
    model_avg_rpc_communicator.local_flat_model.setDataWithLock(data, from_term, from_loss)
    if model_avg_rpc_communicator.local_flat_model.term < model_avg_rpc_communicator.current_t_cb():
        model_avg_rpc_communicator.update_local_flat_model()
    return model_avg_rpc_communicator.local_flat_model.getData()