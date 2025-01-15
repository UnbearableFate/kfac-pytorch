import threading
import torch
from sympy.core.random import random
from torch.distributed import rpc
from typing import TYPE_CHECKING, Dict
from mpi4py import MPI
import kfac.rpc_util.GraphConstruct as GraphConstruct
from kfac.rpc_util.common_util import model2flatten_tensor, flatten_tensor2model
import random
from scipy.special import expit
if TYPE_CHECKING:
    from kfac.rpc_distributed import KFacRPCCommunicator

def rpc_work_name(rank:int) -> str:
    return f"rpc_{rank}"

class ModelStore:
    def __init__(self ,flatten_tensor: torch.Tensor):
        self.term = 0
        self.loss_value = 0
        self.lock = threading.Lock()
        self.flatten_tensor = torch.zeros_like(flatten_tensor)

    def getData(self):
        return [self.flatten_tensor,self.term,self.loss_value]

    def setData(self,data,term,loss_value):
        self.flatten_tensor = data
        self.term = term
        self.loss_value = loss_value

    def setDataWithLock(self,data,term,loss_value):
        with self.lock:
            self.flatten_tensor = data
            self.term = term
            self.loss_value = loss_value

class RootModelAvgRPCCommunicator:
    def __init__(self, rank, model: torch.nn.Module ,rpc_communicator: 'KFacRPCCommunicator'):
        self.rpc_communicator: 'KFacRPCCommunicator' = rpc_communicator
        self.world_size_cb = rpc_communicator.get_world_size
        self.current_t_cb = self.rpc_communicator.current_t
        self.origin_world_size = rpc_communicator.origin_world_size
        self.rank = rank
        self.model = model
        self.local_model_store = ModelStore(model2flatten_tensor(self.model))

    def set_loss(self, loss_value):
        self.local_model_store.loss_value = loss_value

    def get_local_node_speed(self):
        if self.rpc_communicator.node_states[self.rank].speed is not None and self.rpc_communicator.node_states[self.rank].speed != 0:
            return self.rpc_communicator.node_states[self.rank].speed
        elif self.rpc_communicator.time_cost_accumulation != 0:
            self.rpc_communicator.node_states[self.rank].speed = int(self.rpc_communicator.computation_volume_accumulation / self.rpc_communicator.time_cost_accumulation)
            return self.rpc_communicator.node_states[self.rank].speed
        return None

    def update_local_flat_model(self):
        with self.local_model_store.lock:
            self.local_model_store.flatten_tensor = model2flatten_tensor(self.model)
            self.local_model_store.term = self.current_t_cb()

def compute_recv_weight_by_loss(local_loss, recv_loss):
    sigmoid_param = (local_loss - recv_loss) / local_loss
    return expit(sigmoid_param*0.7)