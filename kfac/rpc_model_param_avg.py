import random
import threading
from typing import Dict
from typing import Callable
import torch
from scipy.special import expit
from torch.distributed import rpc
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from kfac.rpc_distributed import KFacRPCCommunicator

def rpc_work_name(rank:int) -> str:
    return f"rpc_{rank}"

class LayerParameterStore:
    def __init__(self):
        self.layer_name = ""
        self.weight = None
        self.bias = None

class ModelAvgRPCCommunicator:
    loss_weight = 1
    iter_weight = 0.5

    def __init__(self,  rank, model,rpc_communicator: 'KFacRPCCommunicator'):
        self.world_size = rpc_communicator.get_world_size
        self.rank = rank
        self.model = model
        self.io_layers = self.register_layer(model)
        self.rpc_communicator: 'KFacRPCCommunicator' = rpc_communicator
        self.start_target = (self.rank + 1 ) % self.world_size()
        self.loss_value = 0
        self.lock = threading.Lock()
        random.seed((rank+13)*17)

        global model_avg_rpc_communicator
        model_avg_rpc_communicator = self

    def set_loss(self, loss_value):
        self.loss_value = loss_value

    def register_layer(self, model):
        io_layers : Dict[str:LayerParameterStore] = {}
        with torch.no_grad():
            for layer_name, module in model.named_modules():
                if len(list(module.children())) != 0 or len(list(module.named_parameters())) == 0:
                    continue
                io_layers[layer_name] = LayerParameterStore()
                io_layers[layer_name].layer_name = layer_name
                for name, param in module.named_parameters():
                    if name.endswith("weight"):
                        io_layers[layer_name].weight = param
                    if name.endswith("bias"):
                        io_layers[layer_name].bias = param
        return io_layers

    def send_model_param(self, target, layer_name, weight, bias):
        with torch.no_grad():
            if target == self.rank:
                return
            try:
                rpc.rpc_async(
                    to=rpc_work_name(target),
                    func=receive_model_param,
                    args=(self.rank,self.rpc_communicator.current_t(),self.loss_value, layer_name,weight, bias)
                )
            except Exception as e:
                print(f"send_model_param failed {e} from {self.rank} to {target}")

    def send_all_model_param_alg01(self):
        target = self.start_target
        for layer_name, layer in self.io_layers.items():
            if target == self.rank:
                target = (target + 1) % self.world_size()
            self.send_model_param(target, layer_name, layer.weight, layer.bias)
            target = (target + 1) % self.world_size()
        self.start_target = (self.start_target + 1) % self.world_size()

    def send_all_model_param_alg02(self):
        target_chioce_list = set([state.rank for state in self.rpc_communicator.get_health_node_state_list()])
        if self.rank in target_chioce_list:
            target_chioce_list.remove(self.rank)
        target_chioce_list = list(target_chioce_list)
        for layer_name, layer in self.io_layers.items():
            target = random.choice(target_chioce_list)
            self.send_model_param(target, layer_name, layer.weight, layer.bias)

model_avg_rpc_communicator: ModelAvgRPCCommunicator

def receive_model_param(from_rank, from_rank_iter,from_loss,layer_name, weight, bias):
    global model_avg_rpc_communicator
    model_avg_rpc_communicator.rpc_communicator.update_other_rank_iter(from_rank,from_rank_iter)

    sigmoid_param = ModelAvgRPCCommunicator.loss_weight * (model_avg_rpc_communicator.loss_value - from_loss) / model_avg_rpc_communicator.loss_value \
                    + ModelAvgRPCCommunicator.iter_weight * (from_rank_iter - model_avg_rpc_communicator.rpc_communicator.current_t()) / from_rank_iter

    recv_weight = expit(sigmoid_param)
    with torch.no_grad():
        if not model_avg_rpc_communicator.lock.acquire(timeout= 5):
            raise Exception("lock acquire failed")
        model_avg_rpc_communicator.io_layers[layer_name].weight = model_avg_rpc_communicator.io_layers[layer_name].weight * (1-recv_weight) + weight * recv_weight
        model_avg_rpc_communicator.io_layers[layer_name].bias = model_avg_rpc_communicator.io_layers[layer_name].bias * (1-recv_weight)  + bias * recv_weight
        model_avg_rpc_communicator.rpc_communicator.print_rpc_state(f"receive_model_param from {from_rank} at {model_avg_rpc_communicator.rpc_communicator.current_t()} ,weight {recv_weight} , sigmoid_param {sigmoid_param}")
        model_avg_rpc_communicator.lock.release()
