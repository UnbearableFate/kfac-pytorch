import threading
from typing import Dict
from typing import Callable
import torch
from torch.distributed import rpc
def rpc_work_name(rank:int) -> str:
    return f"rpc_{rank}"

class LayerParameterStore:
    def __init__(self):
        self.layer_name = ""
        self.weight = None
        self.bias = None

class ModelAvgRPCCommunicator:
    def __init__(self, world_size, rank, model, current_t:Callable):
        self.world_size = world_size
        self.rank = rank
        self.model = model
        self.io_layers = self.register_layer(model)
        self.current_t:Callable = current_t
        self.start_target = (self.rank + 1 ) % world_size
        self.lock = threading.Lock()

        global model_avg_rpc_communicator
        model_avg_rpc_communicator = self

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
            rpc.rpc_async(
                to=rpc_work_name(target),
                func=receive_model_param,
                args=(self.rank, layer_name,weight, bias)
            )

    def send_all_model_param(self):
        target = self.start_target
        for layer_name, layer in self.io_layers.items():
            if target == self.rank:
                target = (target + 1) % self.world_size
            with self.lock:
                self.send_model_param(target, layer_name, layer.weight, layer.bias)
            target = (target + 1) % self.world_size
        self.start_target = (self.start_target + 1) % self.world_size


model_avg_rpc_communicator: ModelAvgRPCCommunicator

def receive_model_param(from_rank, layer_name, weight, bias):
    world_size = model_avg_rpc_communicator.world_size
    with (torch.no_grad() and model_avg_rpc_communicator.lock):
        model_avg_rpc_communicator.io_layers[layer_name].weight = model_avg_rpc_communicator.io_layers[layer_name].weight * 0.5 + weight * 0.5
        model_avg_rpc_communicator.io_layers[layer_name].bias = model_avg_rpc_communicator.io_layers[layer_name].bias *0.5 + bias * 0.5
