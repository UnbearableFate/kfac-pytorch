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
        self.term = 0
        self.weight = None
        self.bias = None

class ModelAvgRPCCommunicator:
    loss_weight = 1.5
    iter_weight = 0.7

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

    def send_model_param(self, target, layer_name ,resurrection_flag = False):
        with torch.no_grad():
            if target == self.rank:
                return
            weight = self.io_layers[layer_name].weight
            bias = self.io_layers[layer_name].bias
            try:
                rpc.rpc_async(
                    to=rpc_work_name(target),
                    func=receive_model_param,
                    args=(self.rank,self.rpc_communicator.current_t(),self.loss_value, layer_name,weight, bias ,resurrection_flag)
                )
            except Exception as e:
                print(f"send_model_param failed {e} from {self.rank} to {target}")

    def send_to_sick_nodes_sometimes(self):
        if random.random() < 0.2:
            target_chioce_list = set([state.rank for state in self.rpc_communicator.get_sick_node_list()])
            if len(target_chioce_list) == 0:
                return
            if self.rank in target_chioce_list:
                target_chioce_list.remove(self.rank)
            target_chioce_list = list(target_chioce_list)
            for layer_name, layer in self.io_layers.items():
                target = random.choice(target_chioce_list)
                self.send_model_param(target, layer_name)

    def send_all_model_param_alg01(self):
        target = self.start_target
        for layer_name, layer in self.io_layers.items():
            if target == self.rank:
                target = (target + 1) % self.world_size()
            self.send_model_param(target, layer_name)
            target = (target + 1) % self.world_size()
        self.start_target = (self.start_target + 1) % self.world_size()

    def send_all_model_param_alg02(self):
        target_chioce_list = set([state.rank for state in self.rpc_communicator.get_health_node_state_list()])
        if self.rank in target_chioce_list:
            target_chioce_list.remove(self.rank)
        target_chioce_list = list(target_chioce_list)
        for layer_name, layer in self.io_layers.items():
            target = random.choice(target_chioce_list)
            self.send_model_param(target, layer_name)
        #self.send_to_sick_nodes_sometimes()

    def Send_to_Easter_Point_Task_Assignment(self,health_node_list):
        result = dict()
        for layer_name, layer in self.io_layers.items():
            from_rank = random.choice(health_node_list)
            if from_rank not in result.keys():
                result[from_rank] = list()
            result[from_rank].append(layer_name)
        return result

model_avg_rpc_communicator: ModelAvgRPCCommunicator

def receive_model_param(from_rank, from_rank_iter, from_loss, layer_name, model_weight, model_bias ,resurrection_flag = False):
    global model_avg_rpc_communicator
    model_avg_rpc_communicator.rpc_communicator.update_node_iter(from_rank, from_rank_iter)
    slow_tolerance_value = model_avg_rpc_communicator.rpc_communicator.slow_tolerance_value
    local_iter = model_avg_rpc_communicator.rpc_communicator.current_t()
    sigmoid_param = 0
    if model_avg_rpc_communicator.loss_value != 0:
        sigmoid_param = ModelAvgRPCCommunicator.loss_weight * (
                    model_avg_rpc_communicator.loss_value - from_loss) / model_avg_rpc_communicator.loss_value

    sigmoid_param += ModelAvgRPCCommunicator.iter_weight * pow(
        (from_rank_iter - local_iter ) / slow_tolerance_value *3  ,3)

    recv_weight = expit(sigmoid_param)

    with torch.no_grad():
        if not model_avg_rpc_communicator.lock.acquire(timeout= 1):
            raise Exception("lock acquire failed")
        model_avg_rpc_communicator.io_layers[layer_name].weight = model_avg_rpc_communicator.io_layers[layer_name].weight * (1-recv_weight) + model_weight * recv_weight
        model_avg_rpc_communicator.io_layers[layer_name].bias = model_avg_rpc_communicator.io_layers[layer_name].bias * (1-recv_weight) + model_bias * recv_weight
        model_avg_rpc_communicator.io_layers[layer_name].term = max(model_avg_rpc_communicator.io_layers[layer_name].term, from_rank_iter, local_iter)
        model_avg_rpc_communicator.lock.release()
    if resurrection_flag:
        model_avg_rpc_communicator.rpc_communicator.update_node_iter(model_avg_rpc_communicator.rank,
                                                                     from_rank_iter)

