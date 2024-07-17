import random
import threading
import time
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

def compute_skip(world_size):
    p = world_size
    q = 0
    while p > 1:
        q = q+1
        p -= int(p/2)
    skips = [0 for _ in range(0,q+1)]
    skips[q] = world_size
    for i in range(q-1, -1, -1):
        skips[i] = skips[i+1]-int(skips[i+1]/2)
    return skips

class LayerParameterStore:
    def __init__(self):
        self.layer_name = ""
        self.term = 0
        self.weight = None
        self.bias = None
        self.loss_value = 0

class ModelAvgRPCCommunicator:
    loss_weight = 1.8
    iter_weight = 0.7

    def __init__(self,  rank, model,rpc_communicator: 'KFacRPCCommunicator'):
        self.world_size = rpc_communicator.get_world_size
        self.origin_world_size = rpc_communicator.get_world_size()
        self.rank = rank
        self.model = model
        self.io_layers = self.register_layer(model)
        self.rpc_communicator: 'KFacRPCCommunicator' = rpc_communicator
        self.start_target = (self.rank + 1 ) % self.world_size()
        self.loss_value = 0
        self.lock = threading.Lock()
        random.seed((rank+13)*17)
        self.skips = compute_skip(self.world_size())[:-1]
        self.skip_index = 0

        self.neighbor_model_store :Dict[int,Dict[str, LayerParameterStore]] = dict()
        self.neighbor_weight_store : Dict[int, float] = dict()
        self.init_neighbor_model_store()

        global model_avg_rpc_communicator
        model_avg_rpc_communicator = self

    def set_loss(self, loss_value):
        self.loss_value = loss_value

    def register_layer(self, model) -> Dict[str,LayerParameterStore]:
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
    
    def init_neighbor_model_store(self):
        for node in self.rpc_communicator.get_health_nodes_rank_list():
            if node == self.rank:
                continue
            self.neighbor_model_store[node] = dict()
            for layer_name in self.io_layers.keys():
                self.neighbor_model_store[node][layer_name] = LayerParameterStore()
                self.neighbor_model_store[node][layer_name].weight = torch.zeros_like(self.io_layers[layer_name].weight)
                if self.io_layers[layer_name].bias is not None:
                    self.neighbor_model_store[node][layer_name].bias = torch.zeros_like(self.io_layers[layer_name].bias)

    """
    def store_model_param(self,model):
        if not self.lock.acquire(timeout= 1):
            raise Exception("lock acquire failed at store_model_param") 
        with torch.no_grad():
            for layer_name, module in model.named_modules():
                if len(list(module.children())) != 0 or len(list(module.named_parameters())) == 0:
                    continue
                for name, param in module.named_parameters():
                    if name.endswith("weight"):
                        self.io_layers[layer_name].weight = param
                    if name.endswith("bias"):
                        self.io_layers[layer_name].bias = param
        self.lock.release()

    
    def load_model_param(self,model):
        if not self.lock.acquire(timeout= 1):
            raise Exception("lock acquire failed at load_model_param") 
        with torch.no_grad():
            for layer_name, module in model.named_modules():
                if len(list(module.children())) != 0 or len(list(module.named_parameters())) == 0:
                    continue
                for name, param in module.named_parameters():
                    if name.endswith("weight"):
                        param.copy_(self.io_layers[layer_name].weight)
                    if name.endswith("bias"):
                        param.copy_(self.io_layers[layer_name].bias)
        self.lock.release()
    """
    def send_model_param(self, target, layer_name ,resurrection_flag = False ,speed = None):
        with torch.no_grad():
            if target == self.rank:
                return
            weight = self.io_layers[layer_name].weight
            bias = self.io_layers[layer_name].bias
            try:
                rpc.rpc_async(
                    to=rpc_work_name(target),
                    func=receive_model_param,
                    args=(self.rank,self.rpc_communicator.current_t(),self.loss_value, layer_name,weight, bias ,resurrection_flag ,speed)
                )
            except Exception as e:
                print(f"send_model_param failed {e} from {self.rank} to {target}")
    
    def send_model_param_to_store(self, target, layer_name,speed= None):
        with torch.no_grad():
            if target == self.rank:
                return
            weight = self.io_layers[layer_name].weight
            bias = self.io_layers[layer_name].bias
            try:
                rpc.rpc_async(
                    to=rpc_work_name(target),
                    func=receive_model_param_to_store,
                    args=(self.rank,self.rpc_communicator.current_t(),self.loss_value, layer_name,weight, bias )
                )
            except Exception as e:
                print(f"send_model_param failed {e} from {self.rank} to {target}")
    
    def send_model_param_dict_to_store(self, target, layer_names = None ,speed = None):
        if layer_names is None:
            layer_names = self.io_layers.keys()

        data = []
        for layer_name in layer_names:
            weight = self.io_layers[layer_name].weight
            bias = self.io_layers[layer_name].bias
            data.append((layer_name, weight, bias))

        try:
            rpc.rpc_async(
                to=rpc_work_name(target),
                func=receive_model_param_dict_to_store,
                args=(self.rank,self.rpc_communicator.current_t(),self.loss_value, data )
            )
        except Exception as e:
            print(f"send_model_param failed {e} from {self.rank} to {target}")

    def get_local_node_speed(self):
        if self.rpc_communicator.time_cost_accumulation != 0:
            speed = self.rpc_communicator.computation_volume_accumulation / self.rpc_communicator.time_cost_accumulation
            self.rpc_communicator.node_states[self.rank].speed = speed
            return speed
        return None

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
                self.send_model_param(target, layer_name,False,self.get_local_node_speed())

    def send_all_model_param_alg01(self):
        target = self.start_target
        for layer_name, layer in self.io_layers.items():
            if target == self.rank:
                target = (target + 1) % self.world_size()
            self.send_model_param(target, layer_name, False, self.get_local_node_speed())
            target = (target + 1) % self.world_size()
        self.start_target = (self.start_target + 1) % self.world_size()

    def send_all_model_param_alg02(self):
        target_chioce_list = set([state.rank for state in self.rpc_communicator.get_health_node_state_list()])
        if self.rank in target_chioce_list:
            target_chioce_list.remove(self.rank)
        target_chioce_list = list(target_chioce_list)
        for layer_name, layer in self.io_layers.items():
            target = random.choice(target_chioce_list)
            self.send_model_param(target, layer_name,False,self.get_local_node_speed())
        #self.send_to_sick_nodes_sometimes()
    
    def send_all_model_param_alg03(self):
        target_chioce_list = [state.rank for state in self.rpc_communicator.get_health_node_state_list()]
        for rank in target_chioce_list:
            if rank == self.rank:
                continue
            self.send_model_param_dict_to_store(rank ,layer_names=None,speed = self.get_local_node_speed())
        
        self.average_model_param_from_store()

    def send_all_model_param_alg05(self):
        target_chioce_list = [state.rank for state in self.rpc_communicator.get_health_node_state_list()]
        for rank in target_chioce_list:
            if rank == self.rank:
                continue
            for layer_name, layer in self.io_layers.items():
                self.send_model_param_to_store(rank, layer_name ,speed= self.get_local_node_speed())
        self.average_model_param_from_store()

    def send_all_model_param_alg06(self):
        target_chioce_list = [state.rank for state in self.rpc_communicator.get_health_node_state_list()]
        for rank in target_chioce_list:
            if rank == self.rank:
                continue
            self.send_model_param_dict_to_store(rank, layer_names=None, speed=self.get_local_node_speed())

        self.average_model_param_from_store()
    
    def average_model_param_from_store(self):
        with torch.no_grad():
            if not model_avg_rpc_communicator.lock.acquire(timeout= 3):
                raise Exception("lock acquire failed at average_model_param_from_store")
            for layer_name, layer in self.io_layers.items():
                weight = torch.zeros_like(layer.weight)
                bias = torch.zeros_like(layer.bias)
                local_w = self.origin_world_size
                for node, layer_dict in self.neighbor_model_store.items():
                    sigmoid_param = 0
                    if model_avg_rpc_communicator.loss_value != 0:
                        sigmoid_param = ModelAvgRPCCommunicator.loss_weight * (
                    self.loss_value - layer_dict[layer_name].loss_value) / self.loss_value
                    recv_w = expit(sigmoid_param)
                    local_w -= recv_w
                    weight += layer_dict[layer_name].weight * recv_w
                    bias += layer_dict[layer_name].bias * recv_w
                weight += layer.weight * local_w
                bias += layer.bias * local_w
                weight /= self.origin_world_size
                bias /= self.origin_world_size
                layer.weight = weight
                layer.bias = bias
            model_avg_rpc_communicator.lock.release()

    def average_model_param_from_store2(self):
        with torch.no_grad():
            if not model_avg_rpc_communicator.lock.acquire(timeout= 3):
                raise Exception("lock acquire failed at average_model_param_from_store")
            loss_sum = 0
            for node, layer_dict in self.neighbor_model_store.items():
                temp = 0
                n = 0
                for layer_name, layer in layer_dict.items():
                    temp += layer.loss_value
                    n+=1
                self.neighbor_weight_store[node] = n/temp
                loss_sum += n/temp

            self.neighbor_weight_store[self.rank] = 1 / self.loss_value
            loss_sum += 1 / self.loss_value

            for rank, weight in self.neighbor_weight_store.items():
                self.neighbor_weight_store[rank] = weight / loss_sum

            for layer_name, layer in self.io_layers.items():
                weight = torch.clone(layer.weight).mul_(self.neighbor_weight_store[self.rank])
                bias = torch.zeros_like(layer.bias).mul_(self.neighbor_weight_store[self.rank])

                for node, layer_dict in self.neighbor_model_store.items():
                    weight += layer_dict[layer_name].weight * self.neighbor_weight_store[node]
                    bias += layer_dict[layer_name].bias * self.neighbor_weight_store[node]
                layer.weight = weight
                layer.bias = bias
            model_avg_rpc_communicator.lock.release()
        
    def Send_to_Easter_Point_Task_Assignment(self,health_node_list):
        result = dict()
        for layer_name, layer in self.io_layers.items():
            from_rank = random.choice(health_node_list)
            if from_rank not in result.keys():
                result[from_rank] = list()
            result[from_rank].append(layer_name)
        return result

model_avg_rpc_communicator: ModelAvgRPCCommunicator

def receive_model_param(from_rank, from_rank_iter, from_loss, layer_name, model_weight, model_bias ,resurrection_flag = False ,speed = None):
    global model_avg_rpc_communicator
    model_avg_rpc_communicator.rpc_communicator.update_node_iter(from_rank, from_rank_iter,speed= speed)
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
        if not model_avg_rpc_communicator.lock.acquire(timeout= 3):
            raise Exception("lock acquire failed")
        model_avg_rpc_communicator.io_layers[layer_name].weight = model_avg_rpc_communicator.io_layers[layer_name].weight * (1-recv_weight) + model_weight * recv_weight
        model_avg_rpc_communicator.io_layers[layer_name].bias = model_avg_rpc_communicator.io_layers[layer_name].bias * (1-recv_weight) + model_bias * recv_weight
        model_avg_rpc_communicator.io_layers[layer_name].term = max(model_avg_rpc_communicator.io_layers[layer_name].term, from_rank_iter, local_iter)
        model_avg_rpc_communicator.lock.release()
    if resurrection_flag:
        model_avg_rpc_communicator.rpc_communicator.update_node_iter(model_avg_rpc_communicator.rank,
                                                                     from_rank_iter)

def receive_model_param_to_store(from_rank, from_rank_iter, from_loss, layer_name, model_weight, model_bias ,speed = None):
    global model_avg_rpc_communicator
    if from_rank == model_avg_rpc_communicator.rank or from_rank < model_avg_rpc_communicator.rpc_communicator.node_states[from_rank].rank:
        return
    model_avg_rpc_communicator.rpc_communicator.update_node_iter(from_rank, from_rank_iter ,speed= speed)
    model_avg_rpc_communicator.neighbor_model_store[from_rank][layer_name].weight = model_weight
    model_avg_rpc_communicator.neighbor_model_store[from_rank][layer_name].bias = model_bias
    model_avg_rpc_communicator.neighbor_model_store[from_rank][layer_name].term = from_rank_iter
    model_avg_rpc_communicator.neighbor_model_store[from_rank][layer_name].loss_value = from_loss

def receive_model_param_dict_to_store(from_rank, from_rank_iter, from_loss, data ,speed = 0):
    global model_avg_rpc_communicator
    model_avg_rpc_communicator.rpc_communicator.update_node_iter(from_rank, from_rank_iter ,speed= speed)
    for layer_name, model_weight, model_bias in data:
        model_avg_rpc_communicator.neighbor_model_store[from_rank][layer_name].weight = model_weight
        model_avg_rpc_communicator.neighbor_model_store[from_rank][layer_name].bias = model_bias
        model_avg_rpc_communicator.neighbor_model_store[from_rank][layer_name].term = from_rank_iter
        model_avg_rpc_communicator.neighbor_model_store[from_rank][layer_name].loss_value = from_loss