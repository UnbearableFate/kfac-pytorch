import gc
import random
import threading
import time
from typing import Dict,List
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

def compute_recv_weight_by_loss(local_loss, recv_loss):
    sigmoid_param = local_loss - recv_loss / local_loss
    return expit(sigmoid_param*0.3)

class LayerParameterStore:
    def __init__(self):
        self.layer_name = ""
        self.term = 0
        self.weight = None
        self.bias = None
        self.loss_value = 0

    def set_model_content(self,weight, bias, loss_value ):
        self.weight = weight
        self.bias = bias
        self.loss_value = loss_value

    def aggregate_model_content_by_loss(self,weight, bias, loss_value, iter = None):
        if self.loss_value == 0 or self.weight is None:
            self.set_model_content(weight,bias,loss_value)
            return
        recv_weight = compute_recv_weight_by_loss(self.loss_value, loss_value)
        self.weight = self.weight* (1-recv_weight) + weight * recv_weight
        if self.bias is not None:
            self.bias = self.bias * (1-recv_weight) + bias * recv_weight
        self.loss_value = loss_value * (1-recv_weight) * loss_value * recv_weight
        if iter is not None:
            self.term = max(self.term, iter)

class ModelStore:
    def __init__(self ,io_layers:Dict[str,LayerParameterStore] = None):
        self.layer_store:Dict[str, LayerParameterStore] = dict()
        self.term = 0
        self.loss_value = 0
        self.lock = threading.Lock()
        self.resurrection_flag = False
        if io_layers is not None:
            for layer_name in io_layers.keys():
                self.layer_store[layer_name] = LayerParameterStore()
                self.layer_store[layer_name].weight = torch.zeros_like(io_layers[layer_name].weight)
                if  io_layers[layer_name].bias is not None:
                    self.layer_store[layer_name].bias = torch.zeros_like(io_layers[layer_name].bias)


class ModelAvgRPCCommunicator:
    loss_weight = 1.8
    iter_weight = 0.7

    def __init__(self,  rank, model,rpc_communicator: 'KFacRPCCommunicator', style = 'buffer'):
        self.world_size = rpc_communicator.get_world_size
        self.origin_world_size = rpc_communicator.origin_world_size
        self.rank = rank
        self.model = model
        self.io_layers = self.register_layer(model)
        self.rpc_communicator: 'KFacRPCCommunicator' = rpc_communicator
        self.current_t = self.rpc_communicator.current_t
        self.start_target = (self.rank + 1 ) % self.world_size()
        self.loss_value = 0
        self.lock = threading.Lock()
        random.seed((rank+13)*17)
        self.skips = compute_skip(self.world_size())[:-1]
        self.skip_index = 0


        self.neighbor_model_store :Dict[int,Dict[str, LayerParameterStore]] = dict()
        self.neighbor_weight_store : Dict[int, float] = dict()
        #self.init_neighbor_model_store()

        self.buffer_size = 2
        self.model_recv_buffer :List[ModelStore]= self.creat_model_recv_buffer(self.buffer_size)

        if style == 'buffer':
            self.send_func = self.send_model_param_to_buffer

        self.index = 1
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

    def creat_model_recv_buffer(self, size) -> List[ModelStore] :
        buffer: List[ModelStore] = []
        for i in range(size):
            buffer.append(ModelStore(self.io_layers))
        return buffer

    def send_model_param(self, target, layer_name ,resurrection_flag = False):
        speed = self.get_local_node_speed()
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
        gc.collect()
    
    def send_model_param_to_store(self, target, layer_name):
        speed = self.get_local_node_speed()
        with torch.no_grad():
            if target == self.rank:
                return
            weight = self.io_layers[layer_name].weight
            bias = self.io_layers[layer_name].bias
            try:
                rpc.rpc_async(
                    to=rpc_work_name(target),
                    func=receive_model_param_to_neighbor_store,
                    args=(self.rank,self.rpc_communicator.current_t(),self.loss_value, layer_name,weight, bias )
                )
            except Exception as e:
                print(f"send_model_param_to_store failed {e} from {self.rank} to {target}")

    def send_model_param_dict_to_store(self, target, layer_names = None):
        speed = self.get_local_node_speed()
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
                func=receive_model_param_dict_to_neighbor_store,
                args=(self.rank,self.rpc_communicator.current_t(),self.loss_value, data ,speed)
            )
        except Exception as e:
            print(f"send_model_param_dict_to_store failed {e} from {self.rank} to {target}")

    def send_model_param_to_buffer(self, target, layer_names = None, resurrection_flag = False):
        speed = self.get_local_node_speed()
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
                func=receive_model_param_dict_to_buffer,
                args=(self.rank,self.rpc_communicator.current_t(),self.loss_value, data ,speed, resurrection_flag)
            )
        except Exception as e:
            print(f"send_model_param_to_buffer failed {e} from {self.rank} to {target}")
        gc.collect()

    def get_local_node_speed(self):
        if self.rpc_communicator.node_states[self.rank].speed is not None and self.rpc_communicator.node_states[self.rank].speed != 0:
            return self.rpc_communicator.node_states[self.rank].speed
        elif self.rpc_communicator.time_cost_accumulation != 0:
            self.rpc_communicator.node_states[self.rank].speed = int(self.rpc_communicator.computation_volume_accumulation / self.rpc_communicator.time_cost_accumulation)
            return self.rpc_communicator.node_states[self.rank].speed
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
                self.send_model_param(target, layer_name,False)

    def send_all_model_param_alg01(self):
        target = self.start_target
        for layer_name, layer in self.io_layers.items():
            if target == self.rank:
                target = (target + 1) % self.world_size()
            self.send_model_param(target, layer_name, False)
            target = (target + 1) % self.world_size()
        self.start_target = (self.start_target + 1) % self.world_size()

    def send_all_model_param_alg02(self):
        target_chioce_list = set([state.rank for state in self.rpc_communicator.get_health_node_state_list()])
        if self.rank in target_chioce_list:
            target_chioce_list.remove(self.rank)
        target_chioce_list = list(target_chioce_list)
        for layer_name, layer in self.io_layers.items():
            target = random.choice(target_chioce_list)
            self.send_model_param(target, layer_name,False)
        #self.send_to_sick_nodes_sometimes()
    
    def send_all_model_param_alg03(self):
        target_chioce_list = [state.rank for state in self.rpc_communicator.get_health_node_state_list()]
        for rank in target_chioce_list:
            if rank == self.rank:
                continue
            self.send_model_param_dict_to_store(rank ,layer_names=None)
        
        self.average_model_param_from_store()

    def send_all_model_param_alg06(self):
        target_chioce_list = [state.rank for state in self.rpc_communicator.get_health_node_state_list()]
        for rank in target_chioce_list:
            if rank == self.rank:
                continue
            self.send_model_param_dict_to_store(rank, layer_names=None)

        #self.average_model_param_from_store2()

    def send_all_model_param_alg07(self):
        target = (self.rank + self.index) % self.origin_world_size
        self.index = (self.index + 1) % self.origin_world_size
        if target == self.rank:
            target = (target + 1) % self.origin_world_size
            self.index = (self.index + 1) % self.origin_world_size
        self.send_model_param_to_buffer(target, layer_names=None)
        self.aggregate_model_from_buff()
    
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
        if self.loss_value == 0:
            return
        loss_sum = 0
        for node, layer_dict in self.neighbor_model_store.items():
            temp = 0
            n = 0
            for layer_name, layer in layer_dict.items():
                temp += layer.loss_value
                n+=1
            if temp == 0:
                self.neighbor_weight_store[node] = 0
            else:
                self.neighbor_weight_store[node] = n/temp
                loss_sum += n/temp

        self.neighbor_weight_store[self.rank] = 1 / self.loss_value
        loss_sum += 1 / self.loss_value

        for rank, weight in self.neighbor_weight_store.items():
            self.neighbor_weight_store[rank] = weight / loss_sum
        
        with torch.no_grad():
            if not model_avg_rpc_communicator.lock.acquire(timeout= 3):
                raise Exception("lock acquire failed at average_model_param_from_store")
            
            for layer_name, layer in self.io_layers.items():
                weight = torch.clone(layer.weight).mul_(self.neighbor_weight_store[self.rank])
                bias = None
                if layer.bias is not None:
                    bias = torch.clone(layer.bias).mul_(self.neighbor_weight_store[self.rank])

                for node, layer_dict in self.neighbor_model_store.items():
                    weight += layer_dict[layer_name].weight * self.neighbor_weight_store[node]
                    if bias is not None:
                        bias += layer_dict[layer_name].bias * self.neighbor_weight_store[node]
                layer.weight = weight
                layer.bias = bias
            model_avg_rpc_communicator.lock.release()

    def aggregate_model_from_buff(self):
        p = 1/ (self.buffer_size+1)
        temp = ModelStore(self.io_layers)
        for buf_model in self.model_recv_buffer:
            if not buf_model.lock.acquire(timeout= 3):
                raise Exception("lock acquire failed at aggregate_model_from_buff")
            for layer_name, layer in buf_model.layer_store.items():
                temp.layer_store[layer_name].weight += layer.weight * p
                if layer.bias is not None:
                    temp.layer_store[layer_name].bias += layer.bias * p
            temp.term += buf_model.term
            if buf_model.resurrection_flag:
                temp.resurrection_flag = True
            buf_model.resurrection_flag = False
            buf_model.lock.release()
        temp.term = int(temp.term / self.buffer_size)

        if not self.lock.acquire(timeout= 3):
            raise Exception("lock acquire failed at aggregate_model_from_buff")
        with torch.no_grad(): 
            for layer_name, layer in self.io_layers.items():
                layer.weight = layer.weight*p + temp.layer_store[layer_name].weight
                if layer.bias is not None:
                    layer.bias = layer.bias*p + temp.layer_store[layer_name].bias
        self.lock.release()
        if temp.resurrection_flag:
            self.rpc_communicator.update_node_iter(self.rank, temp.term)
            self.rpc_communicator.print_rpc_state(f"resurrection in {self.rank} to {temp.term}") 
        gc.collect()
        
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
                                                                     from_rank_iter ,speed)

def receive_model_param_to_neighbor_store(from_rank, from_rank_iter, from_loss, layer_name, model_weight, model_bias, speed = None):
    global model_avg_rpc_communicator
    if from_rank == model_avg_rpc_communicator.rank or from_rank < model_avg_rpc_communicator.rpc_communicator.node_states[from_rank].rank:
        return
    model_avg_rpc_communicator.rpc_communicator.update_node_iter(from_rank, from_rank_iter ,speed= speed)
    model_avg_rpc_communicator.neighbor_model_store[from_rank][layer_name].weight = model_weight
    model_avg_rpc_communicator.neighbor_model_store[from_rank][layer_name].bias = model_bias
    model_avg_rpc_communicator.neighbor_model_store[from_rank][layer_name].term = from_rank_iter
    model_avg_rpc_communicator.neighbor_model_store[from_rank][layer_name].loss_value = from_loss

def receive_model_param_dict_to_neighbor_store(from_rank, from_rank_iter, from_loss, data, speed = 0):
    global model_avg_rpc_communicator
    model_avg_rpc_communicator.rpc_communicator.update_node_iter(from_rank, from_rank_iter ,speed= speed)
    for layer_name, model_weight, model_bias in data:
        model_avg_rpc_communicator.neighbor_model_store[from_rank][layer_name].weight = model_weight
        model_avg_rpc_communicator.neighbor_model_store[from_rank][layer_name].bias = model_bias
        model_avg_rpc_communicator.neighbor_model_store[from_rank][layer_name].term = from_rank_iter
        model_avg_rpc_communicator.neighbor_model_store[from_rank][layer_name].loss_value = from_loss

def receive_model_param_dict_to_buffer(from_rank, from_rank_iter, from_loss, data, speed = 0, resurrection_flag = False):
    global model_avg_rpc_communicator
    model_avg_rpc_communicator.rpc_communicator.update_node_iter(from_rank, from_rank_iter, speed=speed)
    indices = list(range(model_avg_rpc_communicator.buffer_size))  # 创建包含范围内所有数值的列表
    random.shuffle(indices)
    for i in indices:
        if model_avg_rpc_communicator.model_recv_buffer[i].lock.acquire(blocking=False):
            try:
                for layer_name, model_weight, model_bias in data:
                    model_avg_rpc_communicator.model_recv_buffer[i].layer_store[layer_name].aggregate_model_content_by_loss(model_weight, model_bias,from_loss)
                    model_avg_rpc_communicator.model_recv_buffer[i].term = max(model_avg_rpc_communicator.model_recv_buffer[i].term, from_rank_iter)
                    if resurrection_flag:
                        model_avg_rpc_communicator.model_recv_buffer[i].resurrection_flag = True
            finally:
                # 释放锁
                model_avg_rpc_communicator.model_recv_buffer[i].lock.release()
                break
        else:
            continue
    gc.collect()

