import random
import threading
from typing import Dict,List
from typing import Callable
import torch
from scipy.special import expit
from torch.distributed import rpc
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from kfac.rpc_distributed import KFacRPCCommunicator
import math

def get_grid_dimensions(N):
    """
    Compute the grid dimensions (rows and cols) such that:
    - The grid is as close to a square as possible.
    - The difference between rows and cols is at most 1.
    - The grid can accommodate all N nodes, possibly with some empty slots.
    """
    sqrt_N = math.sqrt(N)
    # Possible rows: floor and ceil of sqrt(N)
    possible_rows = [int(math.floor(sqrt_N)), int(math.ceil(sqrt_N))]
    possible_rows = list(set(possible_rows))  # Ensure uniqueness

    # Try to find the best grid dimensions
    best_rows, best_cols = None, None
    min_difference = None

    for rows in possible_rows:
        if rows <= 0:
            continue
        cols = int(math.ceil(N / rows))
        if abs(rows - cols) <= 1:
            if min_difference is None or abs(rows - cols) < min_difference:
                min_difference = abs(rows - cols)
                best_rows, best_cols = rows, cols

    # If no acceptable grid found, increment rows until acceptable
    if best_rows is None:
        rows = int(math.floor(sqrt_N))
        while True:
            cols = int(math.ceil(N / rows))
            if abs(rows - cols) <= 1:
                best_rows, best_cols = rows, cols
                break
            rows += 1

    return best_rows, best_cols

def get_down_and_right_neighbors(origin_world_size, local_rank):
    N = origin_world_size

    # Step 1: Compute grid dimensions (rows and cols)
    rows, cols = get_grid_dimensions(N)

    # Step 2: Compute the current node's position in the grid
    row = local_rank // cols
    col = local_rank % cols

    # Initialize down and right neighbor ranks
    down_rank = None
    right_rank = None

    # Step 3: Compute the down neighbor with wrap-around
    down_row = (row + 1) % rows
    down_col = col
    while True:
        down_rank_candidate = down_row * cols + down_col
        if down_rank_candidate < N:
            down_rank = down_rank_candidate
            break
        else:
            # Move to the next row (wrap-around)
            down_row = (down_row + 1) % rows
            if down_row == row:
                # No valid neighbor found after a full loop
                down_rank = local_rank  # Self-loop if necessary
                break

    # Step 4: Compute the right neighbor with wrap-around
    right_row = row
    right_col = (col + 1) % cols
    while True:
        right_rank_candidate = right_row * cols + right_col
        if right_rank_candidate < N:
            right_rank = right_rank_candidate
            break
        else:
            # Move to the next column (wrap-around)
            right_col = (right_col + 1) % cols
            if right_col == col:
                # No valid neighbor found after a full loop
                right_rank = local_rank  # Self-loop if necessary
                break

    return down_rank, right_rank

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
        self.size = 0

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

    def __init__(self, rank, model,rpc_communicator: 'KFacRPCCommunicator'):

        self.send_rank_group = rpc_communicator.send_rank_group
        self.group_id = rpc_communicator.group_id
        self.world_size = rpc_communicator.get_world_size
        self.origin_world_size = rpc_communicator.origin_world_size
        self.rank = rank
        self.model = model
        self.io_layers = self.register_layer(model)
        self.model_send_packages = self.compute_layer_split()
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
        self.is_aggregated = [False for _ in range(self.buffer_size)]

        #if style == 'buffer':
        #    self.send_func = self.send_model_param_to_buffer

        self.index = 1

        self.split_packages = self.initialize_layer_packages(send_targets_num= self.origin_world_size - 1)
        self.down_neighbor, self.right_neighbor = get_down_and_right_neighbors(self.origin_world_size, self.rank)

        global model_avg_rpc_communicator
        model_avg_rpc_communicator = self

    def initialize_layer_packages(self, send_targets_num=None):
        if send_targets_num is None:
            send_targets_num = self.origin_world_size - 1

        # Step 1: Compute sizes for each layer
        layers = list(self.io_layers.keys())
        layer_sizes = []
        for layer_name in layers:
            layer = self.io_layers[layer_name]
            weight_size = layer.weight.numel() if layer.weight is not None else 0
            bias_size = layer.bias.numel() if layer.bias is not None else 0
            total_size = weight_size + bias_size
            layer.size = total_size  # Update the size attribute
            layer_sizes.append((layer_name, total_size))

        # Step 2: Sort layers by size in descending order
        layer_sizes.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Initialize packages using a greedy algorithm
        num_packages = min(send_targets_num, len(layer_sizes))
        packages = [[] for _ in range(num_packages)]
        package_sizes = [0] * num_packages

        for layer_name, size in layer_sizes:
            # Assign the layer to the package with the smallest total size
            min_package_idx = package_sizes.index(min(package_sizes))
            packages[min_package_idx].append(layer_name)
            package_sizes[min_package_idx] += size

        # Step 4: Save the splitting plan
        return packages

    def compute_layer_split(self):
        eigen_tensor_packages = [[] for _ in range(4)]
        package_sizes = [0] * len(eigen_tensor_packages)  # 初始化每个包的总大小

        def calculate_tensor_size(tensor):
            if tensor is None:
                return 0
            return tensor.numel()  # 返回tensor的元素数量，可以根据需要改成字节数 tensor.element_size() * tensor.numel()

        # 逐层放入包中，使用贪心算法
        for layer_name ,layer in self.io_layers.items():
            # 计算这个layer的总大小
            package_size = (
                    calculate_tensor_size(layer.weight) +
                    calculate_tensor_size(layer.bias)
            )

            # 找到当前最小的包
            min_index = package_sizes.index(min(package_sizes))

            # 将当前layer放入最小的包中
            eigen_tensor_packages[min_index].append(layer_name)
            package_sizes[min_index] += package_size

        filtered_eigen_tensor_packages = [sublist for sublist in eigen_tensor_packages if sublist]

        return filtered_eigen_tensor_packages

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
        for node in self.send_rank_group[self.group_id]:
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
        with torch.no_grad():
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
        target = (self.rank + self.index) % self.origin_world_size
        self.index = (self.index + 1) % self.origin_world_size
        if target == self.rank:
            target = (target + 1) % self.origin_world_size
            self.index = (self.index + 1) % self.origin_world_size
        layers = random.choices(list(self.io_layers.keys()), k=1)
        self.send_model_param_to_buffer(target, layer_names=layers)
        self.aggregate_model_from_buff()

    def send_all_model_param_alg02(self):
        # Step 1: Randomly shuffle the layer names
        layers = list(self.io_layers.keys())
        random.shuffle(layers)

        # Step 2: Partition the layers into self.origin_world_size packages
        num_packages = self.origin_world_size -1
        packages = [[] for _ in range(num_packages)]

        for idx, layer_name in enumerate(layers):
            packages[idx % num_packages].append(layer_name)

        # Step 3: Send each package to a random target (excluding self.rank)
        for package in packages:
            if not package:
                continue  # Ignore empty packages
            target_candidates = list(range(self.origin_world_size))
            target_candidates.remove(self.rank)
            target = random.choice(target_candidates)
            self.send_model_param_to_buffer(target, layer_names=package)

        # Step 4: Aggregate the models from buffer
        self.aggregate_model_from_buff()

    def send_all_model_param_alg03(self):
        # Get the number of packages to send
        num_packages = len(self.split_packages)

        # Get list of possible targets excluding self.rank
        target_candidates = list(range(self.origin_world_size))
        target_candidates.remove(self.rank)

        # Randomly select targets for each package
        random.shuffle(target_candidates)
        targets = target_candidates[:num_packages]

        # Map each package to a target
        for package, target in zip(self.split_packages, targets):
            # Send the package to the target
            self.send_model_param_to_buffer(target, layer_names=package)

        # Aggregate models from buffer
        self.aggregate_model_from_buff()

    def send_all_model_param_alg04(self):
        self.index += 1
        if self.index % 2 == 0:
            if self.right_neighbor:
                layers = random.choices(list(self.io_layers.keys()), k=4)
                self.send_model_param_to_buffer(self.right_neighbor) 
        else:
            if self.down_neighbor:
                layers = random.choices(list(self.io_layers.keys()), k=4)
                self.send_model_param_to_buffer(self.down_neighbor)
        if self.index % 10 == 0:
            self.aggregate_model_from_buff()

    def send_all_model_param_alg05(self):
        # Get the number of packages to send
        send_targets_num = 3

        # Get list of possible targets excluding self.rank
        target_candidates = list(range(self.origin_world_size))
        target_candidates.remove(self.rank)

        # Randomly select targets for each package
        random.shuffle(target_candidates)
        targets = target_candidates[:send_targets_num]
        
        random.shuffle(self.split_packages)
        for index, target in enumerate(targets):
            # Send the package to the target
            self.send_model_param_dict_to_store(target, layer_names=self.split_packages[index])

        # Aggregate models from buffer
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
            #model_avg_rpc_communicator.lock.release()

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
        aggregate_index_list = []
        for i in range(self.buffer_size):
            if not self.is_aggregated[i] and self.model_recv_buffer[i].loss_value != 0:
                aggregate_index_list.append(i)
        p = 1/ (len(aggregate_index_list)+1)
        temp = ModelStore(self.io_layers)
        for index in aggregate_index_list:
            buf_model = self.model_recv_buffer[index]
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
            self.is_aggregated[index] = True
            buf_model.lock.release()
        temp.term = int(temp.term / self.buffer_size)

        #if not self.lock.acquire(timeout= 3):
        #    raise Exception("lock acquire failed at aggregate_model_from_buff")
        with torch.no_grad():
            for layer_name, layer in self.io_layers.items():
                layer.weight = layer.weight*p + temp.layer_store[layer_name].weight
                if layer.bias is not None:
                    layer.bias = layer.bias*p + temp.layer_store[layer_name].bias
        #self.lock.release()
        if temp.resurrection_flag:
            self.rpc_communicator.update_node_iter(self.rank, temp.term)
            self.rpc_communicator.print_rpc_state(f"resurrection in {self.rank} to {temp.term}")

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
    selected_list =[]
    for i in range(model_avg_rpc_communicator.buffer_size):
        if model_avg_rpc_communicator.is_aggregated[i]:
            selected_list.append(i)
    if len(selected_list) > 0:
        indices = selected_list  # 创建包含范围内所有数值的列表
    else:
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
                model_avg_rpc_communicator.is_aggregated[i] = False
                break
        else:
            continue
    i = indices[0]
    if model_avg_rpc_communicator.model_recv_buffer[i].lock.acquire(timeout=0.5):
        try:
            for layer_name, model_weight, model_bias in data:
                model_avg_rpc_communicator.model_recv_buffer[i].layer_store[layer_name].aggregate_model_content_by_loss(
                    model_weight, model_bias, from_loss)
                model_avg_rpc_communicator.model_recv_buffer[i].term = max(
                    model_avg_rpc_communicator.model_recv_buffer[i].term, from_rank_iter)
                if resurrection_flag:
                    model_avg_rpc_communicator.model_recv_buffer[i].resurrection_flag = True
        finally:
            # 释放锁
            model_avg_rpc_communicator.model_recv_buffer[i].lock.release()
            model_avg_rpc_communicator.is_aggregated[i] = False
