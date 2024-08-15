import gc
import math
import random
import statistics
import sys
import time

import torch
import torch.distributed.rpc as rpc
import threading
from typing import Dict, Optional
import logging
import kfac.rpc_model_param_avg as model_param_avg_rpc
import kfac.rpc_task_manager as task_manager

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from kfac.layers.eigen import KFACEigenLayer,KFACBaseLayer
    from kfac.assignment import KAISAAssignment
    from kfac.base_preconditioner import BaseKFACPreconditioner, KFACPreconditioner

# 创建日志记录器
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)  # 设置日志级别

def normalized_l2_similarity(tensor1, tensor2):
    """
    计算两个任意维度 Tensor 之间的归一化 L2 范数相似度。

    Args:
        tensor1 (torch.Tensor): 第一个输入 Tensor
        tensor2 (torch.Tensor): 第二个输入 Tensor

    Returns:
        float: 两个 Tensor 之间的归一化 L2 范数相似度，值范围在 [0, 1] 之间
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError("Both tensors must have the same shape")

    l2_norm = torch.norm(tensor1 - tensor2)
    max_norm = torch.norm(tensor1) + torch.norm(tensor2)
    similarity = l2_norm / max_norm
    return round(similarity.item(),5)

def rpc_work_name(rank:int) -> str:
    return f"rpc_{rank}"

def full_connnection_device_map(world_size,rank):
    device_map = {}
    for i in range(world_size):
        if i == rank:
            continue
        device_map[rpc_work_name(i)] = {0 : 0}
    return device_map

def local_full_connnection_device_map(world_size,rank):
    device_map = {}
    for i in range(world_size):
        if i == rank:
            continue
        device_map[rpc_work_name(i)] = {rank%4 : i%4}
    return device_map

class NodeState():
    def __init__(self,rank):
        self.rank = rank
        self.iter = 0
        self.factor_computation_iter = -1
        self.inverse_computation_iter = -1
        self.model_param_avg_iter = -1
        self.health = True
        self.speed = 0

    def __str__(self):
        return f"R{self.rank} :t{self.iter}, h{self.health}"

class KfacTaskState:
    def __init__(self, layer_name, layer):
        self.layer_name = layer_name
        self.kfac_layer = layer
        self.inverse_comp_ready_iter = {"A":-1, "G":-1}
        self.inverse_comp_ok_iter = {"A":-1, "G":-1}
        self.grad_comp_iter = {"A":-1, "G":-1}
        self.grad_comp_ok_iter = -1

    def __str__(self):
        return f"{self.layer_name} :t{self.t}, A{self.A}, G{self.G}"

    def __repr__(self):
        return self.__str__()

class KfacRPCLayer:
    def __init__(self,a_handler,g_handler ,name , prediv_eigenvalues ,kfac_layer):
        self.factor : Dict[str:Optional['torch.Tensor']] = {"A" : None, "G": None }
        self.factor_recv_ct : Dict[str : int ]= {"A" : 0, "G": 0 }
        self.assigned_worker :Dict[str : int]= {'A': a_handler, 'G': g_handler}
        self.qa = None
        self.da = None
        self.recv_handled_a_version = -2
        self.last_load_handled_a_version = -2
        self.qg = None
        self.dg = None
        self.dgda = None
        self.prediv_eigenvalues = prediv_eigenvalues # False：da, dg must be provided; True: dgda must be provided
        self.kfac_layer :"KFACEigenLayer"=  kfac_layer
        self.recv_handled_g_version = -2
        self.last_load_handled_g_version = -2
        self.outdated_weight_param = 1
        self.ahead_weight_param = 1
        self.name = name

    def reassign_inverse_workers(self, a_handler, g_handler):
        self.assigned_worker['A'] = a_handler
        self.assigned_worker['G'] = g_handler

    def update_local_factor(self, recv_factor, local_t, recv_t, factor_type, world_size = 8):
        self.factor_recv_ct[factor_type] += 1
        if self.factor[factor_type] is None:
            self.factor[factor_type] = recv_factor
            return
        sigmoid_param = (recv_t - local_t) / (local_t + 1)
        recv_world_weight = 2 / ((1 + math.exp(-sigmoid_param)) * world_size)
        self.factor[factor_type] = (1 - recv_world_weight) * self.factor[factor_type] + recv_world_weight * recv_factor

    def update_local_eigen_a(self, qa, da, t):
        if t <= self.recv_handled_a_version :
            return # outdated
        self.qa = qa
        self.da = da
        self.recv_handled_a_version = t

    def update_local_eigen_g(self, qg, dg, dgda, t):
        if t <= self.recv_handled_g_version :
            return # outdated
        self.qg = qg
        self.dg = dg
        self.dgda = dgda
        self.recv_handled_g_version = t

    def clear_count_dict(self,local_t):
        self.factor_recv_ct : Dict[str:Dict[int, int]] = {"A" : {}, "G": {} }

    def load_eigen_tensor(self):
        assert self.name == self.kfac_layer.name
        assert self.recv_handled_a_version >=0 and self.recv_handled_g_version >= 0
        if self.last_load_handled_a_version < self.recv_handled_a_version:
            self.kfac_layer.qa = self.qa.clone()
            self.kfac_layer.qg = self.qg.clone()
        if self.last_load_handled_g_version < self.recv_handled_g_version:
            if self.prediv_eigenvalues:
                self.kfac_layer.dgda = self.dgda.clone()
            else:
                self.kfac_layer.dg = self.dg.clone()
                self.kfac_layer.da = self.da.clone()

class KFacRPCCommunicator:
    def __init__(self, world_size, rank, preconditioner:'BaseKFACPreconditioner' ,model, share_file_path ="", timestamp="" ,log_dir = "" , device = torch.device("cpu")):
        self.writer = None
        self.node_state_lock = threading.Lock()

        self.skip_inverse_computation_ct = 0
        self.slow_tolerance_value = 150
        self.max_election_period = 20

        self.request_regression_record = set()

        self.io_layers = None

        options = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=16,
            init_method=f"file://{share_file_path}/rpc_share{timestamp}",
            rpc_timeout=30,
        )
        if device == "cuda" or device.type == "cuda":
            options = rpc.TensorPipeRpcBackendOptions(
                num_worker_threads=16,
                init_method=f"file://{share_file_path}/rpc_share{timestamp}",
                rpc_timeout=30,
                device_maps=local_full_connnection_device_map(world_size,rank)
            )
        self.device = device
        rpc.init_rpc(name=f"rpc_{rank}", rank=rank, world_size=world_size,rpc_backend_options=options)
        self.origin_world_size = world_size
        self.rank = rank
        self.rpc_layers: Dict[str:KfacRPCLayer] = {} # {layer_name: KfacRPCLayer}
        self.computer_type = "" #eigen / inverse
        self.assigned_layers = []
        self.candidate_participate_factor_computation_layers = []
        self.participate_factor_computation_layers = []
        self.current_inverse_computation_layers = []
        for name, kfac_layer in preconditioner._layers.values():
            a_handler = preconditioner._assignment.inv_worker(name, 'A')
            g_handler = preconditioner._assignment.inv_worker(name, 'G')
            self.rpc_layers[name] = KfacRPCLayer(a_handler,g_handler ,name ,kfac_layer.prediv_eigenvalues ,kfac_layer)
            if a_handler == self.rank or g_handler == self.rank:
                self.assigned_layers.append(name)
                self.current_inverse_computation_layers.append(name)
            else:
                self.candidate_participate_factor_computation_layers.append(name)
                self.participate_factor_computation_layers.append(name)

        self.node_states: Dict[int, NodeState] = {}
        for i in range(world_size):
            self.node_states[i] = NodeState(i)
        self.lock = threading.Lock()

        # hyperparameters
        self.necessary_ct = 1
        self.load_inverse_max_loop = 3
        if rpc.is_available():
            print(f"RPC Communicator initialized for rank {rank}")
        else:
            raise RuntimeError(f"RPC initialization failed for rank {rank}")

        self.init_logger(rank,log_dir)
        self.model_avg_rpc = model_param_avg_rpc.ModelAvgRPCCommunicator(rank, model ,self)
        self.task_reassign_rpc = task_manager.RPCTaskManager(rpc_communicator=self, assignment=preconditioner._assignment ,slow_tolerance_value=self.slow_tolerance_value, max_election_period=self.max_election_period)

        self.model_accuracy_statistic : Dict[int , Dict[str ,int]]= dict() # {epoch: (recv_ct ,correct_ct, total_ct)}

        self.update_assignment_callback = None
        self.send_model_param_callback = None

        self.layers_workload = preconditioner._assignment.work
        self.computation_volume_accumulation = 0
        self.time_cost_accumulation = 0
        self.loop_start_time = 0

        self.shutdown_flag = False

        global global_communicator
        global_communicator = self

        self.gradient_computation_start = False

    def close_rpc(self):
        rpc.shutdown()

    def get_health_node_state_list(self) -> list[NodeState]:
        return [state for rank, state in self.node_states.items() if state.health]

    def get_health_nodes_rank_list(self):
        return [state.rank for state in self.node_states.values() if state.health]

    def get_sick_node_list(self) -> list[NodeState]:
        return [state for rank, state in self.node_states.items() if not state.health]

    def max_iter_in_cluster(self):
        # return max iter in node_states
        return max([state.iter for state in self.get_health_node_state_list()])

    def min_iter_in_health_nodes(self):
        return min([state.iter for state in self.get_health_node_state_list()])

    def median_iter_in_health_nodes(self):
        iters = [state.iter for state in self.get_health_node_state_list()]
        return statistics.median(iters)

    def init_logger(self,rank,log_dir):
        # 创建一个 FileHandler，并设置级别为 DEBUG
        file_handler = logging.FileHandler(f'{log_dir}/log_{rank}.log')
        file_handler.setLevel(logging.DEBUG)

        # 创建一个日志格式器，并将其添加到 FileHandler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        global logger
        # 将 FileHandler 添加到日志记录器
        logger.addHandler(file_handler)

        logger.propagate = False

    def print_rpc_state(self, text = ""):
        global logger
        log_txt = ""
        for node_rank, state in self.node_states.items():
            log_txt += f"{state}; "
        log_txt += self.task_reassign_rpc.print_state()
        logger.debug(f"Rank {self.rank}: {log_txt} , {text}")

    def debug_print(self, text):
        pass
        global logger
        if self.min_iter_in_health_nodes() == self.current_t():
            logger.debug(f"current iter {self.current_t()} in rank {self.rank}, {text}")

    def __repr__(self):
        log = f"Rank {self.rank} : iter {self.current_t()}\n"
        for name, layer in self.rpc_layers.items():
            log += f"Layer {name}:\n"
            for factor_type, factor in layer.factor.items():
                log += f"\t{factor_type} factor: {factor}\n"
                log += f"\t{factor_type} factor recv ct : {layer.factor_recv_ct[factor_type]}\n"
            log += f"\tA eigen: {layer.qa}, {layer.da}\n"
            log += f"\tG eigen: {layer.qg}, {layer.dg}, {layer.dgda}\n"
            log += f"\thandled A recv iter: {layer.recv_handled_a_version}\n"
            log += f"\thandled G recv iter: {layer.recv_handled_g_version}\n"
            log += f"\tA handler: {layer.assigned_worker['A']}\n"
            log += f"\tG handler: {layer.assigned_worker['G']}\n"
        return log

    def load_factor(self,kfac_layer: 'KFACBaseLayer', factor_type):
        if self.assigned_worker(kfac_layer.name, factor_type) == self.rank:
            assert self.rpc_layers[kfac_layer.name].factor[factor_type] is not None
            if factor_type == "A":
                kfac_layer.a_factor = self.rpc_layers[kfac_layer.name].factor["A"].clone().detach()
            if factor_type == "G":
                kfac_layer.g_factor = self.rpc_layers[kfac_layer.name].factor["G"].clone().detach()
        return True

    def shutdown(self):
        rpc.shutdown()

    def update_self_t(self):
        self.loop_start_time = time.time()
        if not self.node_state_lock.acquire(timeout=1):
            raise RuntimeError("Failed to acquire lock in update_self_t")
        self.node_states[self.rank].iter += 1
        self.node_state_lock.release()

    def current_t(self):
        return self.node_states[self.rank].iter

    def update_node_iter(self, from_rank, t , speed = None):
        if not self.node_state_lock.acquire(timeout=1):
            raise RuntimeError("Failed to acquire lock in node_state")
        if from_rank not in self.node_states:
            raise RuntimeError(f"Rank {from_rank} is not in the node_states")
        self.node_states[from_rank].iter = max(self.node_states[from_rank].iter, t)
        if speed is not None:
            self.node_states[from_rank].speed = speed
        self.node_state_lock.release()

    def compute_and_broadcast_inverse(self, preconditioner: 'BaseKFACPreconditioner'):
        current_t = self.current_t()
        task_set = set()
        for layer_name in self.current_inverse_computation_layers:
            task_set.add(layer_name + "#A")
            #task_set.add(layer_name + "#G")
        while len(task_set) > 0:
            ready_list = []
            for task_name in task_set:
                layer_name, factor_type = task_name.split("#")
                if self.is_factor_ready(layer_name, factor_type):
                    ready_list.append(task_name)
            if len(ready_list) == 0:
                ready_list.append(random.choice(list(task_set)))
            for ready_task_name in ready_list:
                layer_name, factor_type = ready_task_name.split("#")
                kfac_layer = self.rpc_layers[layer_name].kfac_layer
                self.load_factor(kfac_layer=kfac_layer, factor_type=factor_type)
                if factor_type == "A":
                    kfac_layer.compute_a_inv(damping=preconditioner.damping)
                    self.broadcast_kfac_eigen_tensor_a(layer_name=layer_name, qa=kfac_layer.qa, da=kfac_layer.da)
                elif factor_type == "G":
                    kfac_layer.compute_g_inv(damping=preconditioner.damping)
                    self.broadcast_kfac_eigen_tensor_g(layer_name=layer_name, qg=kfac_layer.qg, dg=kfac_layer.dg,
                                                dadg=kfac_layer.dgda)

                task_set.remove(ready_task_name)
                if factor_type == "A":
                    task_set.add(layer_name + "#G")

    def compute_preconditioned_gradients(self,damping):
        current_t = self.current_t()
        all_layer = set(self.rpc_layers.keys())
        if not self.gradient_computation_start:
            for layer_name ,layer in self.rpc_layers.items():
                if layer.recv_handled_a_version < 0 or layer.recv_handled_g_version < 0:
                    return False # start next forwarding because not all of the eigen tensor is ready
            self.gradient_computation_start = True
        for layer_name in self.current_inverse_computation_layers:
            self.rpc_layers[layer_name].load_eigen_tensor()
            self.rpc_layers[layer_name].kfac_layer.preconditioned_grad(damping=damping)
            all_layer.remove(layer_name)

        while len(all_layer) > 0:
            ready_set = set()
            try_ct = 0
            while len(ready_set) == 0 and try_ct < 3:
                for layer_name in all_layer:
                    if self.is_eigen_tensor_ready(layer_name,try_ct) :
                        ready_set.add(layer_name)
                try_ct += 1
            if len(ready_set) == 0:
                ready_set.add(random.choice(list(all_layer)))
            for layer_name in ready_set:
                self.rpc_layers[layer_name].load_eigen_tensor()
                self.rpc_layers[layer_name].kfac_layer.preconditioned_grad(damping=damping)
            all_layer = all_layer - ready_set

    def get_computation_speed_dict(self):
        computational_efficiency = dict()
        summ = 0
        ct = 0
        if self.time_cost_accumulation > 0:
            computational_efficiency[self.rank] = self.computation_volume_accumulation / self.time_cost_accumulation
            summ = computational_efficiency[self.rank]
            ct = 1

        for state in self.get_health_node_state_list():
            if state.rank == self.rank:
                continue
            if state.speed is not None and state.speed > 0:
                summ += state.speed
                ct += 1
                computational_efficiency[state.rank] = state.speed

        avg = summ / ct

        for state in self.get_health_node_state_list():
            if state.rank not in computational_efficiency:
                computational_efficiency[state.rank] = avg
        self.print_rpc_state(f"computation efficiency: {computational_efficiency}")
        return computational_efficiency

    def update_node_state_list(self, new_health_node_list):
        for node_rank in self.node_states:
            if node_rank in new_health_node_list:
                self.node_states[node_rank].health = True
            else:
                self.node_states[node_rank].health = False

    def update_inverse_workers(self, new_assignment, new_assignment_generation):
        self.task_reassign_rpc.assignment._inv_assignments = new_assignment
        self.assigned_layers.clear() # not change unless reassign
        self.candidate_participate_factor_computation_layers.clear() # not change unless reassign
        if new_assignment_generation is not None:
            self.task_reassign_rpc.assignment_generation = new_assignment_generation
        for name, kfac_layer in self.rpc_layers.items():
            a_handler = new_assignment[name]['A']
            g_handler =  new_assignment[name]['G']
            if a_handler != self.rank:
                kfac_layer.factor['A'] = None
            if g_handler != self.rank:
                kfac_layer.factor['G'] = None
            self.rpc_layers[name].reassign_inverse_workers(a_handler,g_handler)
            if a_handler == self.rank or g_handler == self.rank:
                self.assigned_layers.append(name)
            else:
                self.candidate_participate_factor_computation_layers.append(name)
        gc.collect()
        self.print_rpc_state(f"update new assignment {new_assignment_generation}: {new_assignment}")
        self.current_inverse_computation_layers = self.assigned_layers.copy()
        self.participate_factor_computation_layers = self.candidate_participate_factor_computation_layers.copy()
        self.update_assignment_callback = None
        self.task_reassign_rpc.running_time = 0

    def get_world_size(self):
        return len(self.node_states.keys())

    def get_health_world_size(self):
        return len(self.get_health_node_state_list())

    def is_factor_ready(self, layer_name, factor_type):
        if self.rpc_layers[layer_name].factor_recv_ct[factor_type] >= self.current_t() * (self.origin_world_size-1):
            return True
        return

    def is_eigen_tensor_ready(self, layer_name,staleness_tolerance = 0):
        current_t = self.current_t()
        if (self.rpc_layers[layer_name].recv_handled_a_version < 0 or
                self.rpc_layers[layer_name].recv_handled_g_version < 0):
            return False
        if (self.rpc_layers[layer_name].recv_handled_a_version < current_t - staleness_tolerance
                or self.rpc_layers[layer_name].recv_handled_g_version < current_t - staleness_tolerance):
            return False
        return True

    def assigned_worker(self, layer_name, factor_type):
        return self.rpc_layers[layer_name].assigned_worker[factor_type]

    def send_kfac_factor(self,layer_name:str,factor_tensor :torch.Tensor, factor_type:str):
        if layer_name not in self.participate_factor_computation_layers and layer_name not in self.assigned_layers:
            return True
        target = 0
        if factor_type == "A":
            target = self.rpc_layers[layer_name].assigned_worker['A']
        elif factor_type == "G":
            target = self.rpc_layers[layer_name].assigned_worker['G']
        t = self.current_t()
        if target == self.rank:
            #self.print_rpc_state(f"update local factor {factor_type} of {layer_name} in rank {self.rank}")
            self.rpc_layers[layer_name].update_local_factor(factor_tensor.clone(), t, t, factor_type, world_size=self.origin_world_size)
            return
        try:
            rpc.rpc_async(
                to=rpc_work_name(target),
                func=receive_kfac_factor,
                args=(self.rank, layer_name, factor_tensor, t, factor_type)
            )
        except Exception as e:
            print(f"Failed to send factor to {target} from {self.rank}: {e}")
        return True

    def broadcast_kfac_eigen_tensor_a(self, layer_name,qa:torch.Tensor,da:torch.Tensor):
        if self.assigned_worker(layer_name, 'A') != self.rank:
            return
        if qa is None:
            raise RuntimeError(
                f'Attempt to broadcast A inv from src={self.rank} but this rank '
                'has not computed inv yet.',
            )
        t= self.current_t()
        self.rpc_layers[layer_name].update_local_eigen_a(qa.clone(), da.clone,t)

        for i in range(self.origin_world_size-1):
            target_rank = (self.rank + i + 1) % self.origin_world_size
            try :
                rpc.rpc_async(
                    to=rpc_work_name(target_rank),
                    func=receive_eigen_tensor_a,
                    args=(self.rank, layer_name, qa, da, t)
                )
            except Exception as e:
                print(f"Failed to send eigen tensor to {target_rank} from {self.rank}: {e}")

    def broadcast_kfac_eigen_tensor_g(self, layer_name,qg:torch.Tensor,dg:torch.Tensor,dadg: None|torch.Tensor):
        t = self.current_t()
        if self.rpc_layers[layer_name].prediv_eigenvalues:
            if dadg is None:
                raise RuntimeError(
                    f'Attempt to broadcast g inv from src={self.rank} but this rank '
                    'has not computed inv yet.',
                )
        elif qg is None:
            raise RuntimeError(
                f'Attempt to broadcast g inv from src={self.rank} but this rank '
                'has not computed inv yet.',
            )

        if self.rpc_layers[layer_name].prediv_eigenvalues:
            self.rpc_layers[layer_name].update_local_eigen_g(qg, dg, dadg.clone(), t)
        else:
            self.rpc_layers[layer_name].update_local_eigen_g(qg.clone(), dg.clone(), dadg, t)

        for i in range(self.origin_world_size-1):
            target_rank = (self.rank + i + 1) % self.origin_world_size
            try:
                rpc.rpc_async(
                    to=rpc_work_name(target_rank),
                    func=receive_eigen_tensor_g,
                    args=(self.rank, layer_name, qg, dg, dadg, t)
                )
            except Exception as e:
                print(f"Failed to send eigen tensor to {target_rank} from {self.rank}: {e}")

    def is_factor_computation_skipped(self, layer_name):
        if layer_name not in self.participate_factor_computation_layers and layer_name not in self.current_inverse_computation_layers:
            return True
        return False

    def computation_volume_statistic(self):
        current_t = self.current_t()
        if current_t % 100 == 0:
            self.computation_volume_accumulation = 0
            self.time_cost_accumulation = 0

        loop_time_cost = time.time() - self.loop_start_time
        self.time_cost_accumulation += loop_time_cost
        for layer_name in self.current_inverse_computation_layers:
            self.computation_volume_accumulation += self.layers_workload[layer_name]["A"]*1.1  +self.layers_workload[layer_name]["G"]
        for layer_name in self.participate_factor_computation_layers:
            self.computation_volume_accumulation += self.layers_workload[layer_name]["A"]*0.1

        self.node_states[self.rank].speed = int(self.computation_volume_accumulation / self.time_cost_accumulation)

    def facotr_comput_lazy_wl_rebal(self):
        self.computation_volume_statistic()
        current_t = self.current_t()
        forward_than_local = sum(state.iter > current_t for state in self.get_health_node_state_list())
        late_than_local = sum(state.iter < current_t for state in self.get_health_node_state_list())
        iter_diff = self.max_iter_in_cluster() - current_t

        random.shuffle(self.candidate_participate_factor_computation_layers)
        random.shuffle(self.assigned_layers)
        self.participate_factor_computation_layers = \
            self.candidate_participate_factor_computation_layers[:len(self.participate_factor_computation_layers)]
        self.current_inverse_computation_layers = self.assigned_layers[:len(self.current_inverse_computation_layers)]
        if forward_than_local >= math.ceil(self.get_world_size() * 0.7) and iter_diff > 3: # local is too slow, work less
            if len(self.participate_factor_computation_layers) > 0:
                layer_name = random.choice(self.participate_factor_computation_layers)
                self.participate_factor_computation_layers.remove(layer_name)
            elif len(self.current_inverse_computation_layers) > 0:
                layer_name = random.choice(self.current_inverse_computation_layers)
                self.current_inverse_computation_layers.remove(layer_name)

        if late_than_local >= 1 or forward_than_local <= 2: #math.ceil(self.world_size * 0.3): # local is quick, work more
            if len(self.current_inverse_computation_layers) < len(self.assigned_layers):
                for layer_name in reversed(self.assigned_layers):
                    if layer_name not in self.current_inverse_computation_layers:
                        self.current_inverse_computation_layers.append(layer_name)
                        break
            elif len(self.participate_factor_computation_layers) < len(self.candidate_participate_factor_computation_layers):
                for layer_name in reversed(self.candidate_participate_factor_computation_layers):
                    if layer_name not in self.participate_factor_computation_layers:
                        self.participate_factor_computation_layers.append(layer_name)
                        break

    def send_model_param(self):
        self.model_avg_rpc.send_all_model_param_alg07()

    def send_rpc_test_result(self, correct_ct, total_ct, epoch):
        for i in range(self.origin_world_size):
            try:
                rpc.rpc_async(
                    to=rpc_work_name(i),
                    func=recv_rpc_test_result,
                    args=(correct_ct, total_ct, epoch)
                )
            except Exception as e:
                print(f"Failed to send test result to 0: {e} from {self.rank}")

    def wait_and_return_test_result(self, epoch):
        wait_time = 0
        while epoch not in self.model_accuracy_statistic or self.model_accuracy_statistic[epoch]['recv_ct'] < self.origin_world_size:
            time.sleep(0.1)
            wait_time += 1
            if wait_time > 2:
                break

        if epoch in self.model_accuracy_statistic:
            return self.model_accuracy_statistic[epoch]['correct_ct'] / self.model_accuracy_statistic[epoch]['total_ct']
        else:
            return 0

    def restart_sick_node(self): # call by sick nodes
        if self.node_states[self.rank].health == False and self.task_reassign_rpc.assignment_generation not in self.request_regression_record:
            self.task_reassign_rpc.resurrection_declaration()
            self.print_rpc_state(f"request regression from sick node {self.rank}")
            self.request_regression_record.add(self.task_reassign_rpc.assignment_generation)

    def arrange_to_send_the_latest_model(self, survived_nodes):
        """
           params: set of resurrection_node
           return: dict of {health node rank : layer_name}
        """
        send_task = self.model_avg_rpc.Send_to_Easter_Point_Task_Assignment(survived_nodes)
        return send_task

    def send_new_model_to_resurrection_node(self,layer_name_list,resurrection_node_list):
        for node_rank in resurrection_node_list:
            self.model_avg_rpc.send_model_param_to_buffer(node_rank, layer_name_list)

        self.send_model_param_callback = None

    def broadcast_shutdown(self):
        if self.rank != self.task_reassign_rpc.leader_rank:
            return
        for i in range(self.origin_world_size):
            if i == self.rank:
                continue
            try :
                rpc.rpc_async(
                    to=rpc_work_name(i),
                    func=shutdown_notification,
                    args=(self.rank,)
                )
            except Exception as e:
                print(f"Failed to send shutdown to {i} from {self.rank}: {e}")

global_communicator: KFacRPCCommunicator = None

def receive_kfac_factor(from_rank, layer_name, factor, from_iter, factor_type):
    global global_communicator
    self = global_communicator

    if self.rpc_layers[layer_name].assigned_worker[factor_type] != self.rank:
        return

    #with self.lock:
    current_t = self.current_t()
    self.rpc_layers[layer_name].update_local_factor(factor, current_t, from_iter, factor_type ,world_size=self.origin_world_size)
    self.update_node_iter(from_rank, from_iter)

def receive_eigen_tensor_a(from_rank, layer_name, qa, da, t):
    global global_communicator
    if t < global_communicator.rpc_layers[layer_name].recv_handled_a_version:
        return
    global_communicator.rpc_layers[layer_name].update_local_eigen_a(qa, da, t)
    global_communicator.update_node_iter(from_rank, t)

def receive_eigen_tensor_g(from_rank, layer_name, qg, dg, dadg, t):
    global global_communicator
    #with global_communicator.lock:
    if t < global_communicator.rpc_layers[layer_name].recv_handled_g_version:
        return
    global_communicator.rpc_layers[layer_name].update_local_eigen_g(qg, dg, dadg, t)
    global_communicator.update_node_iter(from_rank, t)

def recv_rpc_test_result(correct_ct, total_ct, epoch):
    global global_communicator
    if epoch not in global_communicator.model_accuracy_statistic:
        global_communicator.model_accuracy_statistic[epoch] = {'recv_ct': 1, 'correct_ct': correct_ct, 'total_ct': total_ct}
    else:
        global_communicator.model_accuracy_statistic[epoch]['recv_ct'] +=1
        global_communicator.model_accuracy_statistic[epoch]['correct_ct'] += correct_ct
        global_communicator.model_accuracy_statistic[epoch]['total_ct'] += total_ct

def shutdown_notification(from_rank):
    global global_communicator
    time.sleep(2)
    if from_rank == global_communicator.task_reassign_rpc.leader_rank:
        global_communicator.writer.close()
        if rpc.is_available():
            rpc.shutdown()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        sys.exit(1)
    else:
        time.sleep(5)
        global_communicator.writer.close()
        if rpc.is_available():
            rpc.shutdown()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        sys.exit(2)
