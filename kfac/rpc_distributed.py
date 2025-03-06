import gc
import math
import random
import statistics
import sys
import time

import psutil
import torch
import torch.distributed.rpc as rpc
import threading
from typing import Dict, Optional, Tuple
import logging
from kfac.adsgds.exp_swift import ExpTopoSwiftManager
from kfac.adsgds.adpsgd import AdpsgdManager
from kfac.adsgds.swift import SwiftManager
import kfac.rpc_task_manager as task_manager
from kfac.rpc_util.send_scheduler import DataSendScheduler ,PackageSender
import numpy as np
from kfac.rpc_util.communication_statatic import CommunicationStatics

from typing import TYPE_CHECKING ,List
if TYPE_CHECKING:
    from kfac.layers.eigen import KFACEigenLayer,KFACBaseLayer
    from kfac.base_preconditioner import BaseKFACPreconditioner
from kfac.rpc_util.fault_sim import fault_simulator

# 创建日志记录器
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)  # 设置日志级别

def rpc_work_name(rank:int) -> str:
    return f"rpc_{rank}"

def full_connection_device_map(world_size,rank):
    device_map = {}
    for i in range(world_size):
        if i == rank:
            continue
        device_map[rpc_work_name(i)] = {0 : 0}
    return device_map

def local_full_connection_device_map(world_size,rank):
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
        self.health = 0
        self.speed = 0

    def __str__(self):
        return f"R{self.rank} :t{self.iter},s{self.speed}, h{self.health}"


class KfacRPCLayer:
    def __init__(self,a_handler,g_handler ,name ,kfac_layer):
        self.kfac_layer :"KFACEigenLayer"= kfac_layer
        self.tensor_locks :Dict[str, 'threading.Lock'] = {"A": threading.Lock(), "G": threading.Lock() , "qa": threading.Lock() , "qg": threading.Lock()}
        #self.factor_recv_ct : Dict[str , int ]= {"A" : 0, "G": 0 }
        self.assigned_worker :Dict[str , int]= {'A': a_handler, 'G': g_handler}
        self.recv_handled_a_version = -2
        self.recv_handled_g_version = -2
        self.name = name
        self.send_trigger = {"A": False, "G": False}

    def reassign_inverse_workers(self, a_handler, g_handler):
        self.assigned_worker['A'] = a_handler
        self.assigned_worker['G'] = g_handler

    def update_local_factor(self, recv_factor, local_t, recv_t, factor_type, world_size = 8):
        #self.factor_recv_ct[factor_type] += 1
        with self.tensor_locks[factor_type]:
            if self.kfac_layer.get_factor(factor_type) is None:
                self.kfac_layer.set_factor(factor_type, recv_factor)
                return
            #sigmoid_param = (recv_t - local_t) / (local_t + 1)
            #recv_world_weight = 2 / ((1 + math.exp(-sigmoid_param)) * world_size)
            #self.kfac_layer.set_factor(factor_type, self.kfac_layer.get_factor(factor_type) * (1 - recv_world_weight) + recv_factor * recv_world_weight)
            if factor_type == "A":
                self.kfac_layer._a_factor.mul_(0.9).add_(recv_factor, alpha=0.1)
            elif factor_type == "G":
                self.kfac_layer._g_factor.mul_(0.9).add_(recv_factor, alpha=0.1)

    def update_local_eigen_a(self, qa, da, t):
        if t <= self.recv_handled_a_version :
            return # outdated
        with self.tensor_locks['qa']:
            self.kfac_layer.set_factor("qa", qa)
            self.kfac_layer.set_factor("da", da)
            self.recv_handled_a_version = t

    def update_local_eigen_g(self, qg, dg, dgda, t):
        if t <= self.recv_handled_g_version :
            return # outdated
        with self.tensor_locks['qg']:
            self.kfac_layer.set_factor("qg", qg)
            self.kfac_layer.set_factor("dg", dg)
            self.kfac_layer.set_factor("dgda", dgda)
            self.recv_handled_g_version = t

class KFacRPCCommunicator:
    def __init__(self, world_size, rank, preconditioner:'BaseKFACPreconditioner' ,
                 model, share_file_path ="", timestamp="" ,
                 log_dir = "" , device = torch.device("cpu"),steps_per_epoch = 0):
        self.steps_per_epoch = steps_per_epoch
        if device == "cuda" or device.type == "cuda":
            options = rpc.TensorPipeRpcBackendOptions(
                num_worker_threads=32,
                init_method=f"file://{share_file_path}/rpc_share{timestamp}",
                rpc_timeout=30,
                device_maps=full_connection_device_map(world_size,rank),
                _channels=["cma", "mpt_uv", "basic", "cuda_xth", "cuda_ipc", "cuda_basic"]
            )
            self.total_memory = torch.cuda.get_device_properties(0).total_memory
        else:
            options = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=16,
            init_method=f"file://{share_file_path}/rpc_share{timestamp}",
            rpc_timeout=30,
        )

        rpc.init_rpc(name=f"rpc_{rank}", rank=rank, world_size=world_size,rpc_backend_options=options)
        if rpc.is_available():
            print(f"RPC Communicator initialized for rank {rank}")
        else:
            raise RuntimeError(f"RPC initialization failed for rank {rank}")
        self.device  = device
        self.origin_world_size = world_size
        self.rank = rank
        self.data_send_scheduler = DataSendScheduler(is_kfac= (preconditioner is not None))
        self.writer = None
        self.node_states: Dict[int, NodeState] = {}
        for i in range(world_size):
            self.node_states[i] = NodeState(i)
        self.node_state_lock = threading.Lock()
        self.init_logger(rank,log_dir)
        self.model_avg_rpc = AdpsgdManager(rank, model, self)
        self.com_statistic = CommunicationStatics()
        global global_communicator
        global_communicator = self

        if preconditioner is None:
            return
        
        self.is_packaged_send = preconditioner.is_packaged_send
        self.request_regression_record = set()
        self.rpc_layers: Dict[str,KfacRPCLayer] = {} # {layer_name: KfacRPCLayer}
        self.assigned_layers = []
        self.candidate_participate_factor_computation_layers = []
        self.current_participate_factor_computation_layers = []
        self.current_inverse_computation_layers = []
        for name, kfac_layer in preconditioner._layers.values():
            a_handler = preconditioner._assignment.inv_worker(name, 'A')
            g_handler = preconditioner._assignment.inv_worker(name, 'G')
            self.rpc_layers[name] = KfacRPCLayer(a_handler,g_handler ,name ,kfac_layer)
            if a_handler == self.rank or g_handler == self.rank:
                self.assigned_layers.append(name)
                self.current_inverse_computation_layers.append(name)
            else:
                self.candidate_participate_factor_computation_layers.append(name)
                self.current_participate_factor_computation_layers.append(name)

        self.update_send_trigger()
        self.is_packged_send = preconditioner.is_packaged_send

        # hyperparameters
        self.task_reassign_rpc = task_manager.RPCTaskManager(rpc_communicator=self, assignment=preconditioner._assignment)

        self.model_accuracy_statistic : Dict[int , Dict[str ,int]]= dict() # {epoch: (recv_ct ,correct_ct, total_ct)}

        self.update_assignment_callback = None
        self.send_model_param_callback = None

        self.layers_workload = preconditioner._assignment.work
        self.computation_volume_accumulation = 0
        self.time_cost_accumulation = 0

        self.gradient_computation_start = False
        self.package_sender = PackageSender(self)
        self.execution_times_statistic:Dict[str , Dict[int , list]] = {"A":{}, "G":{}}
        self.trian_start_time = time.time()
    
    def add_execution_times_statistic(self,shape:int,type_name:str,time:float):
        assert type_name in self.execution_times_statistic
        if shape not in self.execution_times_statistic:
            self.execution_times_statistic[type_name][shape] = [0,0]
        self.execution_times_statistic[type_name][shape][0] += time
        self.execution_times_statistic[type_name][shape][1] += 1

    def update_send_trigger(self):
        target_set = set()
        target_set.add(self.rank)
        log_info = ""
        for name, layer in reversed(self.rpc_layers.items()):
            handler_rank = layer.assigned_worker['A']
            if handler_rank not in target_set and handler_rank != self.rank:
                target_set.add(handler_rank)
                layer.send_trigger["A"] = True
                log_info += f"A -> {handler_rank} @ {name} "

        target_set.clear()
        target_set.add(self.rank)
        for name, layer in self.rpc_layers.items():
            handler_rank = layer.assigned_worker['G']
            if handler_rank not in target_set and handler_rank != self.rank:
                target_set.add(handler_rank)
                layer.send_trigger["G"] = True
                log_info += f"G -> {handler_rank} @ {name} "
        print(f"{log_info} at {self.rank}")

    def compute_iter_variance(self):
    # 提取所有节点的iter值
        iter_values = [node_state.iter for node_state in self.node_states.values()]
    
    # 计算方差
        variance = np.std(iter_values)
        return variance

    def close_rpc(self):
        rpc.shutdown()

    def get_working_node_state_list(self) -> list[NodeState]:
        return [state for rank, state in self.node_states.items() if state.health == 0 or state.health == 1]

    def get_working_nodes_rank_list(self):
        return [state.rank for state in self.node_states.values() if state.health == 0 or state.health == 1]
    
    def get_health_nodes_rank_list(self) -> list[int]:
        return [state.rank for state in self.node_states.values() if state.health == 0]

    def get_sick_node_list(self) -> list[NodeState]:
        return [state for rank, state in self.node_states.items() if state.health == 2]

    def max_iter_in_cluster(self):
        # return max iter in node_states
        return max([state.iter for state in self.get_working_node_state_list()])

    def min_iter_in_working_nodes(self):
        return min([state.iter for state in self.get_working_node_state_list()])

    def median_iter_in_working_nodes(self):
        iters = [state.iter for state in self.get_working_node_state_list()]
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

    def get_memory_usage_percent(self):
        if self.device == "cuda" or self.device.type == "cuda":
            return torch.cuda.memory_allocated() / self.total_memory
        else:
            return psutil.virtual_memory().percent / 100

    def print_rpc_state(self, text = ""):
        global logger
        log_txt = ""
        for node_rank, state in self.node_states.items():
            log_txt += f"{state}; "
        if hasattr(self, "task_reassign_rpc"):
            log_txt += self.task_reassign_rpc.print_state()
        logger.info(f"{log_txt} , {text}")

    def debug_print(self, text):
        global logger
        logger.warning(f"T{self.current_t()} in R{self.rank}, {text}")

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

    def shutdown(self):
        rpc.shutdown()

    def update_self_t(self):
        self.data_send_scheduler.update_loop_counter()
        with self.node_state_lock:
            self.node_states[self.rank].iter += 1
        self.is_do_kfac_this_loop = False
        if self.data_send_scheduler.get_next_send_type() is not None:
            self.is_do_kfac_this_loop = True
            self.debug_print(f"do {self.data_send_scheduler.get_next_send_type()} this loop ,memory usage: {self.get_memory_usage_percent()}")
            if self.get_memory_usage_percent() > 0.5:
                self.data_send_scheduler.relax_send_interval()
                self.debug_print(f"relax send interval to {self.data_send_scheduler.intervals}")
            if self.get_memory_usage_percent() < 0.3 and self.data_send_scheduler.intervals["model_param"] > self.data_send_scheduler.start_interval["model_param"]:
                self.data_send_scheduler.shorten_send_interval()
                self.debug_print(f"shorten send interval to {self.data_send_scheduler.intervals}")

    def current_t(self):
        return self.node_states[self.rank].iter

    def get_layer_lock(self, layer_name, factor_name):
        assert factor_name in ['A', 'G', 'qa', 'qg']
        assert layer_name in self.rpc_layers
        return self.rpc_layers[layer_name].tensor_locks[factor_name]

    def compute_and_broadcast_inverse(self, preconditioner: 'BaseKFACPreconditioner'):
        current_send_type = self.data_send_scheduler.get_next_send_type()
        if current_send_type is None or current_send_type == "model_param":
            return
        elif current_send_type == "factor":
            self.data_send_scheduler.update_next_send_time("factor")
            return
        task_set = set()
        for layer_name in self.current_inverse_computation_layers:
            task_set.add(layer_name + "#A")
        while len(task_set) > 0:
            ready_list = []
            for task_name in task_set:
                layer_name, factor_type = task_name.split("#")
                if self.is_factor_ready(layer_name, factor_type):
                    ready_list.append(task_name)
            if len(ready_list) == 0:
                ready_list.append(random.choice(list(task_set)))
            for ready_task_name in ready_list:
                start_time = time.time()
                layer_name, factor_type = ready_task_name.split("#")
                kfac_layer = self.rpc_layers[layer_name].kfac_layer
                if factor_type == "A":
                    with self.rpc_layers[layer_name].tensor_locks['qa'], self.rpc_layers[layer_name].tensor_locks['A']:
                        kfac_layer.compute_a_inv(damping=preconditioner.damping)
                        self.rpc_layers[layer_name].recv_handled_a_version = self.current_t()
                    if not self.is_packged_send:
                        self.broadcast_kfac_eigen_tensor_a(layer_name=layer_name)
                    else:
                        self.async_eigen_broadcast_register(layer_name=layer_name,factor_types =['qa'])
                elif factor_type == "G":
                    with self.get_layer_lock(layer_name,'qg'), self.get_layer_lock(layer_name,'qa') ,self.get_layer_lock(layer_name,'G'):
                        kfac_layer.compute_g_inv(damping=preconditioner.damping)
                        self.rpc_layers[layer_name].recv_handled_g_version = self.current_t()
                    if not self.is_packged_send:
                        self.broadcast_kfac_eigen_tensor_g(layer_name=layer_name)
                    else:
                        self.async_eigen_broadcast_register(layer_name=layer_name,factor_types =['qg','dgda'])
                task_set.remove(ready_task_name)
                if factor_type == "A":
                    task_set.add(layer_name + "#G")
                    shape = kfac_layer._a_factor.shape[0]
                elif factor_type == "G":
                    shape = kfac_layer._g_factor.shape[0]
                self.add_execution_times_statistic(shape=shape,type_name=factor_type,time=time.time()-start_time) 
        for rank in range(self.origin_world_size):
            if rank == self.rank:
                continue
            self.send_data_package(rank)
        self.data_send_scheduler.update_next_send_time("eigen")
        self.computation_volume_statistic_and_speed()

    def compute_preconditioned_gradients(self,damping):
        all_layer = set(self.rpc_layers.keys())
        if not self.gradient_computation_start:
            for layer_name ,layer in self.rpc_layers.items():
                if layer.recv_handled_a_version < 0 or layer.recv_handled_g_version < 0:
                    return False # start next forwarding because not all of the eigen tensor is ready
            self.gradient_computation_start = True
        for layer_name in self.current_inverse_computation_layers:
            with self.get_layer_lock(layer_name, "qa") and self.get_layer_lock(layer_name, "qg"):
                self.rpc_layers[layer_name].kfac_layer.preconditioned_grad(damping=damping)
            self.rpc_layers[layer_name].kfac_layer.update_grad(None)
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
                with self.get_layer_lock(layer_name, "qa") and self.get_layer_lock(layer_name, "qg"):
                    self.rpc_layers[layer_name].kfac_layer.preconditioned_grad(damping=damping)
                self.rpc_layers[layer_name].kfac_layer.update_grad(None)
            all_layer = all_layer - ready_set

    def get_computation_speed_dict(self):
        computational_efficiency = dict()
        avg = statistics.mean([state.speed for state in self.get_working_node_state_list() if state.speed is not None and state.speed > 0])
        self.print_rpc_state(f"avg speed: {avg}")
        for state in self.get_working_node_state_list():
            if avg == 0:
                computational_efficiency[state.rank] = 1000
                continue
            if state.speed > 0:
                computational_efficiency[state.rank] = state.speed
            elif state.speed == 0:
                computational_efficiency[state.rank] = avg

        self.print_rpc_state(f"computation efficiency: {computational_efficiency}")
        return computational_efficiency

    def update_inverse_workers(self, new_assignment, new_assignment_generation):
        self.task_reassign_rpc.assignment._inv_assignments = new_assignment
        self.assigned_layers.clear() # not change unless reassign
        self.candidate_participate_factor_computation_layers.clear() # not change unless reassign
        if new_assignment_generation is not None:
            self.task_reassign_rpc.assignment_generation = new_assignment_generation
        for name, kfac_layer in self.rpc_layers.items():
            a_handler = new_assignment[name]['A']
            g_handler =  new_assignment[name]['G']
            self.rpc_layers[name].reassign_inverse_workers(a_handler,g_handler)
            if a_handler == self.rank or g_handler == self.rank:
                self.assigned_layers.append(name)
            else:
                self.candidate_participate_factor_computation_layers.append(name)
        self.print_rpc_state(f"update new assignment {new_assignment_generation}: {new_assignment}")
        self.current_inverse_computation_layers = self.assigned_layers.copy()
        self.current_participate_factor_computation_layers = self.candidate_participate_factor_computation_layers.copy()
        self.update_send_trigger()
        self.update_assignment_callback = None

    def get_world_size(self):
        return len(self.node_states.keys())

    def get_health_world_size(self):
        return len(self.get_working_node_state_list())

    def is_factor_ready(self, layer_name, factor_type):
        if self.rpc_layers[layer_name].kfac_layer.get_factor(factor_type) is not None:
            return True
        else:
            return False

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
        """
        Return the rank of the worker assigned to compute the decomp of the factor.
        """
        return self.rpc_layers[layer_name].assigned_worker[factor_type]
    
    def get_node_states(self):
        with self.node_state_lock:
            return self.node_states.copy()

    def update_node_states(self,node_states:Dict[int,NodeState], from_leader = False):
        with self.node_state_lock:
            for rank, state in node_states.items():
                if rank == self.rank:
                    continue
                if rank not in self.node_states:
                    self.node_states[rank] = state
                    self.debug_print(f"Warning!!!! add new node {rank} to node_states")
                else:
                    if node_states[rank].iter > self.node_states[rank].iter:
                        self.node_states[rank].iter = state.iter
                        self.node_states[rank].speed = state.speed
                    if from_leader:
                        self.node_states[rank].health = state.health

    def send_kfac_factor(self,layer_name:str,factor_type:str):
        target = self.assigned_worker(layer_name, factor_type)
        if target == self.rank:
            return
        with self.node_state_lock:
            node_states = self.node_states.copy()
        with self.get_layer_lock(layer_name, factor_type):
            factor_tensor = self.rpc_layers[layer_name].kfac_layer.get_factor(factor_type)
        if factor_tensor is None:
            return
        try:
            rpc.rpc_async(
                to=rpc_work_name(target),
                func=receive_kfac_factor,
                args=(self.rank, layer_name, factor_tensor, node_states, factor_type)
            )
        except Exception as e:
            print(f"Failed to send factor to {target} from {self.rank}: {e}")
        self.com_statistic.add_send_stat("factor")
        return True
    
    def async_factor_send_register(self,layer_name:str,factor_type:str):
        target = self.assigned_worker(layer_name, factor_type)
        if target == self.rank:
            return
        self.package_sender.add_data(target, layer_name, [factor_type])
    
    def async_eigen_broadcast_register(self,layer_name:str,factor_types:List[str]):
        for rank in range(self.origin_world_size):
            if rank == self.rank:
                continue
            self.package_sender.add_data(rank, layer_name, factor_types)

    def send_data_package(self,target_rank):
        data = self.package_sender.get_packaged_data(target_rank=target_rank)
        if data is None or len(data.keys()) == 0:
            return
        with self.node_state_lock:
            self.node_states[self.rank].speed = self.get_local_node_speed()
            node_states = self.node_states.copy()
        try:
            rpc.rpc_async(
                to=rpc_work_name(target_rank),
                func=receive_packaged_tensors,
                args=(self.rank, data ,node_states)
            )
        except Exception as e:
            print(f"Failed to send factor to {target_rank} from {self.rank}: {e}")
        self.package_sender.clear_package(target_rank)
        self.com_statistic.add_send_stat("data_package")
        return True 

    def broadcast_kfac_eigen_tensor_a(self, layer_name):
        if self.assigned_worker(layer_name, 'A') != self.rank:
            return
        if self.rpc_layers[layer_name].kfac_layer.get_factor("qa") is None:
            raise RuntimeError(
                f'Attempt to broadcast A inv from src={self.rank} but this rank '
                'has not computed inv yet.',
            )
        qa = self.rpc_layers[layer_name].kfac_layer.get_factor("qa")
        da = self.rpc_layers[layer_name].kfac_layer.get_factor("da")
        with self.node_state_lock:
            node_states = self.node_states.copy()
        for target_rank in range(self.origin_world_size):
            if target_rank == self.rank:
                continue
            try :
                rpc.rpc_async(
                    to=rpc_work_name(target_rank),
                    func=receive_eigen_tensor_a,
                    args=(self.rank, layer_name, qa, da, node_states)
                )
            except Exception as e:
                print(f"Failed to send eigen tensor to {target_rank} from {self.rank}: {e}")
        self.com_statistic.add_send_stat("eigen",times=self.origin_world_size-1)

    def broadcast_kfac_eigen_tensor_g(self, layer_name):
        qg = self.rpc_layers[layer_name].kfac_layer.get_factor("qg")
        dg = self.rpc_layers[layer_name].kfac_layer.get_factor("dg")
        dadg = self.rpc_layers[layer_name].kfac_layer.get_factor("dgda")
        with self.node_state_lock:
            node_states = self.node_states.copy()
        if self.rpc_layers[layer_name].kfac_layer.prediv_eigenvalues:
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

        for target_rank in range(self.origin_world_size):
            if target_rank == self.rank:
                continue
            try:
                rpc.rpc_async(
                    to=rpc_work_name(target_rank),
                    func=receive_eigen_tensor_g,
                    args=(self.rank, layer_name, qg, dg, dadg, node_states)
                )
            except Exception as e:
                print(f"Failed to send eigen tensor to {target_rank} from {self.rank}: {e}")
        self.com_statistic.add_send_stat("eigen",times=self.origin_world_size-1)

    def factor_computation_nessary(self, layer_name):
        """
        需要算A和G的因子
        """
        if layer_name in self.current_participate_factor_computation_layers or layer_name in self.current_inverse_computation_layers:
            return True
        return False

    def computation_volume_statistic_and_speed(self):
        self.time_cost_accumulation = time.time() - self.trian_start_time
        for layer_name in self.current_inverse_computation_layers:
            self.computation_volume_accumulation += ((self.layers_workload[layer_name]["A"]  +self.layers_workload[layer_name]["G"]))
        self.node_states[self.rank].speed = int(self.computation_volume_accumulation / self.time_cost_accumulation)
        self.debug_print(f"computation volume: {self.computation_volume_accumulation}, time cost: {self.time_cost_accumulation}, speed: {self.node_states[self.rank].speed}")

    def get_local_node_speed(self):
        if self.node_states[self.rank].speed is not None and self.node_states[self.rank].speed != 0:
            return self.node_states[self.rank].speed
        elif self.time_cost_accumulation != 0:
            self.node_states[self.rank].speed = int(self.computation_volume_accumulation / self.time_cost_accumulation)
            return self.node_states[self.rank].speed
        return 0

    def factor_computation_lazy_rebalance(self):
        current_t = self.current_t()
        forward_than_local = sum(state.iter > current_t for state in self.get_working_node_state_list())
        late_than_local = sum(state.iter < current_t for state in self.get_working_node_state_list())
        iter_diff = self.max_iter_in_cluster() - current_t
        random.shuffle(self.candidate_participate_factor_computation_layers)
        random.shuffle(self.assigned_layers)
        self.current_participate_factor_computation_layers = \
            self.candidate_participate_factor_computation_layers[:len(self.current_participate_factor_computation_layers)]
        self.current_inverse_computation_layers = self.assigned_layers[:len(self.current_inverse_computation_layers)]
        if forward_than_local >= math.ceil(self.get_world_size() * 0.7) and iter_diff > 50: # local is too slow, work less
            if len(self.current_participate_factor_computation_layers) > 0:
                layer_name = random.choice(self.current_participate_factor_computation_layers)
                self.current_participate_factor_computation_layers.remove(layer_name)
                self.debug_print(f"remove {layer_name} from factor computation")
            else:
                self.data_send_scheduler.relax_send_interval("eigen")
            """
            elif len(self.current_inverse_computation_layers) > 0:
                layer_name = random.choice(self.current_inverse_computation_layers)
                self.current_inverse_computation_layers.remove(layer_name)
                self.debug_print(f"remove {layer_name} from inverse computation")
            """
            return
        if late_than_local >= self.origin_world_size * 0.1 or forward_than_local <= 2: #math.ceil(self.world_size * 0.3): # local is quick, work more
            """
            if len(self.current_inverse_computation_layers) < len(self.assigned_layers):
                for layer_name in reversed(self.assigned_layers):
                    if layer_name not in self.current_inverse_computation_layers:
                        self.current_inverse_computation_layers.append(layer_name)
                        self.debug_print(f"add {layer_name} to inverse computation")
                        break
            """
            if self.data_send_scheduler.intervals["eigen"] > self.data_send_scheduler.start_interval["eigen"]:
                self.data_send_scheduler.shorten_send_interval("eigen")
            elif len(self.current_participate_factor_computation_layers) < len(self.candidate_participate_factor_computation_layers):
                for layer_name in reversed(self.candidate_participate_factor_computation_layers):
                    if layer_name not in self.current_participate_factor_computation_layers:
                        self.current_participate_factor_computation_layers.append(layer_name)
                        self.debug_print(f"add {layer_name} to factor computation")
                        break

    def send_model_param(self):
        if self.data_send_scheduler.can_send("model_param"):
            self.model_avg_rpc.process()
            self.data_send_scheduler.update_next_send_time("model_param")

    def restart_sick_node(self): # call by sick nodes
        if self.node_states[self.rank].health == 2 and self.task_reassign_rpc.assignment_generation not in self.request_regression_record:
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
    
    def send_kfac_factor_action(self ,name, factor_type):
        if (self.data_send_scheduler.can_send("factor")):
            target_rank = self.rpc_layers[name].assigned_worker[factor_type]
            if target_rank == self.rank:
                return
            if not self.is_packaged_send:
                self.send_kfac_factor(name, factor_type)
            else:
                self.async_factor_send_register(name, factor_type)
                if self.rpc_layers[name].send_trigger[factor_type]:
                    self.send_data_package(target_rank)


global_communicator: KFacRPCCommunicator = None

def receive_kfac_factor(from_rank, layer_name, factor, from_node_states:Dict[int,NodeState], factor_type):
    if fault_simulator.is_fault():
        return
    global global_communicator
    self = global_communicator

    if self.rpc_layers[layer_name].assigned_worker[factor_type] != self.rank:
        return

    current_t = self.current_t()
    from_iter = from_node_states[from_rank].iter
    self.rpc_layers[layer_name].update_local_factor(factor, current_t, from_iter, factor_type ,world_size=self.origin_world_size)
    #self.update_node_iter(from_rank, from_iter)
    self.update_node_states(from_node_states)

def receive_packaged_tensors(from_rank, data:Dict[str,Dict[str,torch.Tensor]], node_states:Dict[int,NodeState]):
    if fault_simulator.is_fault():
        return
    global global_communicator
    self = global_communicator
    self.update_node_states(node_states)
    log_info = f"receive data {str(from_rank)}:\n"
    qa , da , qg , dg , dgda = None, None, None, None, None
    for layer_name, tensor_name_data_dict in data.items():
        for tensor_name, tensor in tensor_name_data_dict.items():
            current_t = self.current_t()
            if tensor_name in ['A', 'G']:
                self.rpc_layers[layer_name].update_local_factor(tensor, current_t, node_states[from_rank].iter, tensor_name ,world_size=self.origin_world_size)
                log_info += f"{layer_name} {tensor_name} from {from_rank}\n"
            elif tensor_name == 'qa':
                qa = tensor
            elif tensor_name == 'da':
                da = tensor
            elif tensor_name == 'qg':
                qg = tensor
            elif tensor_name == 'dg':
                dg = tensor
            elif tensor_name == 'dgda':
                dgda = tensor
        if qa is not None:
            log_info += f"{layer_name} qa from {from_rank}\n"
            self.rpc_layers[layer_name].update_local_eigen_a(qa, da, node_states[from_rank].iter)
        if qg is not None:
            log_info += f"{layer_name} qg from {from_rank}\n"
            self.rpc_layers[layer_name].update_local_eigen_g(qg, dg, dgda, node_states[from_rank].iter)

def receive_eigen_tensor_a(from_rank, layer_name, qa, da, recv_node_states:Dict[int,NodeState]):
    if fault_simulator.is_fault():
        return
    global global_communicator
    t = recv_node_states[from_rank].iter
    if t < global_communicator.rpc_layers[layer_name].recv_handled_a_version:
        return
    global_communicator.rpc_layers[layer_name].update_local_eigen_a(qa, da, t)
    global_communicator.update_node_states(recv_node_states)

def receive_eigen_tensor_g(from_rank, layer_name, qg, dg, dadg, recv_node_states:Dict[int,NodeState]):
    if fault_simulator.is_fault():
        return
    global global_communicator
    t = recv_node_states[from_rank].iter
    if t < global_communicator.rpc_layers[layer_name].recv_handled_g_version:
        return
    global_communicator.rpc_layers[layer_name].update_local_eigen_g(qg, dg, dadg, t)
    global_communicator.update_node_states(recv_node_states)

def receive_eigen_tensor_package(from_rank,t, eigen_tensor_package):
    """
    deprecated
    """
    if len(eigen_tensor_package) == 0:
        return
    global global_communicator
    global_communicator.update_node_iter(from_rank, t)
    for eigen_tensor in eigen_tensor_package:
        layer_name = eigen_tensor["layer_name"]
        qa, da = eigen_tensor["A"]
        qg, dg, dgda = eigen_tensor["G"]
        global_communicator.rpc_layers[layer_name].update_local_eigen_a(qa, da, t)
        global_communicator.rpc_layers[layer_name].update_local_eigen_g(qg, dg, dgda, t)

    if from_rank in global_communicator.send_rank_group[global_communicator.group_id]:
        return # broaddcast from same group

    for target_rank in global_communicator.send_rank_group[global_communicator.group_id]:
        if target_rank == global_communicator.rank:
            continue
        try:
            rpc.rpc_sync(
                to=rpc_work_name(target_rank),
                func=receive_eigen_tensor_package,
                args=(global_communicator.rank, t, eigen_tensor_package)
            )
        except Exception as e:
            print(f"Failed to send eigen tensor to {target_rank} from {global_communicator.rank}: {e}")
