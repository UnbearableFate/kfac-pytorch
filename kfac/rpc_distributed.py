import datetime
import math
import random
import statistics
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

class NodeState():
    def __init__(self,rank):
        self.rank = rank
        self.iter = -1
        self.factor_computation_iter = -1
        self.inverse_computation_iter = -1
        self.model_param_avg_iter = -1
        self.health = True

    def __str__(self):
        return f"rank {self.rank}'s iter:{self.iter}"

class KfacRPCLayer:
    def __init__(self,a_handler,g_handler ,name , prediv_eigenvalues):
        self.factor : Dict[str:Optional['torch.Tensor']] = {"A" : None, "G": None }
        self.factor_recv_ct : Dict[str:Dict[int, int]] = {"A" : {}, "G": {} } # {t: count}
        self.assigned_worker :Dict[str : int]= {'A': a_handler, 'G': g_handler}
        self.qa = None
        self.da = None
        self.recv_handled_a_version = -2
        self.last_load_handled_tensor_version = -2
        self.qg = None
        self.dg = None
        self.dgda = None
        self.prediv_eigenvalues = prediv_eigenvalues # False：da, dg must be provided; True: dgda must be provided
        self.recv_handled_g_version = -2
        self.outdated_weight_param = 1
        self.ahead_weight_param = 1
        self.name = name

    def reassign_inverse_workers(self, a_handler, g_handler):
        self.assigned_worker['A'] = a_handler
        self.assigned_worker['G'] = g_handler

    def is_handled_a_updated(self):
        return self.qa is not None and self.recv_handled_a_version > self.last_load_handled_tensor_version

    def is_handled_g_updated(self):
        return self.qg is not None and self.recv_handled_g_version > self.last_load_handled_tensor_version

    def update_local_factor(self, recv_factor, local_t, recv_t, factor_type):
        self.factor_recv_ct[factor_type][recv_t] = self.factor_recv_ct[factor_type].get(recv_t, 0) + 1
        if self.factor[factor_type] is None:
            self.factor[factor_type] = recv_factor
            return
        recv_world_weight = 1.0 / self.factor_recv_ct[factor_type][recv_t]
        if local_t > recv_t:
            recv_world_weight *= self.outdated_weight_param
        if local_t < recv_t:
            recv_world_weight *= self.ahead_weight_param
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
        n = 3
        for factor_type in self.factor_recv_ct.keys():
            self.factor_recv_ct[factor_type] = {k: v for k, v in self.factor_recv_ct[factor_type].items() if k >= local_t - n}

    def load_eigen_tensor(self, kfac_layer: 'KFACEigenLayer', t):
        assert self.name == kfac_layer.name
        while self.qa is None or self.qg is None:
            time.sleep(0.1)
            #logger.debug(f"Waiting for eigen tensor of {self.name} to be ready in rank {global_communicator.rank}")
        kfac_layer.qa = self.qa.clone()
        kfac_layer.qg = self.qg.clone()
        if self.prediv_eigenvalues:
            kfac_layer.dgda = self.dgda.clone()
        else:
            kfac_layer.dg = self.dg.clone()
            kfac_layer.da = self.da.clone()
        self.last_load_handled_tensor_version = t

class KFacRPCCommunicator:
    def __init__(self, world_size, rank, preconditioner:'BaseKFACPreconditioner' ,model):
        self.io_layers = None
        self.skip_inverse_computation_flag : bool = False
        rpc.init_rpc(name=f"rpc_{rank}", rank=rank, world_size=world_size)
        self.origin_world_size = world_size
        self.rank = rank
        self.rpc_layers: Dict[str:KfacRPCLayer] = {} # {layer_name: KfacRPCLayer}
        self.computer_type = "" #eigen / inverse
        self.assigned_layers = []
        self.candidate_participate_factor_computation_layers = []
        self.participate_factor_computation_layers = []
        for name, kfac_layer in preconditioner._layers.values():
            a_handler = preconditioner._assignment.inv_worker(name, 'A')
            g_handler = preconditioner._assignment.inv_worker(name, 'G')
            self.rpc_layers[name] = KfacRPCLayer(a_handler,g_handler ,name ,kfac_layer.prediv_eigenvalues)
            if a_handler == self.rank or g_handler == self.rank:
                self.assigned_layers.append(name)
            else:
                self.candidate_participate_factor_computation_layers.append(name)
                self.participate_factor_computation_layers.append(name)

        self.node_states = dict()
        for i in range(world_size):
            self.node_states[i] = NodeState(i)
        self.lock = threading.Lock()

        # hyperparameters
        if self.rank == 2:
            self.necessary_ct = 1
        else:
            self.necessary_ct = 2
        self.load_inverse_max_loop = 3
        if rpc.is_available():
            print(f"RPC Communicator initialized for rank {rank}")

        self.init_logger(rank)
        self.model_avg_rpc = model_param_avg_rpc.ModelAvgRPCCommunicator(rank, model ,self)
        self.task_reassign_rpc = task_manager.RPCTaskManager(rpc_communicator=self, assignment=preconditioner._assignment)

        self.model_accuracy_statistic : Dict[int , Dict[str ,int]]= dict() # {epoch: (recv_ct ,correct_ct, total_ct)}

        global global_communicator
        global_communicator = self

    def get_health_node_list(self) -> list[NodeState]:
        return [state for rank, state in self.node_states.items() if state.health]

    def get_sick_node_list(self) -> list[NodeState]:
        return [state for rank, state in self.node_states.items() if not state.health]

    def max_iter_in_cluster(self):
        # return max iter in node_states
        return max([state.iter for state in self.get_health_node_list()])

    def min_iter_in_cluster(self):
        return min([state.iter for state in self.get_health_node_list()])

    def median_iter_in_cluster(self):
        iters = [state.iter for state in self.get_health_node_list()]
        return statistics.median(iters)

    def init_logger(self,rank):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        # 创建一个 FileHandler，并设置级别为 DEBUG
        file_handler = logging.FileHandler(f'log_{rank}_{timestamp}.log')
        file_handler.setLevel(logging.DEBUG)

        # 创建一个日志格式器，并将其添加到 FileHandler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        global logger
        # 将 FileHandler 添加到日志记录器
        logger.addHandler(file_handler)

        # 移除默认的 StreamHandler（终端输出）
        if logger.hasHandlers():
            logger.handlers.clear()
            logger.addHandler(file_handler)

    def print_rpc_state(self, text = ""):
        global logger
        log_txt = ""
        for node_rank, state in self.node_states.items():
            log_txt += f"{state}; "
        logger.debug(f"{log_txt} in rank {self.rank}, {text}")

    def debug_print(self, text):
        pass
        global logger
        if self.min_iter_in_cluster() == self.current_t():
            logger.debug(f"current iter {self.current_t()} in rank {self.rank}, {text}")
    
    def __repr__(self):
        log = f"Rank {self.rank} : iter {self.current_t()}\n"
        for name, layer in self.rpc_layers.items():
            log += f"Layer {name}:\n"
            for factor_type, factor in layer.factor.items():
                log += f"\t{factor_type} factor: {factor}\n"
                log += f"\t{factor_type} factor recv ct: {layer.factor_recv_ct[factor_type]}\n"
            log += f"\tA eigen: {layer.qa}, {layer.da}\n"
            log += f"\tG eigen: {layer.qg}, {layer.dg}, {layer.dgda}\n"
            log += f"\thandled A recv iter: {layer.recv_handled_a_version}\n"
            log += f"\thandled G recv iter: {layer.recv_handled_g_version}\n"
            log += f"\tA handler: {layer.assigned_worker['A']}\n"
            log += f"\tG handler: {layer.assigned_worker['G']}\n"
        return log

    def load_factor(self,kfac_layer: 'KFACBaseLayer', factor_type):
        if self.assigned_worker(kfac_layer.name, factor_type) == self.rank:
            while not self.is_factor_ready(kfac_layer.name, factor_type):
                time.sleep(0.1)
            assert self.rpc_layers[kfac_layer.name].factor[factor_type] is not None
            kfac_layer.a_factor = self.rpc_layers[kfac_layer.name].factor["A"].clone().detach()
        return True

    def load_eigen_tensor(self,kfac_layer, loop):
        layer_name = kfac_layer.name
        current_t = self.current_t()

        if self.rpc_layers[layer_name].last_load_handled_tensor_version == current_t:
            return False # already loaded

        if (self.rpc_layers[layer_name].recv_handled_g_version >= self.current_t() -loop
                or self.rpc_layers[layer_name].recv_handled_a_version == self.current_t() - loop
                or loop >= self.load_inverse_max_loop -1):
            self.rpc_layers[layer_name].load_eigen_tensor(kfac_layer,current_t)
            return True
        return False

    def clear_count_dict(self):
        for layer in self.rpc_layers.values():
            layer.clear_count_dict(self.current_t())

    def shutdown(self):
        rpc.shutdown()
    def update_self_t(self):
        self.node_states[self.rank].iter += 1

    def current_t(self):
        return self.node_states[self.rank].iter

    def update_other_rank_iter(self, from_rank,t):
        if from_rank not in self.node_states:
            raise RuntimeError(f"Rank {from_rank} is not in the node_states")
        self.node_states[from_rank].iter = max(self.node_states[from_rank].iter, t)

    def update_node_state_list(self, new_health_node_list):
        for node_rank in self.node_states:
            if node_rank in new_health_node_list:
                self.node_states[node_rank].health = True
            else:
                self.node_states[node_rank].health = False

    def update_inverse_workers(self, new_assignment):
        self.task_reassign_rpc.assignment._inv_assignments = new_assignment
        self.assigned_layers.clear()
        self.candidate_participate_factor_computation_layers.clear()
        self.participate_factor_computation_layers.clear()
        for name, kfac_layer in self.rpc_layers.items():
            a_handler = new_assignment[name]['A']
            g_handler =  new_assignment[name]['G']
            self.rpc_layers[name].reassign_inverse_workers(a_handler,g_handler)
            if a_handler == self.rank or g_handler == self.rank:
                self.assigned_layers.append(name)
            else:
                self.candidate_participate_factor_computation_layers.append(name)
                self.participate_factor_computation_layers.append(name)

    def get_world_size(self):
        return len(self.node_states.keys())

    def get_health_world_size(self):
        return len(self.get_health_node_list())

    def is_factor_ready(self, layer_name, factor_type):
        current_t = self.current_t()
        return self.rpc_layers[layer_name].factor_recv_ct[factor_type].get(current_t,0) >= self.necessary_ct
    
    def assigned_worker(self, layer_name, factor_type):
        return self.rpc_layers[layer_name].assigned_worker[factor_type]
    
    def send_kfac_factor(self,layer_name:str,factor_tensor :torch.Tensor, factor_type:str):
        if layer_name not in self.participate_factor_computation_layers:
            return True
        target = 0
        if factor_type == "A":
            target = self.rpc_layers[layer_name].assigned_worker['A']
        elif factor_type == "G":
            target = self.rpc_layers[layer_name].assigned_worker['G']
        t = self.current_t()
        if target == self.rank:
            self.rpc_layers[layer_name].update_local_factor(factor_tensor.clone(), t, t, factor_type)
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

    def send_kfac_eigen_tensor(self, layer_name,q:torch.Tensor,d:torch.Tensor,dd :torch.Tensor, factor_type):
        if self.assigned_worker(layer_name, factor_type) != self.rank:
            return True
        if q is None:
            raise RuntimeError(
                f'Attempt to broadcast {factor_type} inv from src={self.rank} but this rank '
                'has not computed inv yet.',
            )

        #with self.lock:
        t = self.current_t()
        d_clone,dd_clone = None, None
        if d is not None and isinstance(d, torch.Tensor):
            d_clone = d.clone()
        if dd is not None and isinstance(dd, torch.Tensor):
            dd_clone = dd.clone()
        if factor_type == "A":
            self.rpc_layers[layer_name].update_local_eigen_a(q.clone(), d_clone, t)
        if factor_type == "G":
            self.rpc_layers[layer_name].update_local_eigen_g(q.clone(), d_clone, dd_clone, t)

        for i in range(self.get_world_size()-1):
            target_rank = (self.rank + i + 1) % self.get_world_size()
            if factor_type == "A":
                try :
                    rpc.rpc_async(
                        to=rpc_work_name(target_rank),
                        func=receive_eigen_tensor_a,
                        args=(self.rank, layer_name, q, d, t)
                    )
                except Exception as e:
                    print(f"Failed to send eigen tensor to {target_rank} from {self.rank}: {e}")
            if factor_type == "G":
                try :
                    rpc.rpc_async(
                        to=rpc_work_name(target_rank),
                        func=receive_eigen_tensor_g,
                        args=(self.rank, layer_name, q, d,dd, t)
                    )
                except Exception as e:
                    print(f"Failed to send eigen tensor to {target_rank} from {self.rank}: {e}")

    def is_factor_computation_skipped(self, layer_name):
        if self.skip_inverse_computation_flag or (layer_name not in self.participate_factor_computation_layers and layer_name not in self.assigned_layers):
            return True
        return False

    def facotr_comput_lazy_wl_rebal(self):
        current_t = self.current_t()
        forward_than_local = sum(state.iter > current_t for state in self.get_health_node_list())
        late_than_local = sum(state.iter < current_t for state in self.get_health_node_list())
        iter_diff = self.max_iter_in_cluster() - current_t

        random.shuffle(self.candidate_participate_factor_computation_layers)
        self.participate_factor_computation_layers = \
            self.candidate_participate_factor_computation_layers[:len(self.participate_factor_computation_layers)]
        if forward_than_local >= math.ceil(self.get_world_size() * 0.7) and iter_diff > 3: # local is too slow, work less
            if len(self.participate_factor_computation_layers) > 0:
                layer_name = random.choice(self.participate_factor_computation_layers)
                self.participate_factor_computation_layers.remove(layer_name)
            else:
                self.skip_inverse_computation_flag = True
            
        if late_than_local >= 1 or forward_than_local <= 2: #math.ceil(self.world_size * 0.3): # local is quick, work more
            if self.skip_inverse_computation_flag:
                self.skip_inverse_computation_flag = False
            elif len(self.participate_factor_computation_layers) < len(self.candidate_participate_factor_computation_layers):
                for layer_name in reversed(self.candidate_participate_factor_computation_layers):
                    if layer_name not in self.participate_factor_computation_layers:
                        self.participate_factor_computation_layers.append(layer_name)
                        break

    def send_model_param(self):
        self.model_avg_rpc.send_all_model_param_alg02()

    def send_rpc_test_result(self, correct_ct, total_ct, epoch):
        try:
            rpc.rpc_async(
                to=rpc_work_name(0),
                func=recv_rpc_test_result,
                args=(self.rank, correct_ct, total_ct, epoch)
            )
        except Exception as e:
            print(f"Failed to send test result to 0: {e} from {self.rank}")

    def wait_and_return_test_result(self, epoch):
        wait_time = 0
        while epoch not in self.model_accuracy_statistic or self.model_accuracy_statistic[epoch]['recv_ct'] < self.origin_world_size:
            time.sleep(0.1)
            wait_time += 1
            if wait_time > 30:
                break
        
        if epoch in self.model_accuracy_statistic:
            return self.model_accuracy_statistic[epoch]['correct_ct'] / self.model_accuracy_statistic[epoch]['total_ct']
        else:
            return 0

global_communicator: KFacRPCCommunicator = None

def receive_kfac_factor(from_rank, layer_name, factor, from_iter, factor_type):
    global global_communicator
    self = global_communicator

    if self.rpc_layers[layer_name].assigned_worker[factor_type] != self.rank:
        print(f"Received factor from {from_rank} for {layer_name} but this rank is not assigned to this layer")
        '''
        if from_rank in self.health_node_states and from_iter >= self.current_t():
            # it is possible that this node don't receive new assignment yet
            self.rpc_layers[layer_name].reassign_inverse_workers(from_rank, factor_type)
            self.task_reassign_rpc.assignment._inv_assignments[layer_name][factor_type] = self.rank
        '''''

    #with self.lock:
    current_t = self.current_t()
    self.rpc_layers[layer_name].update_local_factor(factor, current_t, from_iter, factor_type)
    self.update_other_rank_iter(from_rank, from_iter)

def receive_eigen_tensor_a(from_rank, layer_name, qa, da, t):
    global global_communicator
    if t < global_communicator.rpc_layers[layer_name].recv_handled_a_version:
        return
    global_communicator.rpc_layers[layer_name].update_local_eigen_a(qa, da, t)
    global_communicator.update_other_rank_iter(from_rank, t)

def receive_eigen_tensor_g(from_rank, layer_name, qg, dg, dadg, t):
    global global_communicator
    #with global_communicator.lock:
    if t < global_communicator.rpc_layers[layer_name].recv_handled_g_version:
        return
    global_communicator.rpc_layers[layer_name].update_local_eigen_g(qg, dg, dadg, t)
    global_communicator.update_other_rank_iter(from_rank,t)

def recv_rpc_test_result(from_rank, correct_ct, total_ct, epoch):
    global global_communicator
    if epoch not in global_communicator.model_accuracy_statistic:
        global_communicator.model_accuracy_statistic[epoch] = {'recv_ct': 1, 'correct_ct': correct_ct, 'total_ct': total_ct}
    else:
        global_communicator.model_accuracy_statistic[epoch]['recv_ct'] +=1
        global_communicator.model_accuracy_statistic[epoch]['correct_ct'] += correct_ct
        global_communicator.model_accuracy_statistic[epoch]['total_ct'] += total_ct