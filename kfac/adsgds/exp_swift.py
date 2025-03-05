import torch
from torch.distributed import rpc
from typing import TYPE_CHECKING
from kfac.rpc_util.GraphConstruct import exponential_topology_sources,exponential_topology_targets
from kfac.adsgds.common import ModelStore, rpc_work_name,RootModelAvgRPCCommunicator 
from torch.nn.utils import vector_to_parameters
from typing import Dict
from scipy.special import expit
from general_util.consts import extreme_threshold

if TYPE_CHECKING:
    from kfac.rpc_distributed import KFacRPCCommunicator

class ExpTopoSwiftManager(RootModelAvgRPCCommunicator):
    def __init__(self, rank: int, model: torch.nn.Module, rpc_communicator: 'KFacRPCCommunicator'):
        super().__init__(rank, model, rpc_communicator)
        self.targets = exponential_topology_targets(self.origin_world_size, rank)
        self.sources = exponential_topology_sources(self.origin_world_size, rank)
        self.local_model_store.weight = 1/(len(self.sources)+1)
        self.neighbor_model_buffers : Dict[int, ModelStore] = {}
        for index,neighbor in enumerate(self.sources):
            self.neighbor_model_buffers[neighbor] = ModelStore(self.local_model_store.flatten_tensor)
            self.neighbor_model_buffers[neighbor].weight = 1/(len(self.sources)+1)
        
        self.index = 0
        global model_avg_rpc_communicator
        model_avg_rpc_communicator = self

    def process(self):
        self.update_local_flat_model()
        result = self.local_model_store.flatten_tensor * self.local_model_store.weight
        for neighbor in self.targets:
            rpc.rpc_async(
                to=rpc_work_name(neighbor),
                func= recv_model_param,
                args=(*self.local_model_store.getData(),self.rank)
            )
        
        for neighbor_store in self.neighbor_model_buffers.values():
            if neighbor_store.loss_value == 0:
                return

        for neighbor_store in self.neighbor_model_buffers.values():
            with neighbor_store.lock:
                result.add_(neighbor_store.flatten_tensor, alpha=neighbor_store.weight)

        with torch.no_grad() and self.local_model_store.lock:
            vector_to_parameters(result, self.model.parameters())

    def update_dynamic_weight(self,aggregating_node_list):
        sum_of_weight = 0
        self.local_model_store.dynamic_weight  = 1/ self.local_model_store.loss_value
        sum_of_weight += self.local_model_store.dynamic_weight

        for rank in aggregating_node_list:
            neighbor_store = self.neighbor_model_buffers[rank]
            self.neighbor_model_buffers[rank].dynamic_weight = expit((neighbor_store.term - self.local_model_store.term)/self.local_model_store.term) * (1/ self.neighbor_model_buffers[rank].loss_value)
            sum_of_weight += self.neighbor_model_buffers[rank].dynamic_weight
        
        self.local_model_store.dynamic_weight /= sum_of_weight
        for rank in aggregating_node_list:
            self.neighbor_model_buffers[rank].dynamic_weight /= sum_of_weight

    def select_aggregating_node(self):
        aggregating_node_list = []
        for rank,neighbor_node in self.neighbor_model_buffers.items():
            if (neighbor_node.loss_value != 0
                and neighbor_node.has_aggregated == False
                and self.local_model_store.term - neighbor_node.term  < extreme_threshold
                and neighbor_node.loss_value / self.local_model_store.loss_value < 2):
                aggregating_node_list.append(rank)
        return aggregating_node_list 
    
    def process_with_dynamic_weight(self):
        self.update_local_flat_model()
        node_states = self.rpc_communicator.get_node_states()
        result = self.local_model_store.flatten_tensor.clone()
        for neighbor in self.targets:
            work = rpc.rpc_async(
                to=rpc_work_name(neighbor),
                func= recv_model_param,
                args=(*self.local_model_store.getData(), self.rank , node_states)
            )

        for neighbor_store in self.neighbor_model_buffers.values():
            if neighbor_store.loss_value == 0:
                return
            
        aggregating_node_list = self.select_aggregating_node()
        self.update_dynamic_weight(aggregating_node_list)
        log_info = f"aggr: {aggregating_node_list} ,weight: {self.local_model_store.dynamic_weight}, "
        result.mul_(self.local_model_store.dynamic_weight)

        for rank in aggregating_node_list:
            neighbor_store = self.neighbor_model_buffers[rank]
            with neighbor_store.lock:
                result.add_(neighbor_store.flatten_tensor, alpha=neighbor_store.dynamic_weight)
                neighbor_store.has_aggregated = True
                log_info += f"{neighbor_store.dynamic_weight} "

        with torch.no_grad() and self.local_model_store.lock:
            vector_to_parameters(result, self.model.parameters())

        self.rpc_communicator.debug_print(f"model avg process done. {log_info} ,memeory usage is {self.rpc_communicator.get_memory_usage_percent()}")
        self.rpc_communicator.com_statistic.add_send_stat("model_param", len(self.targets)) 

model_avg_rpc_communicator: ExpTopoSwiftManager

def recv_model_param(data, term, loss_value,from_rank,from_node_states):
    global model_avg_rpc_communicator
    model_avg_rpc_communicator.rpc_communicator.update_node_states(from_node_states)
    if from_rank not in model_avg_rpc_communicator.neighbor_model_buffers:
        return None
    model_avg_rpc_communicator.neighbor_model_buffers[from_rank].setDataWithLock(data, term, loss_value)