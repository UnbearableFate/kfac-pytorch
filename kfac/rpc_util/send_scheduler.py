from typing import TYPE_CHECKING, Dict, List

import torch
if TYPE_CHECKING:
    from kfac.rpc_distributed import KFacRPCCommunicator
class DataSendScheduler:
    def __init__(self, is_kfac:bool = True):
        if is_kfac:
            send_intervals = {'model_param': 29, 'factor': 25, 'eigen': 100}
            self.next_send = {'model_param': 9, 'factor': 2, 'eigen': 5}
        else:
            send_intervals = {'model_param': 27}
            self.next_send = {'model_param': 9}
        self.intervals = dict(send_intervals)
        self.start_interval = dict(send_intervals)
        #self.next_send = {k: v for k, v in send_intervals.items()}
        self.priority_order = list(send_intervals.keys())
        self.current_iter = 0

    def update_loop_counter(self):
        """更新当前迭代轮次"""
        self.current_iter += 1
    
    def relax_send_interval(self, type_name = None):
        """放宽所有数据类型的发送时间间隔"""
        if type_name is not None:
            self.intervals[type_name] = int(self.intervals[type_name] * 1.2)
        else :
            for dt in self.intervals.keys():
                self.intervals[dt]  = int(self.intervals[dt] * 1.2)
        return f"relax send interval to {self.intervals}"
    
    def shorten_send_interval(self ,type_name = None):
        if type_name is not None:
            self.intervals[type_name] = max(int(self.intervals[type_name] / 1.1),self.start_interval[type_name])
        else :
            for dt in self.intervals.keys():
                self.intervals[dt] = max(int(self.intervals[dt] / 1.1),self.start_interval[dt])
        return f"shorten send interval to {self.intervals}"
            

    def get_next_send_type(self):
        """获取当前可发送的数据类型（优先级高的优先）"""
        current_iter = self.current_iter
        candidates = []
        
        # 按优先级顺序检查所有数据类型
        for data_type in self.priority_order:
            if self.next_send[data_type] <= current_iter:
                candidates.append(data_type)
        
        if not candidates:
            return None

        selected = candidates[0]

        # 将其他冲突的候选者延迟到下一次迭代
        for dt in candidates[1:]:
            self.next_send[dt] = current_iter + 1
        return selected

    def can_send(self, data_type):
        """检查指定的数据类型是否可以发送"""
        return self.get_next_send_type() == data_type

    def update_next_send_time(self, data_type):
        """标记某个数据类型已发送，并更新其下次发送时间"""
        if data_type not in self.next_send:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # 基于当前迭代计算新的发送时间
        self.next_send[data_type] = self.current_iter + self.intervals[data_type]

class PackageSender:
    def __init__(self, communicator: 'KFacRPCCommunicator'):
        self.communicator = communicator
        self.packages: Dict[int, Dict[str, List]]= dict () # target -> (layer_name -> data_names) {1 : {'layer_0': ["A", "G"]}}
        for rank in range(communicator.origin_world_size):
            if rank == communicator.rank:
                continue
            self.packages[rank] = dict()
        
    def add_data(self, target_rank, layer_name, data_types:List):
        if target_rank not in self.packages:
            raise ValueError(f"Unknown target rank: {target_rank}")
        
        if isinstance(data_types,str):
            data_types = [data_types]
    
        if layer_name not in self.packages[target_rank]:
            self.packages[target_rank][layer_name] = data_types
        else:
            self.packages[target_rank][layer_name].extend(data_types)
        
    def get_packaged_data(self, target_rank):
        data : Dict[str, Dict[str, torch.Tensor]] = dict()
        log_info = f"Sending data to rank {target_rank}:"
        if target_rank not in self.packages:
            raise ValueError(f"Unknown target rank: {target_rank}")
        for layer_name, data_name_list in self.packages[target_rank].items():
            kfac_layer = self.communicator.rpc_layers[layer_name].kfac_layer
            data[layer_name] = dict()
            log_info += f"\n  {layer_name}:"
            for tensor_name in data_name_list:
                data[layer_name][tensor_name] = kfac_layer.get_factor(tensor_name)
                log_info += f" {tensor_name} : {data[layer_name][tensor_name].shape}"
        return data
    
    def clear_package(self, target_rank):
        if target_rank in self.packages:
            self.packages[target_rank].clear()