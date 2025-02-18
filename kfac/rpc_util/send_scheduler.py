from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from kfac.rpc_distributed import KFacRPCCommunicator
class DataSendScheduler:
    def __init__(self, send_intervals):
        self.intervals = dict(send_intervals)
        self.next_send = {k: v for k, v in send_intervals.items()}
        self.priority_order = list(send_intervals.keys())
        self.current_iter = 0

    def update_loop_counter(self):
        """更新当前迭代轮次"""
        self.current_iter += 1

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
