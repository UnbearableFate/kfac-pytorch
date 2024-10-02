class DataSendScheduler:
    def __init__(self, send_intervals):
        """
        初始化调度器。

        参数：
        - send_intervals: 字典，键为数据类型（'model_param'、'factor'、'eigen'），值为对应的发送间隔。
          例如：{'model_param': 5, 'factor': 10, 'eigen': 15}
        """
        # 设置发送间隔
        self.send_intervals = send_intervals  # {'model_param': 5, 'factor': 10, 'eigen': 15}

        # 初始化循环计数器
        self.loop_counter = 0

        # 初始化下一次发送时间
        self.next_send_times = {data_type: send_intervals[data_type]-1 for data_type in send_intervals}

        # 已经安排的发送时间集合，用于避免时间冲突
        self.scheduled_times = set()

        # 全局的最新发送时间，用于确保发送时间不重叠
        self.latest_send_time = 0

    def update_loop_counter(self):
        """更新循环计数器，在每次训练循环开始时调用。"""
        self.loop_counter += 1

        # 清除已经过去的发送时间，防止集合无限增长
        self.scheduled_times = {t for t in self.scheduled_times if t >= self.loop_counter}

    def can_send(self, data_type):
        """
        检查指定的数据类型是否可以发送。

        参数：
        - data_type: 字符串，取值为 'model_param'、'factor' 或 'eigen'。

        返回：
        - 布尔值，表示是否可以发送。
        """
        can_send_type = self.get_next_send_type()
        return can_send_type == data_type

    def update_next_send_time(self, data_type):
        """
        在发送数据后更新下一次可以发送的时间，确保下一次发送时间不与其他数据类型冲突。

        参数：
        - data_type: 字符串，取值为 'model_param'、'factor' 或 'eigen'。
        """
        interval = self.send_intervals[data_type]

        # 下一次发送时间基于最新的发送时间加上间隔
        next_time = max(self.loop_counter, self.latest_send_time) + interval

        # 更新最新的发送时间
        self.latest_send_time = next_time

        self.next_send_times[data_type] = next_time
        self.scheduled_times.add(next_time)

    def get_next_send_type(self):
        """
        获取当前可以发送的数据类型，确保同一时间只能发送一种数据。

        返回：
        - data_type: 可以发送的数据类型，如果没有则返回 None。
        """
        # 找出所有满足发送条件的数据类型
        ready_types = [data_type for data_type, next_time in self.next_send_times.items()
                       if next_time <= self.loop_counter]

        # 确保 ready_types 的长度不超过 1
        if not ready_types:
            return None
        elif len(ready_types) == 1:
            return ready_types[0]
        else:
            # 理论上不会出现 ready_types 长度大于 1 的情况，因为我们已经避免了发送时间的重叠
            # 但为了保险起见，可以抛出异常或按照优先级选择一个
            # 在这里，我们按照优先级选择
            priority = ['model_param','eigen','factor']
            for data_type in priority:
                if data_type in ready_types:
                    return data_type
            return ready_types[0]  # 如果未匹配到，返回第一个
    def reset(self):
        """重置调度器状态。"""
        self.loop_counter = 0
        self.next_send_times = {data_type: 0 for data_type in self.send_intervals}
        self.scheduled_times = set()
        self.latest_send_time = 0