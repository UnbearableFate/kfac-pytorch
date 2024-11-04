import time
from random import random

class FaultSimulator:
    def __init__(self, fault_rate = 0.2, fault_continue_time = 2000):
        self.recover_flg = False
        self.fault_rate = fault_rate
        self.fault_continue_time = fault_continue_time
        self.fault_total_time = 0
        self.train_total_time_cb = None
        self.fault_over_time = None

    def update_fault_status(self):
        if self.fault_over_time is None and self.fault_total_time / self.train_total_time_cb() < self.fault_rate:
            fault_time = self.fault_continue_time * random()
            self.fault_over_time = time.time() + fault_time
            self.fault_total_time += fault_time
            self.recover_flg = False

        if self.fault_over_time is not None and time.time() > self.fault_over_time:
            self.fault_over_time = None
            self.recover_flg = True

    def is_fault(self):
        return self.fault_over_time is not None

fault_simulator = FaultSimulator(fault_rate=0.2, fault_continue_time=2)