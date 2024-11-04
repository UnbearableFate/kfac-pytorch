import time

from kfac.rpc_util.fault_sim import fault_simulator

train_time = 0

def get_train_time():
    return train_time

fault_simulator.train_total_time_cb = get_train_time

for i in range(1000):
    time.sleep(0.05)
    train_time += 0.05
    fault_simulator.update_fault_status()
    if fault_simulator.is_fault():
        print(f"Fault at {i}, fault time: {fault_simulator.fault_total_time} ,fault_over_time: {fault_simulator.fault_over_time}"
              f"train_total_time: {train_time}, fault_rate: {fault_simulator.fault_total_time / train_time}")
    else:
        print("Normal at", i)