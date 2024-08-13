def rpc_work_name(rank:int) -> str:
    return f"rpc_{rank}"

def local_full_connnection_device_map(world_size,rank):
    device_map = {}
    for i in range(world_size):
        if i == rank:
            continue
        device_map[rpc_work_name(i)] = {rank :i}
    return device_map

if __name__ == '__main__':
    print(local_full_connnection_device_map(4,1))