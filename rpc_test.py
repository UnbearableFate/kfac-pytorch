import argparse
import datetime
import threading

import torch
import torch.distributed.rpc as rpc
import os
import torch.distributed as dist

ompi_world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', -1))
ompi_world_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', -1))

if ompi_world_size == -1 or ompi_world_rank == -1:
    raise RuntimeError("This script is intended to be launched with mpirun")

DATA_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/data"
Share_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/share_files"

def rpc_work_name(rank:int) -> str:
    return f"rpc_{rank}"

def full_connection_device_map(world_size,rank):
    device_map = {}
    for i in range(world_size):
        if i == rank:
            continue
        device_map[rpc_work_name(i)] = {ompi_world_rank%4:i%4}
    return device_map

device = torch.device(f"cuda:{ompi_world_rank % 4}")
local_tensor = torch.rand([1024,1024,32]).to(device)

def get_cuda_tensor(tensor,from_rank,ct=0):
    global local_tensor
    print(f"get tensor shape {tensor.shape} from {from_rank} ,deivce :{tensor.device.type},count {ct} " )
    local_tensor = local_tensor*1/2 + tensor*1/2

def recv_tensor(tensor_shape,from_rank):
    def get_tensor():
        tensor = torch.zeros(tensor_shape).to(device)
        dist.recv(tensor,from_rank)
        print(f"recv tensor shape {tensor.shape} from {from_rank} ,deivce :{tensor.device.type}")
    thread = threading.Thread(target=get_tensor)
    thread.daemon = False
    print(f"start recv tensor from {from_rank} at {ompi_world_rank}")
    thread.start()

def send_tensor(tensor,target):
    try:
        res = rpc.rpc_async(rpc_work_name(target),
                    recv_tensor,
                    args=(tensor.shape, ompi_world_rank))
    except Exception as e:
        print(f"send tensor to {target} error {e}")

    res.wait()
    dist.send(tensor, target)

if __name__ == '__main__':
    dmap = full_connection_device_map(ompi_world_size,ompi_world_rank)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    parser = argparse.ArgumentParser(description="experiment script")
    parser.add_argument('--timestamp', type=str, default=timestamp)
    args = parser.parse_args()
    timestamp = args.timestamp
    print(f"timestamp: {timestamp}")

    dist.init_process_group(backend="gloo", init_method=f"file://{Share_DIR}/pg_test_{timestamp}", rank=ompi_world_rank, world_size=ompi_world_size)
    if dist.is_initialized():
        print(f"init process group success at {ompi_world_rank}")
    
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        init_method=f"file://{Share_DIR}/rpc_share_test_{timestamp}",
        rpc_timeout=30,
        device_maps=dmap
    )

    rpc.init_rpc(rpc_work_name(ompi_world_rank), rank=ompi_world_rank, world_size=ompi_world_size, rpc_backend_options=options)
    print(f"Hello from {ompi_world_rank}")

    if ompi_world_rank == 0:
        send_tensor(local_tensor,1)

    rpc.shutdown()
    
    dist.destroy_process_group()