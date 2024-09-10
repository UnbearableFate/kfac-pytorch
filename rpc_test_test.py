import argparse
import datetime
import threading
import time

import torch
import torch.distributed.rpc as rpc
import torch.distributed as dist
import os

ompi_world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', -1))
ompi_world_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', -1))

if ompi_world_size == -1 or ompi_world_rank == -1:
    raise RuntimeError("This script is intended to be launched with mpirun")

DATA_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/data"
Share_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/share_files"


def rpc_work_name(rank: int) -> str:
    return f"rpc_{rank}"


def full_connection_device_map(world_size, rank):
    device_map = {}
    for i in range(world_size):
        if i == rank:
            continue
        device_map[rpc_work_name(i)] = {ompi_world_rank % 4: i % 4}
    return device_map


device = torch.device(f"cuda:{ompi_world_rank % 4}")
local_tensor = torch.rand([1024, 1024, 32]).to(device)


class TensorDataMeta:
    def __init__(self, tensor, layer_name, type_name1, type_name2):
        # type_name1 : factor, eigen, model
        # type_name2 : qa, qg ,da, dg, dadg, weight, bias
        self.shape = tensor.shape
        self.layer_name = layer_name
        self.type_name1 = type_name1
        self.type_name2 = type_name2
        self.tensor = None  # only recv side keep tensor

    def __str__(self):
        return f"Tensor info: layer : {self.layer_name}'s {self.type_name1}_{self.type_name2}, shape: {self.shape}"

recv_buffer = []
recv_buffer_lock = threading.Lock()


def recv_tensor(tensor_meta: TensorDataMeta, from_rank):
    global recv_buffer
    global recv_buffer_lock
    def get_tensor():
        tensor = torch.zeros(tensor_meta.shape).to(device)
        dist.recv(tensor, from_rank)
        tensor_meta.tensor = tensor
        print(f"recv tensor {tensor_meta} from {from_rank} to {dist.get_rank()}")
        with recv_buffer_lock:
            recv_buffer.append(tensor_meta)

    thread = threading.Thread(target=get_tensor)
    thread.start()
    time.sleep(0.1)

# type_name : factor_A  factor_G,
def send_tensor(tensor_meta: TensorDataMeta, tensor, from_rank, to_rank):
    rpc.rpc_sync(rpc_work_name(to_rank),
                        recv_tensor,
                        args=(tensor_meta, from_rank))

    dist.send(tensor,to_rank)

if __name__ == '__main__':
    dmap = full_connection_device_map(ompi_world_size, ompi_world_rank)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    parser = argparse.ArgumentParser(description="experiment script")
    parser.add_argument('--timestamp', type=str, default=timestamp)
    args = parser.parse_args()
    timestamp = args.timestamp
    print(f"timestamp: {timestamp}")
    dist.init_process_group(backend="nccl", init_method=f"file://{Share_DIR}/share_test_{timestamp}",
                            world_size=ompi_world_size, rank=ompi_world_rank)
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        init_method=f"file://{Share_DIR}/rpc_share_test_{timestamp}",
        rpc_timeout=30,
        device_maps=dmap
    )

    rpc.init_rpc(rpc_work_name(ompi_world_rank), rank=ompi_world_rank, world_size=ompi_world_size,
                 rpc_backend_options=options)
    if dist.is_initialized() and rpc.is_available():
        print(f"Hello from {ompi_world_rank}")
    tensor1 = torch.rand(32).to(device)
    meta1 = TensorDataMeta(tensor1,"layer1","factor","a")
    tensor2 = torch.rand(32).to(device)
    meta2 = TensorDataMeta(tensor2,"layer2","factor","g")
    send_tensor(meta1,tensor1,dist.get_rank(), (dist.get_rank()+1)% dist.get_world_size())
    send_tensor(meta2,tensor2,dist.get_rank(), (dist.get_rank()+2)% dist.get_world_size())
    with recv_buffer_lock:
        print(f"recv buffer {' '.join(map(str,recv_buffer))}  at {dist.get_rank()}")
    time.sleep(1)
    '''
    for thread in threading.enumerate():
        if thread == threading.main_thread():
            continue
        thread.join()
    '''
    dist.barrier()
    rpc.shutdown(timeout = 5)
    dist.destroy_process_group()
