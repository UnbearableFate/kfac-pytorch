import torch
import torch.distributed.rpc as rpc
import os

ompi_world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', -1))
ompi_world_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', -1))

if ompi_world_size == -1 or ompi_world_rank == -1:
    raise RuntimeError("This script is intended to be launched with mpirun")

DATA_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/data"
Share_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/share_files"

def get_cuda_tensor(tensor,from_rank):
    print(f"get tensor{tensor} from {from_rank} ,deivce :{tensor.device}")

def rpc_work_name(rank:int) -> str:
    return f"rpc_{rank}"

def full_connnection_device_map(world_size,rank):
    device_map = {}
    for i in range(world_size):
        if i == rank:
            continue
        device_map[rpc_work_name(i)] = {0 : 0}
    return device_map

if __name__ == '__main__':
    dmap = full_connnection_device_map(ompi_world_size,ompi_world_rank)
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        init_method=f"file://{Share_DIR}/rpc_share_test_00",
        rpc_timeout=30,
        device_maps=dmap
    )

    rpc.init_rpc(rpc_work_name(ompi_world_rank), rank=ompi_world_rank, world_size=ompi_world_size, rpc_backend_options=options)
    print(f"Hello from {ompi_world_rank}")

    tensor = torch.rand(2,2).to("cuda:0")
    rpc.rpc_sync(rpc_work_name((ompi_world_rank+1)%ompi_world_size),
                 get_cuda_tensor,
                 args=(tensor,ompi_world_rank))
    rpc.shutdown()
