import torch.distributed.rpc as rpc
import os

ompi_world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', -1))
ompi_world_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', -1))

if ompi_world_size == -1 or ompi_world_rank == -1:
    raise RuntimeError("This script is intended to be launched with mpirun")

DATA_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/data"
Share_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/data/share_files"

if __name__ == '__main__':
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        init_method=f"file://{Share_DIR}/rpc_share",
        rpc_timeout=30,
    )

    rpc.init_rpc(f"worker{ompi_world_rank}", rank=ompi_world_rank, world_size=ompi_world_size, rpc_backend_options=options)
    print(f"Hello from {ompi_world_rank}")
    rpc.shutdown()