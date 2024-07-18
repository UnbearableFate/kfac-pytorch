import datetime
import os

import torch
import torch.distributed as dist
from my_module.custom_resnet import ResNetForCIFAR10, MLP
from general_util.GeneralManager import GeneralManager
from my_module.model_split import ModelSplitter
import shutil

def delete_all_files_in_directory(directory_path):
    # 检查路径是否存在
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return

    # 遍历目录下的所有文件和文件夹
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # 如果是文件则删除
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        # 如果是目录，则可以选择递归删除或者跳过
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
            print(f"Deleted directory: {file_path}")

gpu = torch.device("cuda:0")
today = datetime.date.today().strftime('%m%d')
pg_share_file = "pg_share"
rpc_share_fie = "rpc_share"

DATA_DIR = ""
LOG_DIR = ""
Share_DIR = ""
if os.path.exists("/home/yu"):
    DATA_DIR = "/home/yu/data"
    LOG_DIR = "/home/yu/workspace/kfac-pytorch/runs"+today
    Share_DIR = "/home/yu/workspace/kfac-pytorch/share_files"
elif os.path.exists("/Users/unbearablefate"):
    DATA_DIR = "/Users/unbearablefate/workspace/data"
    LOG_DIR = "/Users/unbearablefate/workspace/kfac-pytorch/runs"+today
    Share_DIR = "/Users/unbearablefate/workspace/kfac-pytorch/share_files"

if DATA_DIR == "" or LOG_DIR == "" or Share_DIR == "":
    raise RuntimeError("Unknown environment.")

ompi_world_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', -1))
if ompi_world_rank == 0:
    delete_all_files_in_directory(Share_DIR)

if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    model = MLP(num_hidden_layers=3,hidden_size=64)
    model = ModelSplitter(model, 64)
    mgr = GeneralManager(data_dir=DATA_DIR, dataset_name="FashionMNIST", model=model,
                         sampler_func= None,
                         train_com_method='rpc', interval=1, is_2nd_order=True, epochs=50,device='cpu',
                         share_file_path=Share_DIR)

    mgr.train_and_test(log_dir=LOG_DIR, timestamp=timestamp, experiment_name="test01")
    mgr.close_all()
    print("Done!")