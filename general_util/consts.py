import os
import datetime

ompi_world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', -1))
ompi_world_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', -1))
DATA_DIR = ""
LOG_DIR = ""
SHARE_FILES_DIR = ""
CHECK_POINT_PATH = ""
SHARED_MODEL_PATH = ""
today = datetime.date.today().strftime('%m%d')

if os.path.exists("/home/yu"):
    DATA_DIR = "/home/yu/data"
    LOG_DIR = "/home/yu/workspace/kfac-pytorch/runs/runs"+today
    SHARE_FILES_DIR = "/home/yu/workspace/kfac-pytorch/share_files"
    CHECK_POINT_PATH = "/home/yu/workspace/kfac-pytorch/checkpoints"
elif os.path.exists("/Users/unbearablefate"):
    DATA_DIR = "/Users/unbearablefate/workspace/data"
    LOG_DIR = "/Users/unbearablefate/workspace/kfac-pytorch/runs/runs"+today
    SHARE_FILES_DIR = "/Users/unbearablefate/workspace/kfac-pytorch/share_files"
    CHECK_POINT_PATH = "/Users/unbearablefate/workspace/kfac-pytorch/checkpoints"
elif os.path.exists("/work/NBB/yu_mingzhe/kfac-pytorch"):
    DATA_DIR = "/work/NBB/yu_mingzhe/data"
    LOG_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/runs/runs"+today
    SHARE_FILES_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/share_files"
    CHECK_POINT_PATH = "/work/NBB/yu_mingzhe/kfac-pytorch/checkpoints"
    SHARED_MODEL_ROOT = CHECK_POINT_PATH+"/shared_models"

delay_list_dict = [
    [0.0006, 0.0071, 0.0046, 0.0043, 0.0119, 0.0049, 0.0121, 0.0097, 0.0029, 0.0238, 0.0069, 0.0062, 0.0008, 0.0148, 0.0114, 0.0098],
    [0.0009, 0.0031, 0.0159, 0.0018, 0.015, 0.0603, 0.0068, 0.0025, 0.0033, 0.0138, 0.0321, 0.0184, 0.0112, 0.0149, 0.0013, 0.0605],
    [0.0016, 0.0311, 0.0176, 0.0288, 0.0139, 0.0399, 0.016, 0.0082, 0.1098, 0.0273, 0.0048, 0.0105, 0.0076, 0.0638, 0.0028, 0.0155],
    [0.0345, 0.0063, 0.0216, 0.1123, 0.0504, 0.0433, 0.0113, 0.0069, 0.0411, 0.0138, 0.0422, 0.0527, 0.0197, 0.0156, 0.0114, 0.0788],
    [0.004, 0.0055, 0.0506, 0.1451, 0.0862, 0.0365, 0.0095, 0.046, 0.116, 0.0207, 0.0056, 0.0395, 0.0476, 0.0403, 0.0086, 0.0179]
]
