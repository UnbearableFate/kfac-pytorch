
import datetime
import torch.distributed as dist
import general_util
from my_module.custom_resnet import ResNetForCIFAR10
from functools import partial

if __name__ == '__main__':
    timeout = datetime.timedelta(seconds=30)
    dist.init_process_group("gloo", timeout=timeout)
    if not dist.is_initialized():
        raise RuntimeError("Unable to initialize process group.")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    DATA_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/data"
    LOG_DIR = "/work/NBB/yu_mingzhe/kfac-pytorch/runs"
    
    model_fn = partial(ResNetForCIFAR10, layers=18)
    general_util.general_main(data_dir= DATA_DIR,log_dir= LOG_DIR ,dataset_name= "CIFAR10",
                              timestamp= timestamp,model_func=model_fn,disconnect_ratio=0.2)
    general_util.general_main(data_dir= DATA_DIR,log_dir= LOG_DIR ,dataset_name= "CIFAR10",
                              timestamp= timestamp, model_func=model_fn,disconnect_ratio=0.4)
    dist.destroy_process_group()