
from kfac.adsgds.layer_type_common import LayerwiseModelStore, rpc_work_name
from kfac.rpc_util.common_util import model2flatten_tensor, flatten_tensor2model ,compute_l2_norm
from my_module.mobile_net import CustomMobileNetV3Small
from torch.nn import init
import torch.nn as nn
import torch

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight)  # 使用 He 初始化
        if m.bias is not None:
            init.zeros_(m.bias)

if __name__ == '__main__':
    model = CustomMobileNetV3Small()
    model.apply(initialize_weights)
    for name, param in model.named_parameters():
        print(name, param.size())
    flat = model2flatten_tensor(model)
    layered_model_store = LayerwiseModelStore(model)
    tensor_list = []
    for name, param in layered_model_store.layer_parameters.items():
        tensor_list.append(param.view(-1))
    flat2 = torch.cat(tensor_list,dim=0)

    print(f"flat shape: {flat.size()}, flat2 shape: {flat2.size()}")
    print(f"norm diff: {compute_l2_norm(flat - flat2)}")