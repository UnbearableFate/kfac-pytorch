import torch
import torch.distributed as dist
from enum import Enum
from typing import Dict, List, Tuple

import torch

import torch

def pack_tensor(tensor: torch.Tensor, info: list[float]) -> torch.Tensor:
    """
    将原始张量与一个 Python float list 拼接在一起，生成一个 1D 的打包后张量。

    参数：
        tensor: 原始数据张量，可以是任意形状 (dims)。
        info:   Python 的 float 列表。

    返回值：
        packed: 拼接后的 1D 张量，其长度 = (原数据元素总数 + len(info))。
    """
    # 先将原始张量展平为 1D
    flat_tensor = tensor.view(-1)

    # 将 info 转为 1D 的 PyTorch 张量
    info_tensor = torch.tensor(info, dtype=torch.float32).view(-1)

    # 拼接得到打包后的结果
    packed = torch.cat([flat_tensor, info_tensor], dim=0)
    return packed


def unpack_tensor(packed: torch.Tensor, original_shape: torch.Size, N: int) -> Tuple[torch.Tensor, List[float]]:
    """
    从打包后的 1D 张量中拆分出原始张量和附加信息( Python float list)。

    参数：
        packed:         打包后的 1D 张量，长度 = (原数据元素总数 + N)。
        original_shape: 原始张量的形状（torch.Size）。
        N:              附加信息向量(列表)的长度。

    返回值：
        original_tensor: 复原后的原始张量，形状为 original_shape。
        info_list:       拆分得到的 float 列表。
    """
    total_elements = original_shape.numel()  # 原张量的元素总数

    # 前面部分是原数据，后面 N 个元素是 info
    data_part = packed[:-N]
    info_part = packed[-N:]

    # 将 data_part reshape 回 original_shape
    original_tensor = data_part.view(original_shape)

    # 将 info_part 转为 Python float list
    info_list = info_part.cpu().tolist()

    return original_tensor, info_list


class CommState:
    def __init__(self, work: dist.Work ,send_recv:int, target_rank:int, tensor_tag: str, tensor: torch.Tensor , layer_name = None):
        self.work = work
        self.send_recv = send_recv
        self.target_rank = target_rank
        self.tensor_tag = tensor_tag # for model weight transfer, tensor_tag is "model" , for kfac, tensor_tag is "layer name + # + A, G..."
        self.tensor = tensor


class AsyncCommunicator:
    def __init__(self,type = "model"):
        if type == "model":
            self.send_comm_states: Dict[str, List] = {}
            self.recv_comm_states: Dict[str, List] = {}
       

    def isend(self, tensor: torch.Tensor, tensor_type: TensorType, target_rank: int , term = 0):
        work = dist.isend(tensor=tensor, dst= target_rank)
        self.send_comm_states[tensor_type].append(CommState(work, 0, target_rank, tensor_type, tensor))
    
    def irecv(self, tensor: torch.Tensor, tensor_type: TensorType, target_rank: int):
        work = dist.irecv(tensor=tensor, src= target_rank)
        self.recv_comm_states[tensor_type].append(CommState(work, 1, target_rank, tensor_type, tensor))
    
    def collect_data(self, tensor_type: TensorType):
        results = []
        for comm_state in self.recv_comm_states[tensor_type]:
            if comm_state.work.is_completed():
                results.append(comm_state.tensor)
                self.recv_comm_states[tensor_type].remove(comm_state)