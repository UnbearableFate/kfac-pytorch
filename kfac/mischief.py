import logging
import random

import torch.distributed as dist
import torch
from kfac.enums import AllreduceMethod

class NodeStatus:
    def __init__(self,rank,discon_start_time = -1):
        self.is_connected = True
        self.resume_countdown = -1
        self.disconnect_start_time = discon_start_time
        self.connect_start_time = 0
        self.disconnect_time_count = 0
        self.rank = rank

    def __repr__(self):
        return (f"rank:{self.rank} in T{ITER}  "
                f"is_connected: {self.is_connected} "
                f"countdown {self.resume_countdown} "
                f"start_time {self.disconnect_start_time} "
                f"sick {self.disconnect_time_count}/{ITER} {self.disconnect_time_count/ITER}\n")

    def disconnet(self):
        global POSSIBLE_DISCONNECTED_NODE, SICK_NODES_NUM
        self.is_connected = False
        self.resume_countdown = random.randint(1,MAX_DISCONNECT_ITER)
        POSSIBLE_DISCONNECTED_NODE[self.rank].is_connected = False
        SICK_NODES_NUM += 1

    def connect_resume(self):
        global POSSIBLE_DISCONNECTED_NODE, SICK_NODES_NUM
        self.connect_start_time = ITER
        self.is_connected = True
        self.resume_countdown = -1
        self.disconnect_start_time = -1
        POSSIBLE_DISCONNECTED_NODE[self.rank].is_connected = True
        SICK_NODES_NUM -= 1


    def update_status_in_iter(self):
        if ITER == self.disconnect_start_time:
            self.disconnet()
            return

        if not self.is_connected:
            self.resume_countdown -= 1
            self.disconnect_time_count += 1
        else:
            if self.disconnect_start_time <=0 and self.disconnect_time_count / ITER < DISCONNECT_RATIO :
                self.disconnect_start_time = ITER+1

DDP_TRIGGER = False
FACTOR_COMM_TRIGGER = False
INVERSE_COMM_TRIGGER = False
MAX_DISCONNECTED_NODE_NUM = 4
MAX_DISCONNECT_ITER = 3
DISCONNECT_RATIO = 0.2

POSSIBLE_DISCONNECTED_NODE : dict[int, NodeStatus] = {}
ITER = 0
SICK_NODES_NUM = 0

WORLD_SIZE = 8

sick_weight_magnification_ratio = 1
health_weight_magnification_ratio = 1

def mischief_init(world_size, max_disconnected_node_num=3,
                  max_disconnect_iter=3, disconnect_ratio=0.2,possible_disconnect_node=[],
                  ddp_trigger=False, factor_comm_trigger=False, inverse_comm_trigger=False,seed = 12):
    global WORLD_SIZE, MAX_DISCONNECTED_NODE_NUM, MAX_DISCONNECT_ITER, DISCONNECT_RATIO
    global DDP_TRIGGER,FACTOR_COMM_TRIGGER,INVERSE_COMM_TRIGGER
    global sick_weight_magnification_ratio, health_weight_magnification_ratio
    global POSSIBLE_DISCONNECTED_NODE, ITER, SICK_NODES_NUM

    random.seed(seed)
    WORLD_SIZE = world_size
    MAX_DISCONNECTED_NODE_NUM = max(max_disconnected_node_num,world_size-1)
    MAX_DISCONNECT_ITER = max_disconnect_iter
    DISCONNECT_RATIO = disconnect_ratio
    if possible_disconnect_node:
        if max(possible_disconnect_node) >= world_size:
            raise ValueError("possible disconnect node out of world size")
        contruct_node_status(possible_disconnect_node)
    else:
        contruct_node_status([i for i in range(world_size)])

    DDP_TRIGGER = ddp_trigger
    FACTOR_COMM_TRIGGER = factor_comm_trigger
    INVERSE_COMM_TRIGGER = inverse_comm_trigger

    if len(POSSIBLE_DISCONNECTED_NODE) != world_size and len(POSSIBLE_DISCONNECTED_NODE) != 0:
        health_weight_magnification_ratio =  WORLD_SIZE / (WORLD_SIZE - DISCONNECT_RATIO * len(POSSIBLE_DISCONNECTED_NODE)) 
        sick_weight_magnification_ratio = (1- DISCONNECT_RATIO) * health_weight_magnification_ratio
    
    ITER = 0
    SICK_NODES_NUM = 0

def open_all_trigger():
    global DDP_TRIGGER,FACTOR_COMM_TRIGGER,INVERSE_COMM_TRIGGER
    DDP_TRIGGER = True
    FACTOR_COMM_TRIGGER = True
    INVERSE_COMM_TRIGGER = True

def contruct_node_status(possible_disconnect_node):
    global POSSIBLE_DISCONNECTED_NODE
    for n in possible_disconnect_node:
        POSSIBLE_DISCONNECTED_NODE[n] = NodeStatus(n,random.randint(1,MAX_DISCONNECT_ITER))

def update_iter(iter_para=None):
    global ITER

    if iter_para:
        ITER = iter_para
    else:
        ITER += 1

    if not (DDP_TRIGGER or FACTOR_COMM_TRIGGER or INVERSE_COMM_TRIGGER):
        return

    for rank, node in POSSIBLE_DISCONNECTED_NODE.items():
        if node.resume_countdown == 0:
            node.connect_resume()
        if node.disconnect_start_time == ITER: # 如果满了推迟到下一次
            if SICK_NODES_NUM+1 > MAX_DISCONNECTED_NODE_NUM:
                node.disconnect_start_time += 1
        node.update_status_in_iter()

def get_connnecting_world_size():
    return WORLD_SIZE - SICK_NODES_NUM

def is_sick_at(rank):
    if rank in POSSIBLE_DISCONNECTED_NODE:
        return not POSSIBLE_DISCONNECTED_NODE[rank].is_connected
    else:
        return False
def print_node_status():
    for rank, node in POSSIBLE_DISCONNECTED_NODE.items():
        print(node)

def get_health_nodes():
    health_nodes = [i for i in range(WORLD_SIZE)]
    for rank, node in POSSIBLE_DISCONNECTED_NODE.items():
        if not node.is_connected:
            health_nodes.remove(rank)
    return health_nodes

def average_health_nodes_param(model):
    health_nodes = get_health_nodes()
    ratio = 0
    if dist.get_rank() in POSSIBLE_DISCONNECTED_NODE:
        ratio = sick_weight_magnification_ratio / len(health_nodes)
    else:
        ratio = health_weight_magnification_ratio / len(health_nodes)
    for param in model.parameters():
        if dist.get_rank() in health_nodes:
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data.mul_(ratio)
        else:
            dist.all_reduce(torch.zeros_like(param.data), op=dist.ReduceOp.SUM)

def average_health_nodes_param_without_just_start(model,rank):  # temp no use
    health_nodes = get_health_nodes()
    ratio = 0
    allreduce_num = len(health_nodes)
    for node in POSSIBLE_DISCONNECTED_NODE.values():
        if node.connect_start_time == ITER:
            allreduce_num -= 1
    if rank in POSSIBLE_DISCONNECTED_NODE:
        ratio = sick_weight_magnification_ratio / allreduce_num
    else:
        ratio = health_weight_magnification_ratio / allreduce_num
    for param in model.parameters():
        if dist.get_rank() in health_nodes:
            if POSSIBLE_DISCONNECTED_NODE[rank].connect_start_time == ITER:
                param.data.zero_()
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data.mul_(ratio)
        else:
            dist.all_reduce(torch.zeros_like(param.data), op=dist.ReduceOp.SUM)


log_once = dict()

def easy_log(words:str, ranks:list):
    if dist.get_rank() in ranks :
        print(f"{words} in rank {dist.get_rank()} in iter {iter}")

def loglog(logger :logging.Logger,ranks, words, lv = logging.INFO):
    if dist.get_rank() in ranks:
        logger.log(level=lv, msg=f"rank {dist.get_rank()} : {words}")

def easy_log_once(words, rank=0):
    global log_once
    if dist.get_rank() == rank and words not in log_once:
        print(f"{words} in rank {rank}")
        log_once[words] = 1

def all_reduce_with_disconnected(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    if is_sick_at(dist.get_rank()):
        easy_log_once("ddp miss", rank=dist.get_rank())
        dist.all_reduce(tensor=torch.zeros_like(bucket.buffer()), async_op=True)
        fut = torch.futures.Future()
        fut.set_result(bucket.buffer())
        return fut
    else:
        return (
            dist.all_reduce(tensor= (bucket.buffer() / (get_connnecting_world_size())), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )

def add_hook_to_model(model):
    if not DDP_TRIGGER:
        return
    model.register_comm_hook(state=None, hook=all_reduce_with_disconnected)


def reduce_a_factor_with_sick(self, group: dist.ProcessGroup) -> bool:
    """Initiate reduction of A and store future to result.

    Note:
        all ranks should enter this function.

    Args:
        group (ProcessGroup): process group to use for the reduce
            operation. All ranks in the group should enter this function.
            Defaults to None, the default process group.
    """
    if not (FACTOR_COMM_TRIGGER and is_sick_at(dist.get_rank())):
        return False
    #easy_log_once("sick a factor comm", rank=dist.get_rank())
    if self.a_factor is None:
        raise RuntimeError('a_factor is None, cannot reduce')
    if self.allreduce_method == AllreduceMethod.ALLREDUCE:
        allreduce = self.tdc.allreduce
    elif self.allreduce_method == AllreduceMethod.ALLREDUCE_BUCKETED:
        allreduce = self.tdc.allreduce_bucketed
    else:
        raise AssertionError(
            f'Unknown allreduce_method={self.allreduce_method}',
        )
    ### disconnection in reduce_a_factor
    allreduce(  # type: ignore
        torch.zeros_like(self.a_factor),
        average=True,
        symmetric=self.symmetric_factors and self.symmetry_aware,
        group=group,
    )

    return True


def reduce_g_factor_with_sick(self, group: dist.ProcessGroup) -> bool:
    """Initiate reduction of G and store future to result.

    Note:
        all ranks should enter this function.

    Args:
        group (ProcessGroup): process group to use for the reduce
            operation. All ranks in the group should enter this function.
            Defaults to None, the default process group.
    """
    if not (FACTOR_COMM_TRIGGER and is_sick_at(dist.get_rank())):
        return False
    easy_log_once("sick g factor comm", rank=dist.get_rank())
    if self.g_factor is None:
        raise RuntimeError('g_factor is None, cannot reduce')
    if self.allreduce_method == AllreduceMethod.ALLREDUCE:
        allreduce = self.tdc.allreduce
    elif self.allreduce_method == AllreduceMethod.ALLREDUCE_BUCKETED:
        allreduce = self.tdc.allreduce_bucketed
    else:
        raise AssertionError(
            f'Unknown allreduce_method={self.allreduce_method}',
        )
    allreduce(  # type: ignore
        torch.zeros_like(self.g_factor),
        average=True,
        symmetric=self.symmetric_factors and self.symmetry_aware,
        group=group,
    )

    return True


def broadcast_with_sick(
    NonSquareTensorError, get_triu, fill_triu, get_world_size,
    tensor: torch.Tensor,
    *,
    src: int,
    group: dist.ProcessGroup,
    symmetric: bool = False
):
    """Broadcast tensor from src to all other workers asynchronously.

    Args:
        tensor (torch.Tensor): tensor for broadcast.
        src (int): rank of worker with src tensor.
        group (torch.distributed.ProcessGroup): optional process group
            to perform communication within.
        symmetric (bool): communicate symmetric tensor using upper
            triangle.

    Returns:
        Future to tensor. Tensor can be retrieved with `future.wait()`.
        The returned tensor buffer may be different from the input buffer
        depending on the bucketing configuration.

        If group size is 1, no communication is performed and the tensor
        is returned.

    Raises:
        NonSquareTensorError:
            if symmetric is True and tensor is not a 2D square tensor.
    """
    if (not INVERSE_COMM_TRIGGER) and (not is_sick_at(dist.get_rank() and not is_sick_at(src))):
        return None

    if is_sick_at(src):
        #easy_log_once(f"can not get boradcast from {src}",dist.get_rank())
        return tensor

    clone_tensor = None
    if is_sick_at(dist.get_rank()):
        #easy_log_once(f"broadcast without {dist.get_rank()} from {src}",dist.get_rank())
        clone_tensor = torch.clone(tensor)

    if get_world_size(group) == 1:
        return tensor
    shape = tensor.size()
    if symmetric:
        if len(shape) != 2 or shape[0] != shape[1]:
            raise NonSquareTensorError(
                'Symmetric communication can only be done with a 2D '
                f'square tensor. Got tensor with shape {shape}.',
            )
        tensor = get_triu(tensor)
    tensor = tensor.contiguous()
    future = dist.broadcast(
        tensor,
        src=src,
        group=group,
        async_op=True,
    ).get_future()
    if symmetric:
        future = future.then(  # pragma: no cover
            lambda fut: fill_triu(shape, fut.value()[0]),
        )
    else:
        future = future.then(  # pragma: no cover
            lambda fut: fut.value()[0],
        )
    ### disconnection in inverse boardcast
    if clone_tensor is not None:
        future.wait()
        return clone_tensor
    return future