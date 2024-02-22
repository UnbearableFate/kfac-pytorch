import logging

import torch.distributed as dist
import torch

DISCONNECTED_NODE = []
DISCONNECTED_NODE_NUM = 4
DISCONNECT_TERM = 1
CONNECT_TERM = 9
TIMES = 0

term = 0

DDP_TRIGGER = True
FACTOR_COMM_TRIGGER = True
INVERSE_COMM_TRIGGER = True

log_once = dict()
def easy_log_once(words, rank=0):
    if dist.get_rank() == rank and words not in log_once:
        print(f"{words} in rank {rank}")
        log_once[words] = 1

def easy_log(words:str, ranks:list):
    if dist.get_rank() in ranks :
        print(f"{words} in rank {dist.get_rank()} in term {term}")

def loglog(logger :logging.Logger,ranks, words, lv = logging.INFO):
    if dist.get_rank() in ranks:
        logger.log(level=lv, msg=f"rank {dist.get_rank()} : {words}")

def shuffle_disconneted_node_list(world_size,epochs):
    r = int(term / (CONNECT_TERM + DISCONNECT_TERM))
    rounds = int(epochs/ (CONNECT_TERM + DISCONNECT_TERM))
    step = world_size/rounds
    node_list = []
    offset = int(world_size / DISCONNECTED_NODE_NUM)
    for i in range(DISCONNECTED_NODE_NUM):
        node = (int(r*step) + i * offset)%world_size +TIMES
        node_list.append(node)
    DISCONNECTED_NODE = node_list
    return node_list

def is_connected_in_this_term():
    t = term % (CONNECT_TERM + DISCONNECT_TERM)
    if t < CONNECT_TERM :
        return True
    return False

def get_connnecting_world_size():
    return dist.get_world_size() - DISCONNECTED_NODE_NUM

def all_reduce_with_disconnected(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    if dist.get_rank() in DISCONNECTED_NODE and not is_connected_in_this_term():
        easy_log_once("ddp miss", rank=1)
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