import logging

import torch.distributed as dist
import torch

DISCONNECTED_NODE = [1,2]
DISCONNECT_TERM = 4
CONNECT_TERM = 6

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

def is_inverse_communication_jumped(iter, src_worker_no, jump_step = 3) -> bool:
    if not INVERSE_COMM_TRIGGER:
        return True
    if iter % jump_step == 0 and src_worker_no in DISCONNECTED_NODE:
        easy_log_once("broadcast block", rank=1)
        return False
    else:
        return True

def is_connected_in_this_term():
    t = term % (CONNECT_TERM + DISCONNECT_TERM)
    if t < CONNECT_TERM :
        return True
    return False

def all_reduce_with_disconnected(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    if dist.get_rank() in DISCONNECTED_NODE and not is_connected_in_this_term():
        easy_log_once("ddp miss", rank=1)
        dist.all_reduce(tensor=torch.zeros_like(bucket.buffer()), async_op=True)
        fut = torch.futures.Future()
        fut.set_result(bucket.buffer())
        return fut
    else:
        return (
            dist.all_reduce(tensor= (bucket.buffer() / (dist.get_world_size() -1)), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )

def add_hook_to_model(model):
    if not DDP_TRIGGER:
        return
    model.register_comm_hook(state=None, hook=all_reduce_with_disconnected)