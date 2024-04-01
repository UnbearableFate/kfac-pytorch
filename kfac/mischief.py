import logging
import random

import torch.distributed as dist
import torch

class NodeStatus:
    def __init__(self,rank,discon_start_time = -1):
        self.is_connected = True
        self.resume_countdown = -1
        self.disconnect_start_time = discon_start_time
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

POSSIBLE_DISCONNECTED_NODE = dict()
ITER = 0
SICK_NODES_NUM = 0

WORLD_SIZE = 8

def mischief_init(world_size, max_disconnected_node_num=3,
                  max_disconnect_iter=3, disconnect_ratio=0.2,possible_disconnect_node=[],
                  ddp_trigger=False, factor_comm_trigger=False, inverse_comm_trigger=False,seed = 12):
    global WORLD_SIZE, MAX_DISCONNECTED_NODE_NUM, MAX_DISCONNECT_ITER, DISCONNECT_RATIO
    global DDP_TRIGGER,FACTOR_COMM_TRIGGER,INVERSE_COMM_TRIGGER

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
        contruct_node_status([i for i in range(world_size-1)])

    DDP_TRIGGER = ddp_trigger
    FACTOR_COMM_TRIGGER = factor_comm_trigger
    INVERSE_COMM_TRIGGER = inverse_comm_trigger

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