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
        return (f"rank:{self.rank}\n"
                f"is_connected: {self.is_connected} "
                f"countdown {self.resume_countdown} "
                f"start_time {self.disconnect_start_time}\n\n")

    def disconnet(self):
        self.is_connected = False
        self.resume_countdown = random.randint(1,Mischief.MAX_DISCONNECT_ITER)
        Mischief.DISCONNECTING_NODES.add(self.rank)

    def connect_resume(self):
        self.is_connected = True
        self.resume_countdown = 0
        Mischief.DISCONNECTING_NODES.remove(self.rank)

    def update_status_in_iter(self):
        if Mischief.ITER == self.disconnect_start_time:
            self.disconnet()
            return

        if not self.is_connected:
            self.resume_countdown -= 1
            self.disconnect_time_count += 1
            if self.resume_countdown <= 0:
                self.connect_resume()
        else:
            if self.disconnect_time_count / Mischief.ITER < Mischief.DISCONNECT_RATIO :
                self.disconnect_start_time = Mischief.ITER+1

class Mischief:
    DDP_TRIGGER = True
    FACTOR_COMM_TRIGGER = True
    INVERSE_COMM_TRIGGER = True

    POSSIBLE_DISCONNECTED_NODE = []
    MAX_DISCONNECTED_NODE_NUM = 4
    MAX_DISCONNECT_ITER = 3
    ITER = 0
    DISCONNECTING_NODES = set()

    WORLD_SIZE = 8
    DISCONNECT_RATIO = 0.2

    def __init__(self,ddp_trigger = True, factor_comm_trigger = True,
             inverse_comm_trigger = True, possible_disconnect_node=[],
             max_disconnect_iter = 3,max_disconnect_node_num = 2,
                 world_size = 8):
        Mischief.DDP_TRIGGER = ddp_trigger
        Mischief.FACTOR_COMM_TRIGGER = factor_comm_trigger
        Mischief.INVERSE_COMM_TRIGGER = inverse_comm_trigger
        Mischief.POSSIBLE_DISCONNECTED_NODE = possible_disconnect_node
        Mischief.MAX_DISCONNECT_ITER = max_disconnect_iter
        Mischief.MAX_DISCONNECTED_NODE_NUM = max_disconnect_node_num
        Mischief.ITER = 0
        Mischief.WORLD_SIZE = world_size

        self.nodes = dict()
        random.seed(12)
        for n in Mischief.POSSIBLE_DISCONNECTED_NODE:
            self.nodes[n] = NodeStatus(n,random.randint(1,Mischief.MAX_DISCONNECT_ITER))

    def contruct_node_status(self,possible_disconnect_node):
        Mischief.POSSIBLE_DISCONNECTED_NODE = possible_disconnect_node
        random.seed(12)
        for n in Mischief.POSSIBLE_DISCONNECTED_NODE:
            self.nodes[n] = NodeStatus(n,random.randint(1,Mischief.MAX_DISCONNECT_ITER))

    def update_iter(self, iter_para=None):
        if iter_para:
            Mischief.ITER = iter_para
        else:
            Mischief.ITER += 1
        for rank, node in self.nodes.items():
            if node.disconnect_start_time == Mischief.ITER: # 如果满了推迟到下一次
                if len(Mischief.DISCONNECTING_NODES)+1 > Mischief.MAX_DISCONNECTED_NODE_NUM:
                    node.disconnect_start_time += 1
            node.update_status_in_iter()

    def get_connnecting_world_size(self):
        return Mischief.WORLD_SIZE - len(Mischief.DISCONNECTING_NODES)

    def is_connected_in(self,rank):
        if rank not in self.nodes:
            return True
        return self.nodes[rank].is_connected

    def close_all(self):
        Mischief.DDP_TRIGGER = False
        Mischief.FACTOR_COMM_TRIGGER = False
        Mischief.INVERSE_COMM_TRIGGER = False

MischiefHelper = Mischief(possible_disconnect_node=[1,2,3])

log_once = dict()

def easy_log(words:str, ranks:list):
    if dist.get_rank() in ranks :
        print(f"{words} in rank {dist.get_rank()} in iter {iter}")

def loglog(logger :logging.Logger,ranks, words, lv = logging.INFO):
    if dist.get_rank() in ranks:
        logger.log(level=lv, msg=f"rank {dist.get_rank()} : {words}")

def easy_log_once(words, rank=0):
    if dist.get_rank() == rank and words not in log_once:
        print(f"{words} in rank {rank}")
        log_once[words] = 1

def all_reduce_with_disconnected(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    if MischiefHelper.is_connected_in(dist.get_rank()):
        easy_log_once("ddp miss", rank=1)
        dist.all_reduce(tensor=torch.zeros_like(bucket.buffer()), async_op=True)
        fut = torch.futures.Future()
        fut.set_result(bucket.buffer())
        return fut
    else:
        return (
            dist.all_reduce(tensor= (bucket.buffer() / (MischiefHelper.get_connnecting_world_size())), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )

def add_hook_to_model(model):
    if not Mischief.DDP_TRIGGER:
        return
    model.register_comm_hook(state=None, hook=all_reduce_with_disconnected)