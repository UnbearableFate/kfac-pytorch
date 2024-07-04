import threading
from functools import reduce
from functools import partial
import torch.distributed.rpc as rpc

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from kfac.rpc_distributed import KFacRPCCommunicator
    from kfac.assignment import KAISAAssignment
def rpc_work_name(rank:int) -> str:
    return f"rpc_{rank}"
class RPCTaskManager:
    slow_tolerance_value = 50
    max_election_period = 10
    def __init__(self,rpc_communicator: 'KFacRPCCommunicator' , assignment : 'KAISAAssignment'):
        self.rpc_communicator: 'KFacRPCCommunicator' = rpc_communicator
        self.rank = rpc_communicator.rank
        self.world_size = rpc_communicator.get_health_world_size
        self.assignment :'KAISAAssignment' = assignment

        self.leader_rank = 0
        self.currentTerm = 0
        self.votedFor = None
        self.voted_for_me_set = set()
        self.election_period = 0
        self.assignment_generation = 0

        self.reassign_task_callback = None
        self.reassign_task_reserve_health_nodes = set()
        self.resurrection_nodes = set()

        self.reassign_lock = threading.Lock()

        if self.rank == 0:
            self.identity = 0 # leader
        else:
            self.identity = 1 # 1:follower 2:candidate

        global rpc_task_manager
        rpc_task_manager = self

    def check_and_reassign(self): # call by leader
        if self.leader_rank == self.rank:
            median_iter = self.rpc_communicator.median_iter_in_health_nodes()
            min_iter = self.rpc_communicator.min_iter_in_health_nodes()
            if median_iter - min_iter > RPCTaskManager.slow_tolerance_value:
                health_nodes, sick_nodes = self.pick_health_and_sick_nodes(min_iter)
                self.rpc_communicator.print_rpc_state(f"find slow nodes {sick_nodes} and health nodes {health_nodes}")
                if self.rank in sick_nodes:
                    self.rpc_communicator.print_rpc_state(f"leader itself is slow ,wait for next election")
                    return # leader itself is slow ,wait for next election
                with self.reassign_lock:
                    self.reassign_task_reserve_health_nodes = set(health_nodes)
                    if self.reassign_task_callback is None:
                        self.reassign_task_callback = partial(self.reassign_task)
                self.rpc_communicator.print_rpc_state(
                    f"update reassign_task_reserve_health_nodes {self.reassign_task_reserve_health_nodes}")

    def pick_health_and_sick_nodes(self, min_iter): # call by leader
        health_nodes = []
        sick_nodes = []
        for state in self.rpc_communicator.get_health_node_state_list():
            rank = state.rank
            if self.rpc_communicator.node_states[rank].iter - min_iter <= 2:
                sick_nodes.append(rank)
            else:
                health_nodes.append(rank)
        return health_nodes, sick_nodes

    def reassign_task(self): # call by leader
        self.reassign_lock.acquire()
        if len(self.reassign_task_reserve_health_nodes) == 0:
            self.reassign_task_reserve_health_nodes = set(self.rpc_communicator.get_health_nodes_rank_list())
        workgroup = list(self.reassign_task_reserve_health_nodes | self.resurrection_nodes)
        new_assignment = self.assignment.greedy_assignment(self.assignment.work, [workgroup],
                                                           self.world_size(), True)
        self.rpc_communicator.update_node_state_and_inverse_assignment(workgroup, new_assignment, self.assignment_generation+1)

        send_task = dict()
        if len(self.resurrection_nodes) > 0:
            send_task = self.rpc_communicator.arrange_to_send_the_latest_model()
            self.rpc_communicator.print_rpc_state(
                f"send new model to resurrection nodes {self.resurrection_nodes} with send_task {send_task}")

        for rank in self.rpc_communicator.get_health_nodes_rank_list():
            if rank == self.rank:
                if rank in send_task:
                    self.rpc_communicator.send_model_param_callback = partial(
                        self.rpc_communicator.send_new_model_to_resurrection_node, send_task[rank],
                        list(self.resurrection_nodes))
                continue
            if rank in send_task:
                self.rpc_communicator.print_rpc_state(
                    f"reassign task {new_assignment} to {rank} with send_task {send_task[rank]}")
                try:
                    rpc.rpc_async(
                        to=rpc_work_name(rank),
                        func=recv_reassign_task,
                        args=(workgroup, new_assignment, self.assignment_generation, self.rank, self.currentTerm, send_task[rank], list(self.resurrection_nodes))
                    )
                except Exception as e:
                    print(f"reassign task failed {e} from {self.rank} to {rank}")
            else:
                self.rpc_communicator.print_rpc_state(
                    f"reassign task {new_assignment} to {rank}")
                try:
                    rpc.rpc_async(
                        to=rpc_work_name(rank),
                        func=recv_reassign_task,
                        args=(workgroup ,new_assignment , self.assignment_generation ,self.rank ,self.currentTerm , None, None)
                    )
                except Exception as e:
                    print(f"reassign task failed {e} from {self.rank} to {rank}")
        for rank in [state.rank for state in self.rpc_communicator.get_sick_node_list()]:
            self.rpc_communicator.print_rpc_state(
                f"reassign task {new_assignment} to sick {rank}")
            try:
                rpc.rpc_async(
                    to=rpc_work_name(rank),
                    func=recv_reassign_task,
                    args=(workgroup,new_assignment , self.assignment_generation ,self.rank ,self.currentTerm, None, None)
                )
            except Exception as e:
                print(f"reassign task failed {e} from {self.rank} to {rank}")
        self.reassign_task_reserve_health_nodes.clear()
        self.resurrection_nodes.clear()
        self.reassign_task_callback = None
        self.reassign_lock.release()

    def update_follwer_state(self,from_rank,from_term):
        self.currentTerm = from_term
        self.election_period = -1
        self.identity = 1
        self.votedFor = None
        self.voted_for_me_set.clear()
        self.leader_rank = from_rank

    def electing_new_leader(self): # call in loop
        #leader_iter = self.rpc_communicator.health_node_states[self.leader_rank].iter
        #forward_than_leader = sum(state.iter > leader_iter for state in self.rpc_communicator.health_node_states.values())

        if ((self.rpc_communicator.current_t() - self.rpc_communicator.node_states[self.leader_rank].iter > RPCTaskManager.slow_tolerance_value
             and self.identity == 1
             and self.votedFor is None
             and self.election_period < 0 ) or (
            self.identity == 2 and
            self.election_period > RPCTaskManager.max_election_period)):
            print("start election at rank ", self.rank)
            self.identity = 2
            self.currentTerm += 1
            self.votedFor = self.rank
            self.voted_for_me_set.clear()
            self.voted_for_me_set.add(self.rank)
            self.election_period = 0
            for state in self.rpc_communicator.get_health_node_state_list():
                rank = state.rank
                if rank == self.leader_rank:
                    continue
                try:
                    if rank != self.rank:
                        rpc.rpc_async(
                            to=rpc_work_name(rank),
                            func=request_vote,
                            args=(self.currentTerm, self.rank)
                        )
                except Exception as e:
                    print(f"request vote failed {e} from {self.rank} to {rank}")
        elif self.identity == 2 and self.election_period < RPCTaskManager.max_election_period:
            self.election_period += 1

    def resurrection_declaration(self):  # call by sick node
        for rank in self.rpc_communicator.node_states.keys():
            if rank == self.rank:
                continue
            try:
                rpc.rpc_async(
                    to=rpc_work_name(rank),
                    func=accept_regression_request,
                    args=(self.rank, self.currentTerm)
                )
            except Exception as e:
                print(f"resurrection declaration failed {e} from {self.rank} to {rank}")

rpc_task_manager:RPCTaskManager

def request_vote(candidateTerm, candidateId): # recv by follower
    global rpc_task_manager
    if candidateTerm < rpc_task_manager.currentTerm:
        return
    if rpc_task_manager.votedFor is None or rpc_task_manager.votedFor == candidateId:
        rpc_task_manager.votedFor = candidateId
        rpc_task_manager.vote_timeout = 0
        rpc_task_manager.identity = 1
        try:
            rpc.rpc_async(
                to=rpc_work_name(candidateId),
                func=grant_vote,
                args=(rpc_task_manager.currentTerm, rpc_task_manager.rank, candidateTerm)
            )
        except :
            print(f"grant vote failed from {rpc_task_manager.rank} to {candidateId}")

def grant_vote(voterTerm, voterId, candidateTerm): # recv by candidate
    global rpc_task_manager
    if candidateTerm == rpc_task_manager.currentTerm and rpc_task_manager.identity == 2:
        rpc_task_manager.voted_for_me_set.add(voterId)
    if len(rpc_task_manager.voted_for_me_set) > rpc_task_manager.world_size() // 2:
            rpc_task_manager.identity = 0
            rpc_task_manager.leader_rank = rpc_task_manager.rank
            rpc_task_manager.election_period = -1
            rpc_task_manager.check_and_reassign()

def recv_reassign_task(new_health_node_list, new_assignment, assignment_generation, from_rank, from_term , send_need_layer_names = None, send_to = None): # recv by health follower
    global rpc_task_manager
    if from_term < rpc_task_manager.currentTerm:
        return
    rpc_task_manager.update_follwer_state(from_rank, from_term)
    rpc_task_manager.rpc_communicator.update_node_state_list(new_health_node_list)
    rpc_task_manager.rpc_communicator.print_rpc_state(f"get new assignment{assignment_generation}: {new_assignment} from leader {from_rank} update health_nodes {new_health_node_list} ")
    rpc_task_manager.rpc_communicator.update_assignment_callback = partial(rpc_task_manager.rpc_communicator.update_inverse_workers, new_assignment, assignment_generation)
    rpc_task_manager.rpc_communicator.update_assignment_flag = True
    if send_need_layer_names is not None and send_to is not None:
        rpc_task_manager.rpc_communicator.send_model_param_callback = partial(rpc_task_manager.rpc_communicator.send_new_model_to_resurrection_node,send_need_layer_names,send_to)

def accept_regression_request(from_rank, from_term):
    if rpc_task_manager.rank != rpc_task_manager.leader_rank:
        return
    #if rpc_task_manager.rpc_communicator.min_iter_in_health_nodes() - from_term < RPCTaskManager.slow_tolerance_value / 3:
    with rpc_task_manager.reassign_lock:
        rpc_task_manager.rpc_communicator.print_rpc_state(f"resurrection declaration from {from_rank} accepted")
        rpc_task_manager.resurrection_nodes.add(from_rank)
        if rpc_task_manager.reassign_task_callback is None:
            rpc_task_manager.reassign_task_callback = partial(rpc_task_manager.reassign_task)
"""
@deprecated
def statistics_slow_nodes(from_rank, slow_nodes, vote_index):
    
    global rpc_task_manager
    if vote_index not in rpc_task_manager.vote_store:
        rpc_task_manager.vote_store[vote_index] = [slow_nodes]
    else:
        rpc_task_manager.vote_store[vote_index].append(slow_nodes)

    if len(rpc_task_manager.vote_store[rpc_task_manager.vote_index])  > rpc_task_manager.world_size() // 2:
        # 有一半以上的节点都投票了，开始重新分配任务
        # 求出所有被投票慢节点的并集
        widely_accepted_slow = reduce(set.intersection, map(set, rpc_task_manager.vote_store[vote_index]))
        # 重新分配任务
        for slow_node in widely_accepted_slow:
            if slow_node in rpc_task_manager.rpc_communicator.health_node_states:
                rpc_task_manager.rpc_communicator.health_node_states.pop(slow_node)

        new_health_node_list = [rank for rank in rpc_task_manager.rpc_communicator.health_node_states.keys()]
        workgroup = [[rank for rank in rpc_task_manager.rpc_communicator.health_node_states.keys()]]
        new_assignment = rpc_task_manager.assignment.greedy_assignment(rpc_task_manager.assignment.work, workgroup,
                                                                       rpc_task_manager.world_size(), True)

        rpc_task_manager.rpc_communicator.update_inverse_workers(new_assignment)

        for health_nodes_rank in rpc_task_manager.rpc_communicator.health_node_states.keys():
            rpc.rpc_async(
                to=rpc_work_name(health_nodes_rank),
                func=reassign_task,
                args=(new_health_node_list ,vote_index ,rpc_task_manager.assignment)
            )
"""
