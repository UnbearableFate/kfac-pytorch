import threading
import time
from functools import reduce
from functools import partial
import random

import torch.distributed.rpc as rpc

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from kfac.rpc_distributed import KFacRPCCommunicator
    from kfac.assignment import KAISAAssignment
def rpc_work_name(rank:int) -> str:
    return f"rpc_{rank}"
class RPCTaskManager:
    slow_tolerance_value = 150
    max_election_period = 20
    def __init__(self,rpc_communicator: 'KFacRPCCommunicator' , assignment : 'KAISAAssignment' ,slow_tolerance_value = 150,max_election_period=20):
        self.rpc_communicator: 'KFacRPCCommunicator' = rpc_communicator
        self.rank = rpc_communicator.rank
        self.world_size = rpc_communicator.get_health_world_size
        self.assignment :'KAISAAssignment' = assignment

        self.leader_rank = 0
        self.currentTerm = 0
        self.votedFor = None
        self.voted_for_me_set = set()
        self.election_period = -1
        self.assignment_generation = 0

        self.reassign_task_callback = None
        self.reassign_task_reserve_health_nodes = set()
        self.resurrection_nodes = set()

        self.reassign_lock = threading.Lock()

        if self.rank == 0:
            self.identity = 0 # leader
        else:
            self.identity = 1 # 1:follower 2:candidate

        RPCTaskManager.slow_tolerance_value = slow_tolerance_value
        RPCTaskManager.max_election_period = max_election_period

        global rpc_task_manager
        rpc_task_manager = self

    def print_state(self):
        result = ""
        result += f"identity : {self.identity} "
        result += f"currentTerm : {self.currentTerm} "
        result += f"votedFor : {self.votedFor} "
        result += f"leader_rank : {self.leader_rank} "
        result += f"election_period : {self.election_period} "

        return result
    def check_and_reassign(self): # call by leader
        if self.leader_rank == self.rank and self.reassign_task_callback is None:
            median_iter = self.rpc_communicator.median_iter_in_health_nodes()
            health_nodes, sick_nodes = self.pick_health_and_sick_nodes(median_iter)
            if set(health_nodes) != set(self.rpc_communicator.get_health_nodes_rank_list()):
                self.rpc_communicator.print_rpc_state(f"find slow nodes {sick_nodes} and health nodes {health_nodes}")
                if self.rank in sick_nodes:
                    self.rpc_communicator.print_rpc_state(f"leader itself is slow ,wait for next election")
                    return # leader itself is slow ,wait for next election
                if not self.reassign_lock.acquire(timeout= 1):
                    raise Exception("reassign task lock is not released")
                self.reassign_task_reserve_health_nodes = set(health_nodes)
                if self.reassign_task_callback is None:
                    self.reassign_task_callback = partial(self.reassign_task)
                self.reassign_lock.release()
                self.rpc_communicator.print_rpc_state(
                    f"update reassign_task_reserve_health_nodes {self.reassign_task_reserve_health_nodes}")

    def pick_health_and_sick_nodes(self, median_iter): # call by leader
        health_nodes = []
        sick_nodes = []
        for rank, state in self.rpc_communicator.node_states.items():
            if state.health == True and median_iter - state.iter  < RPCTaskManager.slow_tolerance_value:
                health_nodes.append(rank)
            if state.health == False and median_iter - state.iter  < 0.8 * RPCTaskManager.slow_tolerance_value:
                health_nodes.append(rank)
        return health_nodes, sick_nodes

    def reassign_task(self): # call by leader
        if not self.reassign_lock.acquire(timeout=1):
            raise Exception("can not acquire reassign task lock in reassign_task")
        
        if len(self.reassign_task_reserve_health_nodes) == 0:
            self.reassign_task_reserve_health_nodes = set(self.rpc_communicator.get_health_nodes_rank_list())
        # reassign task in local
        new_health_nodes = list(self.reassign_task_reserve_health_nodes | self.resurrection_nodes)
        old_health_nodes = set(self.rpc_communicator.get_health_nodes_rank_list())
        self.rpc_communicator.update_node_state_list(new_health_nodes)
        new_assignment = self.assignment.greedy_assignment_efficiency(self.assignment.work, [new_health_nodes], True,self.rpc_communicator.get_computation_speed_dict())
        self.rpc_communicator.update_inverse_workers(new_assignment, self.assignment_generation+1)
        
        self.reassign_task_reserve_health_nodes.clear()
        self.resurrection_nodes.clear()
        self.reassign_task_callback = None
        
        self.reassign_lock.release()
        # reassign task in remote
        send_task = dict()
        new_resurrection_nodes = set(new_health_nodes)- old_health_nodes
        survived_nodes = set(new_health_nodes) - new_resurrection_nodes
        if len(new_resurrection_nodes) > 0:
            send_task = self.rpc_communicator.arrange_to_send_the_latest_model(list(survived_nodes))
            self.rpc_communicator.print_rpc_state(
                f"send new model to resurrection nodes {new_resurrection_nodes} with send_task {send_task}")

        for rank in self.rpc_communicator.get_health_nodes_rank_list():
            if rank == self.rank:
                if rank in send_task:
                    self.rpc_communicator.send_model_param_callback = partial(
                        self.rpc_communicator.send_new_model_to_resurrection_node, send_task[rank],
                        list(new_resurrection_nodes))
                continue
            if rank in send_task:
                self.rpc_communicator.print_rpc_state(
                    f"reassign task {new_assignment} to {rank} with send_task {send_task[rank]}")
                try:
                    rpc.rpc_async(
                        to=rpc_work_name(rank),
                        func=recv_reassign_task,
                        args=(new_health_nodes, new_assignment, self.assignment_generation, self.rank, self.currentTerm, send_task[rank], list(new_resurrection_nodes))
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
                        args=(new_health_nodes ,new_assignment , self.assignment_generation ,self.rank ,self.currentTerm , None, None)
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
                    args=(new_health_nodes,new_assignment , self.assignment_generation ,self.rank ,self.currentTerm, None, None)
                )
            except Exception as e:
                print(f"reassign task failed {e} from {self.rank} to {rank}")

    def update_follwer_state(self,from_rank,from_term):
        self.currentTerm = from_term
        self.election_period = -1
        self.identity = 1
        self.votedFor = None
        self.voted_for_me_set.clear()
        self.leader_rank = from_rank

    def start_election(self): # call by no leader
        self.identity = 2
        self.currentTerm += 1
        self.votedFor = self.rank
        self.voted_for_me_set.clear()
        self.voted_for_me_set.add(self.rank)
        self.election_period = 0
        for state in self.rpc_communicator.get_health_node_state_list():
            rank = state.rank
            if rank == self.rank: # or rank == self.leader_rank ??
                continue
            try:
                rpc.rpc_async(
                    to=rpc_work_name(rank),
                    func=request_vote,
                    args=(self.currentTerm, self.rank)
                )
            except Exception as e:
                print(f"request vote failed {e} from {self.rank} to {rank}")

    def electing_new_leader_loop(self): # call by in loop
        #leader_iter = self.rpc_communicator.health_node_states[self.leader_rank].iter
        #forward_than_leader = sum(state.iter > leader_iter for state in self.rpc_communicator.health_node_states.values())
        if (self.rpc_communicator.current_t() - self.rpc_communicator.node_states[self.leader_rank].iter > RPCTaskManager.slow_tolerance_value
             and self.identity == 1
             and self.votedFor is None
             and self.election_period < 0 ):
            self.rpc_communicator.print_rpc_state(f"start new election")
            self.start_election()
        elif self.election_period > RPCTaskManager.max_election_period:
            if self.identity == 2:
                time.sleep(random.random() * self.rank * 0.1)
                if self.rpc_communicator.current_t() - self.rpc_communicator.node_states[self.leader_rank].iter > RPCTaskManager.slow_tolerance_value :
                    self.rpc_communicator.print_rpc_state(f"start election again")
                    self.start_election()
                else:
                    self.rpc_communicator.print_rpc_state(f"give up election")
                    self.update_follwer_state(self.leader_rank,self.currentTerm)
            elif (self.identity == 1 and
                  (self.votedFor is not None and  self.rpc_communicator.node_states[self.votedFor].iter - self.rpc_communicator.node_states[self.leader_rank].iter < RPCTaskManager.slow_tolerance_value)):
                self.update_follwer_state(self.leader_rank,self.currentTerm)
        elif self.election_period >= 0 and self.votedFor is not None:
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
    if (candidateTerm < rpc_task_manager.currentTerm or
    rpc_task_manager.rpc_communicator.node_states[candidateId].iter - rpc_task_manager.rpc_communicator.node_states[rpc_task_manager.leader_rank].iter < RPCTaskManager.slow_tolerance_value /2
        ):
        return
    if rpc_task_manager.votedFor is None or rpc_task_manager.votedFor == candidateId:
        rpc_task_manager.votedFor = candidateId
        rpc_task_manager.election_period = 0
        rpc_task_manager.identity = 1
        rpc_task_manager.rpc_communicator.print_rpc_state(f"grant vote from {rpc_task_manager.rank} to {candidateId}")
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
            rpc_task_manager.votedFor = None
            rpc_task_manager.voted_for_me_set.clear()
            rpc_task_manager.rpc_communicator.print_rpc_state(f"become new leader {rpc_task_manager.rank} with term {rpc_task_manager.currentTerm}")
            rpc_task_manager.check_and_reassign()

def recv_reassign_task(new_health_node_list, new_assignment, assignment_generation, from_rank, from_term , send_need_layer_names = None, send_to = None): # recv by health follower
    global rpc_task_manager
    if from_term < rpc_task_manager.currentTerm:
        return
    if not rpc_task_manager.reassign_lock.acquire(timeout=1):
        raise Exception("can not acquire reassign task lock in recv_reassign_task")
    rpc_task_manager.update_follwer_state(from_rank, from_term)
    rpc_task_manager.rpc_communicator.update_node_state_list(new_health_node_list)
    rpc_task_manager.rpc_communicator.print_rpc_state(f"get new assignment{assignment_generation}: {new_assignment} from leader {from_rank} update health_nodes {new_health_node_list} ")
    rpc_task_manager.rpc_communicator.update_assignment_callback = partial(rpc_task_manager.rpc_communicator.update_inverse_workers, new_assignment, assignment_generation)
    if send_need_layer_names is not None and send_to is not None:
        rpc_task_manager.rpc_communicator.send_model_param_callback = partial(rpc_task_manager.rpc_communicator.send_new_model_to_resurrection_node,send_need_layer_names,send_to)
    rpc_task_manager.reassign_lock.release()

def accept_regression_request(from_rank, from_term):
    if rpc_task_manager.rank != rpc_task_manager.leader_rank:
        return
    #if rpc_task_manager.rpc_communicator.min_iter_in_health_nodes() - from_term < RPCTaskManager.slow_tolerance_value / 3:
    if (from_rank in rpc_task_manager.resurrection_nodes
            or from_rank in rpc_task_manager.reassign_task_reserve_health_nodes
            or rpc_task_manager.rpc_communicator.node_states[from_rank].health == True):
        rpc_task_manager.rpc_communicator.print_rpc_state(
            f"resurrection declaration from {from_rank} accepted ,but already in health nodes")
        return
    rpc_task_manager.rpc_communicator.print_rpc_state(f"resurrection declaration from {from_rank} accepted")
    if rpc_task_manager.reassign_lock.acquire(timeout=1):
        raise Exception("can not acquire reassign task lock in accept_regression_request")
    rpc_task_manager.resurrection_nodes.add(from_rank)
    if rpc_task_manager.reassign_task_callback is None:
        rpc_task_manager.reassign_task_callback = partial(rpc_task_manager.reassign_task)
    rpc_task_manager.reassign_lock.release()