from functools import reduce

import torch.distributed.rpc as rpc

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from kfac.rpc_distributed import KFacRPCCommunicator
    from kfac.assignment import KAISAAssignment
def rpc_work_name(rank:int) -> str:
    return f"rpc_{rank}"
class RPCTaskManager:
    slow_tolerance_value = 20
    max_election_period = 10
    def __init__(self,rpc_communicator: 'KFacRPCCommunicator' , assignment : 'KAISAAssignment'):
        self.rpc_communicator: 'KFacRPCCommunicator' = rpc_communicator
        self.rank = rpc_communicator.rank
        self.world_size = rpc_communicator.get_world_size
        self.assignment :'KAISAAssignment' = assignment

        self.leader_rank = 0
        self.currentTerm = 0
        self.votedFor = None
        self.voted_for_me_set = set()
        self.election_period = 0
        self.assignment_generation = 0
        if self.rank == 0:
            self.identity = 0 # leader
        else:
            self.identity = 1 # 1:follower 2:candidate

        global rpc_task_manager
        rpc_task_manager = self
    def check_and_reassign(self): # call by leader
        median_iter = self.rpc_communicator.median_iter_in_cluster()
        min_iter = self.rpc_communicator.min_iter_in_cluster()
        if median_iter - min_iter > RPCTaskManager.slow_tolerance_value:
            self.reassign_task()

    def reassign_task(self): # call by leader
        min_iter = self.rpc_communicator.min_iter_in_cluster()
        for rank in self.rpc_communicator.health_node_states.keys():
            if self.rpc_communicator.health_node_states[rank].iter - min_iter <= 1:
                if rank == self.rank:
                    return # leader itself is slow ,wait for next election
                self.rpc_communicator.health_node_states.pop(rank)
        new_health_node_list = [rank for rank in self.rpc_communicator.health_node_states.keys()]
        workgroup = [[rank for rank in self.rpc_communicator.health_node_states.keys()]]
        new_assignment = self.assignment.greedy_assignment(self.assignment.work, workgroup,
                                                           self.world_size(), True)
        self.rpc_communicator.update_inverse_workers(new_assignment)
        self.assignment_generation += 1
        #for health_nodes_rank in self.rpc_communicator.health_node_states.keys():
        for rank in range(self.rpc_communicator.origin_world_size):
            rpc.rpc_async(
                to=rpc_work_name(rank),
                func=reassign_task,
                args=(new_health_node_list ,new_assignment , self.assignment_generation ,self.rank ,self.currentTerm)
            )

    def electing_new_leader(self): # call in loop
        #leader_iter = self.rpc_communicator.health_node_states[self.leader_rank].iter
        #forward_than_leader = sum(state.iter > leader_iter for state in self.rpc_communicator.health_node_states.values())

        if ((self.rpc_communicator.current_t() - self.rpc_communicator.health_node_states[self.leader_rank].iter > RPCTaskManager.slow_tolerance_value
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
            for rank in self.rpc_communicator.health_node_states.keys():
                if rank == self.leader_rank:
                    continue
                if rank != self.rank:
                    rpc.rpc_async(
                        to=rpc_work_name(rank),
                        func=request_vote,
                        args=(self.currentTerm, self.rank)
                    )
        elif self.identity == 2 and self.election_period < RPCTaskManager.max_election_period:
            self.election_period += 1

rpc_task_manager:RPCTaskManager

def request_vote(candidateTerm, candidateId): # recv by follower
    global rpc_task_manager
    if candidateTerm < rpc_task_manager.currentTerm:
        return
    if rpc_task_manager.votedFor is None or rpc_task_manager.votedFor == candidateId:
        rpc_task_manager.votedFor = candidateId
        rpc_task_manager.vote_timeout = 0
        rpc_task_manager.identity = 1
        rpc.rpc_async(
            to=rpc_work_name(candidateId),
            func=grant_vote,
            args=(rpc_task_manager.currentTerm, rpc_task_manager.rank, candidateTerm)
        )

def grant_vote(voterTerm, voterId, candidateTerm): # recv by candidate
    global rpc_task_manager
    if candidateTerm == rpc_task_manager.currentTerm and rpc_task_manager.identity == 2:
        rpc_task_manager.voted_for_me_set.add(voterId)
    if len(rpc_task_manager.voted_for_me_set) > rpc_task_manager.world_size() // 2:
            rpc_task_manager.identity = 0
            rpc_task_manager.leader_rank = rpc_task_manager.rank
            rpc_task_manager.election_period = -1
            rpc_task_manager.check_and_reassign()

def reassign_task(new_health_node_list, new_assignment , assignment_generation, from_rank , from_term): # recv by health follower
    global rpc_task_manager
    if from_term < rpc_task_manager.currentTerm:
        return
    rpc_task_manager.rpc_communicator.update_node_state_list(new_health_node_list)
    rpc_task_manager.rpc_communicator.update_inverse_workers(new_assignment)
    rpc_task_manager.assignment_generation = assignment_generation
    rpc_task_manager.currentTerm = from_term
    rpc_task_manager.election_period = -1
    rpc_task_manager.identity = 1
    rpc_task_manager.votedFor = None
    rpc_task_manager.voted_for_me_set.clear()
    rpc_task_manager.leader_rank = from_rank

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
