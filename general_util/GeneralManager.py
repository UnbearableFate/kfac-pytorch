import math
import time
import torch
from tqdm import tqdm
import kfac.mischief as mischief
from general_util.data_preparation import DataPreparer
import torch.distributed as dist
import logging
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import kfac
from general_util.tensor_funsion import fuse_tensors, fuse_model_paramenters, unfuse_tensors_to_model
import kfac.rpc_distributed as rpc_distributed
class GeneralManager:
    def __init__(self,data_dir,dataset_name,model,sampler_func = None,is_ddp=False,interval=10,is_2nd_order =True,epochs=100,device=torch.device("cuda:0")):
        self.writer = None
        batch_size=64
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if rank == 0:
            logging.basicConfig(level=logging.NOTSET)

        self.data_manager = DataPreparer(data_path_root=data_dir, dataset_name=dataset_name, world_size=world_size, rank=rank,
                                    sampler=sampler_func, batch_size=batch_size)

        model = model.to(device)
        if is_ddp:
            model = torch.nn.parallel.DistributedDataParallel(model)
        
        self.loss_func = nn.CrossEntropyLoss() 
        self.optimizer = torch.optim.Adam(model.parameters())
        if is_2nd_order:
            self.preconditioner = kfac.preconditioner.KFACPreconditioner(model=model)
            rpc_distributed.KFacRPCCommunicator(world_size=world_size, rank=rank, preconditioner=self.preconditioner,model=model)
        else:
            self.preconditioner = None

        self.dataset_name = dataset_name 
        self.device = device
        self.model = model
        self.epochs = epochs
        self.world_size = world_size
        self.rank = rank
        self.model_avg_interval = interval
        self.is_ddp = is_ddp
        self.batch_size = batch_size
        self.is_fault = False

    def init_mischief(self,disconnect_ratio=0,max_sick_iter_ratio=0.2,max_disconnected_node_num = 2, possible_disconnect_node = None):
        max_disconnect_iter = int(len(self.data_manager.train_dataset) / self.batch_size / self.world_size * max_sick_iter_ratio)
        mischief.mischief_init(world_size=self.world_size, possible_disconnect_node=possible_disconnect_node,
                           max_disconnect_iter=max_disconnect_iter, disconnect_ratio=disconnect_ratio,
                            max_disconnected_node_num=max_disconnected_node_num,
                           ddp_trigger=True, factor_comm_trigger=True, inverse_comm_trigger=True)
        self.is_fault = True
        self.experiment_name_detail = f"mdn{max_disconnected_node_num}_dr{disconnect_ratio}_mdi{max_disconnect_iter}_ws{self.world_size}_avg{self.model_avg_interval}"

    def train_and_test(self,log_dir,experiment_name,timestamp):
        model = self.model
        model_name =  type(model).__name__
        if hasattr(model, "model_name"):
            model_name = model.model_name
        if hasattr(self, "experiment_name_detail"):
            experiment_name = f"{experiment_name}_{self.experiment_name_detail}"
        writer_name = f"{self.dataset_name}/{model_name}/{experiment_name}/{timestamp}/{dist.get_rank()}"
        self.writer = SummaryWriter(
            log_dir=os.path.join(log_dir, writer_name)) 

        for i in range(0, self.epochs):
            self.train_with_rpc(epoch=i)
            #self.test_all(epoch=i)
        self.writer.close()

    def train_with_rpc(self, epoch):
        self.model.train()
        self.data_manager.set_epoch(epoch)
        train_loader = self.data_manager.train_loader
        with tqdm(
                total=math.ceil(len(train_loader)),
                bar_format='{l_bar}{bar:6}{r_bar}',
                desc=f'Epoch {epoch:3d}/{self.epochs:3d}',
                disable=(self.rank != 0)
        ) as t:
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                mischief.update_iter()
                rpc_distributed.global_communicator.update_self_t()
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_func(output, target)
                loss.backward()
                if self.preconditioner is not None:
                    self.preconditioner.step()
                self.optimizer.step()
                if not self.is_ddp and self.model_avg_interval > 0:
                    if rpc_distributed.global_communicator.current_t() % self.model_avg_interval == 0:
                        rpc_distributed.global_communicator.send_model_param()
                rpc_distributed.global_communicator.facotr_comput_lazy_wl_rebal()
                t.update()
            rpc_distributed.global_communicator.clear_count_dict()
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', loss.item(), epoch)

    def train(self, epoch):
        self.model.train()
        self.data_manager.set_epoch(epoch)
        train_loader = self.data_manager.train_loader
        with tqdm(
                total=math.ceil(len(train_loader)),
                bar_format='{l_bar}{bar:6}{r_bar}',
                desc=f'Epoch {epoch:3d}/{self.epochs:3d}',
                disable=(self.rank != 0)
        ) as t:
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                mischief.update_iter()
                rpc_distributed.global_communicator.update_self_t()
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_func(output, target)
                loss.backward()
                if self.preconditioner is not None:
                    self.preconditioner.step()
                self.optimizer.step()
                if not self.is_ddp and self.model_avg_interval > 0:
                    if mischief.ITER >= mischief.LAST_AVG_ITER + self.model_avg_interval :
                        fut_list = mischief.average_health_nodes_param_async(self.model)
                        torch.futures.wait_all(fut_list)
                rpc_distributed.global_communicator.facotr_comput_lazy_wl_rebal()
                t.update()
            rpc_distributed.global_communicator.clear_count_dict()
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', loss.item(), epoch)

    def test_all(self, epoch):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.data_manager.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        # 把 correct 和 total 转换成tensor以便进行分布式计算
        correct_total_tensor = torch.tensor([correct,total]).to(self.device)

        # 使用dist.reduce把所有节点的correct和total累加到rank 0节点
        dist.all_reduce(correct_total_tensor)

        # 只在rank 0上计算最终的准确率并记录
        if self.writer is not None and self.rank == 0:  # 假设self.rank存储了当前进程的rank
            correct_sum, total_sum = correct_total_tensor.unbind()
            accuracy = correct_sum.item() / total_sum.item()
            self.writer.add_scalar('Accuracy/test', accuracy, epoch)

    def average_health_nodes_param_tensor_fusion_async(self):
        model = self.model
        health_nodes = mischief.get_health_nodes()
        ratio = 0
        result_list = []
        if dist.get_rank() in mischief.POSSIBLE_DISCONNECTED_NODE:
            ratio = mischief.sick_weight_magnification_ratio / len(health_nodes)
        else:
            ratio = mischief.health_weight_magnification_ratio / len(health_nodes)
        
        flat_tensor = fuse_model_paramenters(model)
        if dist.get_rank() in health_nodes:
            fut = dist.all_reduce(flat_tensor, op=dist.ReduceOp.SUM,async_op=True).get_future()
            fut.then(lambda fut: fut.value()[0].mul_(ratio)).then(lambda fut: unfuse_tensors_to_model(fut.value()[0], model))
            result_list.append(fut)
        else:
            result_list.append(dist.all_reduce(torch.zeros_like(flat_tensor), op=dist.ReduceOp.SUM,async_op=True).get_future())
            mischief.LAST_AVG_ITER = mischief.ITER
        return result_list