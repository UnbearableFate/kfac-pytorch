import gc
import math
import time
import datetime

import psutil
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

ompi_world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', -1))
ompi_world_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', -1))
class GeneralManager:
    def __init__(self, data_dir, dataset_name, model, sampler_func = None, train_com_method="ddp", interval=10,
                 is_2nd_order =True, epochs=100, batch_size =64, device=torch.device("cuda:0"), share_file_path=None, timestamp="", log_dir ='',
                 transform_train=None, transform_test=None,
                 precondtioner=None):
        self.experiment_name_detail = None
        self.writer = None
        batch_size=batch_size
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        model_name = type(model).__name__
        if hasattr(model, "model_name"):
            model_name = model.model_name
        log_detail = f"{dataset_name}/{model_name}/{timestamp}"
        log_dir = os.path.join(log_dir,log_detail)
        self.log_dir = log_dir

        try :
            os.makedirs(log_dir)
        except FileExistsError:
            pass
        except Exception as e:
            raise RuntimeError(f"Unable to create log directory: {log_dir}")

        self.data_manager = DataPreparer(data_path_root=data_dir, dataset_name=dataset_name, world_size=world_size, rank=rank,
                                         sampler=sampler_func, batch_size=batch_size, train_transform=transform_train, test_transform=transform_test)

        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(params=model.parameters(),lr=0.006, momentum = 0.8) #torch.optim.Adam(model.parameters())
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.1, steps_per_epoch=50, epochs=epochs)
        if is_2nd_order:
            if precondtioner is not None:
                self.preconditioner = precondtioner
            else:
                self.preconditioner = kfac.preconditioner.KFACPreconditioner(model=model)
            if train_com_method == "rpc":
                self.rpc_communicator = rpc_distributed.KFacRPCCommunicator(world_size=world_size, rank=rank,
                                                                            preconditioner=self.preconditioner,model=model ,
                                                                            share_file_path=share_file_path, timestamp=timestamp ,
                                                                            log_dir = log_dir, device=device)
        else:
            self.preconditioner = None

        self.dataset_name = dataset_name
        self.device = device
        self.model = model
        self.epochs = epochs
        self.world_size = world_size
        self.rank = rank
        self.model_avg_interval = interval
        self.train_com_method = train_com_method
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
        if self.train_com_method == "rpc":
            mischief.recover_func = self.rpc_communicator.restart_sick_node

    def train_and_test(self):
        writer_path = self.log_dir
        if self.experiment_name_detail is not None:
            writer_path = os.path.join(writer_path,self.experiment_name_detail)
        writer_name = os.path.join(writer_path,str(self.rank))
        self.writer = SummaryWriter(
            log_dir=writer_name)

        for i in range(0, self.epochs):
            self.train(epoch=i)
            self.test_all(epoch=i)

        self.writer.close()

    def rpc_train_and_test(self):
        writer_path = self.log_dir
        if self.experiment_name_detail is not None:
            writer_path = os.path.join(writer_path, self.experiment_name_detail)
        writer_name = os.path.join(writer_path, str(self.rank))
        self.writer = SummaryWriter(
            log_dir=writer_name)
        self.rpc_communicator.writer = self.writer
        dist.barrier()
        print(f"rpc OK? {rpc_distributed.rpc.is_available()} ,dist OK? {dist.is_initialized()} in rank {self.rank}")

        for i in range(0, self.epochs):
            self.rpc_train(epoch=i)
            self.test_by_rpc(epoch=i)
            self.rpc_communicator.write_model_test_accuracy(i,self.epochs)

        self.writer.close()
        dist.barrier()

    def close_all(self):
        if rpc_distributed.rpc.is_available():
            self.rpc_communicator.close_rpc()
        if dist.is_initialized():
            dist.destroy_process_group()

    def train(self, epoch):
        start_time = time.time()
        self.model.train()
        self.data_manager.set_epoch(epoch)
        train_loader = self.data_manager.train_loader
        with (tqdm(
                total=math.ceil(len(train_loader)),
                bar_format='{l_bar}{bar:6}{r_bar}',
                desc=f'Epoch {epoch:3d}/{self.epochs:3d}',
                disable=(self.rank != 0)
        ) as t):
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                #mischief.update_iter()
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_func(output, target)
                loss.backward()
                if self.preconditioner is not None:
                    self.preconditioner.step()
                self.optimizer.step()
                self.scheduler.step()
                t.update()

            if self.writer is not None:
                self.writer.add_scalar('Loss/train', loss.item(), epoch)
                self.writer.add_scalar('Time/train',time.time() - start_time, epoch)

    def rpc_train(self, epoch):
        start_time = time.time()
        self.model.train()
        self.data_manager.set_epoch(epoch)
        train_loader = self.data_manager.train_loader
        #print(f"Total batches: {len(train_loader)} at rank {self.rank}")
        data_iter = iter(train_loader)
        data, target = next(data_iter)
        #print(f"Data shape: {data.shape}, Target shape: {target.shape} at rank {self.rank}")
        with (tqdm(
                total=math.ceil(len(train_loader)),
                bar_format='{l_bar}{bar:6}{r_bar}',
                desc=f'Epoch {epoch:3d}/{self.epochs:3d}',
                disable=(self.rank != 0)
        ) as t):
            for batch_idx, (data, target) in enumerate(train_loader):
                rpc_distributed.global_communicator.update_self_t()
                rpc_distributed.global_communicator.send_model_param()
                '''
                mischief.update_iter()
                if self.is_fault:
                    if mischief.is_sick_at(self.rank):
                        time.sleep(0.1)
                '''
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_func(output, target)
                loss.backward()

                if self.preconditioner is not None:
                    self.preconditioner.step()

                self.optimizer.step()
                self.scheduler.step()

                self.rpc_communicator.model_avg_rpc.set_loss(loss.item())

                if batch_idx % 50 == 0:
                    rpc_distributed.global_communicator.print_rpc_state()

                """
                rpc_distributed.global_communicator.facotr_comput_lazy_wl_rebal()
                rpc_distributed.global_communicator.task_reassign_rpc.check_and_reassign()
                self.rpc_communicator.task_reassign_rpc.electing_new_leader_loop()

                if self.rpc_communicator.task_reassign_rpc.reassign_task_callback is not None:
                    self.rpc_communicator.task_reassign_rpc.reassign_task_callback()
                if self.rpc_communicator.update_assignment_callback is not None:
                    self.rpc_communicator.update_assignment_callback()
                if self.rpc_communicator.send_model_param_callback is not None:
                    self.rpc_communicator.send_model_param_callback()

                if self.writer is not None and batch_idx % 20 == 0:
                    process = psutil.Process(os.getpid())
                    self.writer.add_scalar('Memory', process.memory_info().rss / 1024**3, (epoch+1)*batch_idx)
                    allocated_memory = torch.cuda.memory_allocated(self.rank%4)  # 0 表示 GPU 0
                    cached_memory = torch.cuda.memory_reserved(self.rank%4)  # 0 表示 GPU 0
                    self.writer.add_scalar('Memory/GPU_Allocated', allocated_memory / 1024**3, (epoch+1)*batch_idx)
                    self.writer.add_scalar('Memory/GPU_Cached', cached_memory / 1024**3, (epoch+1)*batch_idx)
                """
                #rpc_distributed.global_communicator.print_rpc_state(f"end epoch {epoch} batch {batch_idx}")
                t.update()
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', loss.item(), epoch)
                self.writer.add_scalar('Time/train',time.time() - start_time, epoch)

    def train_communication_allreduce_avg(self):
        if mischief.ITER >= mischief.LAST_AVG_ITER + self.model_avg_interval :
            fut_list = mischief.average_health_nodes_param_async(self.model)
            torch.futures.wait_all(fut_list)

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

    def test_by_rpc(self, epoch):
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
        rpc_distributed.global_communicator.send_rpc_test_result(correct, total, epoch)

    def write_test_result_rpc(self):
        for e in range(self.epochs):
            accuracy = rpc_distributed.global_communicator.wait_and_return_test_result(e)
            self.writer.add_scalar('Accuracy/test', accuracy, e)

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
