import math
import time

import torch
from tqdm import tqdm
import kfac.mischief as mischief
from general_util.data_preparation import DataPreparer
import torch.distributed as dist

class GeneralManager:
    def __init__(self, model, data_manager: DataPreparer, loss_func, optimizer, preconditioner, epochs, world_size,
                 rank, device, writer ,interval=10):
        self.writer = writer
        self.device = device
        self.model = model
        self.data_manager = data_manager
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.preconditioner = preconditioner
        self.epochs = epochs
        self.world_size = world_size
        self.rank = rank
        self.model_avg_interval = interval

    def train_and_test(self):
        for i in range(0, self.epochs):
            self.train(epoch=i)
            #if mischief.ITER > mischief.LAST_AVG_ITER :
            #    mischief.average_health_nodes_param(self.model)
            self.test_all(epoch=i)

    def train_and_test_with_hook_ddp(self):
        mischief.add_hook_to_model(self.model)
        for i in range(0, self.epochs):
            self.train_ddp(epoch=i)
            #if mischief.ITER > mischief.LAST_AVG_ITER :
            #    mischief.average_health_nodes_param(self.model)
            self.test_all(epoch=i)
    

    def train_and_test_async(self):
        for i in range(0, self.epochs):
            self.train_async_avg(epoch=i)
            if mischief.ITER > mischief.LAST_AVG_ITER :
                futs = mischief.average_health_nodes_param_async(self.model)
                torch.futures.wait_all(futs)
            self.test_all(epoch=i)
    
    def train_and_test_semi_async(self):
        for i in range(0, self.epochs):
            self.train_semi_async_avg(epoch=i)
            if mischief.ITER > mischief.LAST_AVG_ITER :
                futs = mischief.average_health_nodes_param_async(self.model)
                torch.futures.wait_all(futs)
            self.test_all(epoch=i)
    
    def train_and_test_with_normal_ddp(self):
        for i in range(0, self.epochs):
            self.train_ddp(epoch=i)
            self.test_all(epoch=i)

    def train_ddp(self, epoch):
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
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_func(output, target)
                loss.backward()
                if self.preconditioner is not None:
                    self.preconditioner.step()
                self.optimizer.step()
                t.update()
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
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_func(output, target)
                loss.backward()
                if self.preconditioner is not None:
                    self.preconditioner.step()
                self.optimizer.step()
                if mischief.ITER >= mischief.LAST_AVG_ITER + self.model_avg_interval :
                    mischief.average_health_nodes_param(self.model)
                t.update()
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', loss.item(), epoch)

    def train_semi_async_avg(self, epoch):
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

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_func(output, target)
                loss.backward()
                
                if self.preconditioner is not None:
                    self.preconditioner.step()
                self.optimizer.step()
                
                if mischief.ITER >= mischief.LAST_AVG_ITER + self.model_avg_interval :
                    futs = mischief.average_health_nodes_param_async(self.model)
                    torch.futures.wait_all(futs)
                t.update()

            if self.writer is not None:
                self.writer.add_scalar('Loss/train', loss.item(), epoch)

    def train_async_avg(self, epoch):
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

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_func(output, target)
                loss.backward()
                
                futs = None
                if mischief.ITER >= mischief.LAST_AVG_ITER + self.model_avg_interval :
                    futs = mischief.average_health_nodes_param_async(self.model) 

                if self.preconditioner is not None:
                    self.preconditioner.step()
                
                if futs is not None:
                    torch.futures.wait_all(futs)
                
                self.optimizer.step()
                t.update()

            if self.writer is not None:
                self.writer.add_scalar('Loss/train', loss.item(), epoch)
    
    def test(self, epoch):
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
        accuracy = correct / total
        self.writer.add_scalar('Accuracy/test', accuracy, epoch)

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
        #print(f"Rank {self.rank} correct_tensor: {correct_total_tensor} in epoch {epoch}")
        #dist.barrier()
        dist.all_reduce(correct_total_tensor)

        # 只在rank 0上计算最终的准确率并记录
        if self.rank == 0:  # 假设self.rank存储了当前进程的rank
            correct_sum, total_sum = correct_total_tensor.unbind()
            accuracy = correct_sum.item() / total_sum.item()
            self.writer.add_scalar('Accuracy/test', accuracy, epoch)
