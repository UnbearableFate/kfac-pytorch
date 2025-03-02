import math
import random
import time
import torch
import datetime
from tqdm import tqdm
from general_util.data_preparation import DataPreparer
import torch.distributed as dist
import logging
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import kfac
from general_util.tensor_funsion import fuse_tensors, fuse_model_paramenters, unfuse_tensors_to_model
import kfac.rpc_distributed as rpc_distributed
from  kfac.rpc_util.fault_sim import fault_simulator
from kfac.rpc_util.common_util import get_model_total_l2_norm
from general_util.consts import CHECK_POINT_PATH, DATA_DIR, LOG_DIR, SHARE_FILES_DIR
from general_util.lr_schduler import get_scheduler ,WarmupScheduler
class GeneralManager:
    def __init__(self,experiment_name:str, dataset_name, model, 
                 sampler_func = None, train_com_method="ddp", is_2nd_order =True,
                   epochs=100, batch_size =64, device=torch.device("cuda:0"), timestamp="",
                   transform_train=None, transform_test=None, precondtioner=None ,recover = False):
        self.experiment_name_detail = None
        self.writer = None
        batch_size=batch_size
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        model_name = type(model).__name__
        if hasattr(model, "model_name"):
            model_name = model.model_name
        if recover:
            log_detail = f"{dataset_name}/{model_name}/{experiment_name}_re"
        else:
            log_detail = f"{dataset_name}/{model_name}/{experiment_name}_{timestamp}"
        log_dir = os.path.join(LOG_DIR,log_detail)
        self.log_dir = log_dir

        try :
            os.makedirs(log_dir)
        except FileExistsError:
            pass
        except Exception as e:
            raise RuntimeError(f"Unable to create log directory: {log_dir}")

        self.data_manager = DataPreparer(data_path_root=DATA_DIR, dataset_name=dataset_name, world_size=world_size, rank=rank,
                                         sampler=sampler_func, batch_size=batch_size, train_transform=transform_train, test_transform=transform_test,train_com_method=train_com_method)

        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(params=model.parameters(),lr=0.0001, momentum = 0.9) #torch.optim.Adam(model.parameters())
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.0008, epochs = epochs, steps_per_epoch = len(self.data_manager.train_loader))
        #self.warmup_scheduler = WarmupScheduler(self.optimizer, warmup_epochs=5, base_lr=self.optimizer.param_groups[0]['lr'])
        #self.decay_scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs-5)
        #self.optimizer = torch.optim.Adam(model.parameters())

        if is_2nd_order:
            assert precondtioner is not None
            self.preconditioner = precondtioner
        else:
            self.preconditioner = None

        if train_com_method == "rpc":
                self.rpc_communicator:rpc_distributed.KFacRPCCommunicator \
                = rpc_distributed.KFacRPCCommunicator(world_size=world_size, rank=rank,
                    preconditioner=self.preconditioner, model=model,
                    share_file_path=SHARE_FILES_DIR, timestamp=timestamp,
                    log_dir = log_dir, device=device)

        self.start_epoch = 0
        self.checkpoint_file_path = os.path.join(CHECK_POINT_PATH, experiment_name, f"{rank}.pth")
        current_checkpoint_path = os.path.join(CHECK_POINT_PATH, experiment_name)

        self.train_total_time = 0
        if not os.path.exists(current_checkpoint_path):
            if rank == 0:
                os.makedirs(current_checkpoint_path)
        elif os.path.exists(self.checkpoint_file_path) and recover:
            checkpoint = torch.load(self.checkpoint_file_path)
            model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.preconditioner.load_state_dict(checkpoint["preconditioner"])
            if "scheduler" in checkpoint and hasattr(self, "schduler"):
                self.schduler.load_state_dict(checkpoint["scheduler"])
            if "decay_scheduler" in checkpoint and hasattr(self, "decay_scheduler"):
                self.decay_scheduler.load_state_dict(checkpoint["decay_scheduler"])
            if "warmup_scheduler" in checkpoint and hasattr(self, "warmup_scheduler"):
                self.warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.train_total_time = checkpoint["train_total_time"]
            print(f"Checkpoint loaded in rank {rank} at epoch {self.start_epoch}")
        dist.barrier()

        self.dataset_name = dataset_name
        self.device = device
        self.model = model
        self.epochs = epochs
        self.world_size = world_size
        self.rank = rank
        self.train_com_method = train_com_method
        self.batch_size = batch_size
        self.is_fault = False

        if fault_simulator is not None:
            fault_simulator.train_total_time_cb = self.get_total_training_time

    def train_and_test(self):
        writer_path = self.log_dir
        if self.experiment_name_detail is not None:
            writer_path = os.path.join(writer_path,self.experiment_name_detail)
        writer_name = os.path.join(writer_path,str(self.rank))
        self.writer = SummaryWriter(
            log_dir=writer_name)

        for i in range(self.start_epoch, self.epochs):
            self.train(epoch=i)
            self.test_all(epoch=i)
            self.save_checkpoint(epoch=i)
            
            train_total_time = torch.tensor(self.train_total_time, dtype=torch.int, device="cuda")  # 或者"cpu"
            dist.all_reduce(train_total_time)
            if train_total_time.item() / self.world_size > 950:
                break

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

        for i in range(self.start_epoch, self.epochs):
            if self.preconditioner is None:
                self.ad_sgd_train(epoch=i)
            else:
                self.ad_kfac_train(epoch=i)
            self.test_local(epoch=i)
            self.save_checkpoint(epoch=i)

        self.writer.close()
        print(f"Rank {self.rank} : total train time: {self.train_total_time}")
        print(f"Rank {self.rank} : {self.rpc_communicator.com_statistic} at iteration {self.rpc_communicator.current_t()}")
        print(f"Rank {self.rank} : real fault rate {fault_simulator.fault_total_time / self.train_total_time}")
        dist.barrier()

    def close_all(self):
        if rpc_distributed.rpc.is_available():
            self.rpc_communicator.close_rpc()
        if dist.is_initialized():
            dist.destroy_process_group()

    def train(self, epoch):
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
                start_time = time.time()
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_func(output, target)
                loss.backward()

                #if self.rank not in [0,6,9,15]:
                #    time.sleep(0.5)
                #time.sleep(delay_list_dict[0][self.rank])

                if self.preconditioner is not None:
                    self.preconditioner.step()
                self.optimizer.step()
                #self.scheduler.step()
                t.update()
                self.train_total_time += time.time() - start_time
        
        if hasattr(self, "warmup_scheduler"):
            if epoch < 5:
                self.warmup_scheduler.step()
            else:
                self.decay_scheduler.step()
        if self.writer is not None:
            self.writer.add_scalar('Loss/train', loss.item(), epoch)
            self.writer.add_scalar('Total train time', self.train_total_time, epoch)
            self.writer.add_scalar('LR/train', self.optimizer.param_groups[0]["lr"], epoch)

    def ad_kfac_train(self, epoch):
        self.model.train()
        self.data_manager.set_epoch(epoch)
        train_loader = self.data_manager.train_loader
        com = rpc_distributed.global_communicator
        with (tqdm(
                total=math.ceil(len(train_loader)),
                bar_format='{l_bar}{bar:6}{r_bar}',
                desc=f'Epoch {epoch:3d}/{self.epochs:3d}',
                disable=(self.rank != 0)
        ) as t):
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                start_time = time.time()

                rpc_distributed.global_communicator.update_self_t()
                self.optimizer.zero_grad()
                
                output = self.model(data)
                loss = self.loss_func(output, target)
                self.rpc_communicator.model_avg_rpc.set_loss(loss.item())
                loss.backward()

                if self.preconditioner is not None:
                    self.preconditioner.step()

                self.optimizer.step()
                if hasattr(self, "scheduler"):
                    self.scheduler.step()
                self.rpc_communicator.send_model_param()
                
                if com.current_t() % 30 == 29:
                    rpc_distributed.global_communicator.factor_computation_lazy_rebalance()
                    #rpc_distributed.global_communicator.task_reassign_rpc.electing_new_leader_loop()
                
                if rpc_distributed.global_communicator.current_t() % 200 == 199:
                    rpc_distributed.global_communicator.task_reassign_rpc.check_and_reassign()
                
                if com.task_reassign_rpc.reassign_task_callback is not None:
                    com.task_reassign_rpc.reassign_task_callback()
                if com.update_assignment_callback is not None:
                    com.update_assignment_callback()
                    
                if rpc_distributed.global_communicator.current_t() % 200 == 0:
                    rpc_distributed.global_communicator.print_rpc_state()
                
                self.train_total_time += time.time() - start_time
                t.update()
        
        if hasattr(self, "warmup_scheduler"):
            if epoch < 5:
                self.warmup_scheduler.step()
            else:
                self.decay_scheduler.step()
            
        if self.writer is not None:
            self.writer.add_scalar("Total train time", self.train_total_time, epoch)
            self.writer.add_scalar('Loss/train', loss.item(), epoch)
            self.writer.add_scalar('LR/train', self.optimizer.param_groups[0]['lr'], epoch)
    
    def ad_sgd_train(self, epoch):
        self.model.train()
        self.data_manager.set_epoch(epoch)
        train_loader = self.data_manager.train_loader
        com = rpc_distributed.global_communicator
        with (tqdm(
                total=math.ceil(len(train_loader)),
                bar_format='{l_bar}{bar:6}{r_bar}',
                desc=f'Epoch {epoch:3d}/{self.epochs:3d}',
                disable=(self.rank != 0)
        ) as t):
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                start_time = time.time()

                rpc_distributed.global_communicator.update_self_t()
                self.optimizer.zero_grad()
                
                output = self.model(data)
                loss = self.loss_func(output, target)
                self.rpc_communicator.model_avg_rpc.set_loss(loss.item())
                loss.backward()

                if self.preconditioner is not None:
                    self.preconditioner.step()

                self.optimizer.step()
                self.rpc_communicator.send_model_param()
                    
                if rpc_distributed.global_communicator.current_t() % 200 == 0:
                    rpc_distributed.global_communicator.print_rpc_state()
                
                self.train_total_time += time.time() - start_time
                t.update()
        
        if hasattr(self, "scheduler"):
                self.scheduler.step()
        elif hasattr(self, "warmup_scheduler"):
            if epoch < 5:
                self.warmup_scheduler.step()
            else:
                self.decay_scheduler.step() 
            
        if self.writer is not None:
            self.writer.add_scalar("Total train time", self.train_total_time, epoch)
            self.writer.add_scalar('Loss/train', loss.item(), epoch)
            self.writer.add_scalar('LR/train', self.optimizer.param_groups[0]['lr'], epoch)


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
            time_as_step = round(self.train_total_time * 1000)
            self.writer.add_scalar('test_accuracy/train_time', accuracy, time_as_step)
            self.writer.add_scalar('test_accuracy/train_epoch', accuracy, epoch)

    def test_local(self, epoch):
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
        time_as_step = round(self.train_total_time * 1000)
        self.rpc_communicator.model_avg_rpc.set_acc(accuracy)
        self.writer.add_scalar('test_accuracy/train_time', accuracy, time_as_step)
        self.writer.add_scalar('test_accuracy/train_epoch', accuracy, epoch)

    def save_checkpoint(self, epoch):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'preconditioner': self.preconditioner.state_dict() if self.preconditioner is not None else None,
            'epoch': epoch,
            'train_total_time': self.train_total_time,
        }
        if hasattr(self, "scheduler"):
            state["scheduler"] = self.scheduler.state_dict()
        if hasattr(self, "warmup_scheduler"):
            state["warmup_scheduler"] = self.warmup_scheduler.state_dict()
        if hasattr(self, "decay_scheduler"):
            state["decay_scheduler"] = self.decay_scheduler.state_dict()
        try:
            temp_path = self.checkpoint_file_path + ".temp"
            torch.save(state, temp_path)
            os.rename(temp_path,self.checkpoint_file_path)   
        except Exception as e:
            print(f"Save checkpoint error: {e} in rank {self.rank} at epoch {epoch} file path {self.checkpoint_file_path}")

    def get_total_training_time(self):
        return self.train_total_time