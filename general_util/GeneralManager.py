import math
import random
import time
import torch
from tqdm import tqdm
from general_util.data_preparation import DataPreparer, NonIidSampler
from functools import partial
import torch.distributed as dist
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import kfac.rpc_distributed as rpc_distributed
from  kfac.rpc_util.fault_sim import fault_simulator
from general_util.consts import CHECK_POINT_PATH, DATA_DIR, LOG_DIR, SHARE_FILES_DIR
from examples.vision.optimizers import get_optimizer
from .optimizers import get_kfac_preconditioner, get_swin_optimizer
import csv

class GeneralManager:
    def __init__(self, model, sampler_func = None,
                 transform_train=None, transform_test=None,
                 device=None,
                 args=None):
        
        experiment_name = args.experiment_name
        dataset_name = args.dataset_name
        train_com_method = args.train_com_method
        epochs = args.epochs
        batch_size = args.batch_size
        recover = args.recover
        timestamp = args.timestamp
        self.writer = None
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

        if args.degree_noniid > 0:
            sampler_func =  partial(NonIidSampler, degree_noniid=args.degree_noniid)
        self.data_manager = DataPreparer(args)
        self.loss_func =nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        self.optimizer, self.lr_scheduler = get_swin_optimizer(model, args)
        if args.not_kfac:
            self.preconditioner = None
            self.kfac_scheduler = None
        else:
            self.preconditioner,self.kfac_scheduler = get_kfac_preconditioner(model, args,self.optimizer)
        self.scaler = torch.amp.GradScaler("cuda") if args.amp else None
        if train_com_method == "rpc":
                self.rpc_communicator:rpc_distributed.KFacRPCCommunicator \
                = rpc_distributed.KFacRPCCommunicator(world_size=world_size, rank=rank,
                    preconditioner=self.preconditioner, model=model,
                    share_file_path=SHARE_FILES_DIR, timestamp=timestamp,
                    log_dir = log_dir, device=device ,steps_per_epoch=len(self.data_manager.train_loader))

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
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.train_total_time = checkpoint["train_total_time"]
            if "scaler" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler"])
            print(f"Checkpoint loaded in rank {rank} at epoch {self.start_epoch}")
        self.dataset_name = dataset_name
        self.device = device
        self.model = model
        self.epochs = epochs
        self.world_size = world_size
        self.rank = rank
        self.train_com_method = train_com_method
        self.batch_size = batch_size
        self.is_fault = False
        self.fault_in_last_iteraion = False
        self.args = args

        if fault_simulator is not None:
            fault_simulator.train_total_time_cb = self.get_total_training_time
        
        if rank == 0:
            print(f"Model: {model_name}, Dataset: {dataset_name}, Experiment: {experiment_name}, Epochs: {epochs}, Batch size: {batch_size}, Recover: {recover}, Timestamp: {timestamp}")
            print(f"Optimizer: {self.optimizer}, LR Scheduler: {self.lr_scheduler}, Train Communication Method: {train_com_method}")

    def train_and_test(self):
        writer_path = self.log_dir
        writer_name = os.path.join(writer_path,str(self.rank))
        self.writer = SummaryWriter(
            log_dir=writer_name)

        for i in range(self.start_epoch+1, self.epochs+1):
            start_time = time.time()
            self.train(epoch=i)
            self.train_total_time += time.time() - start_time
            if self.writer is not None:
                self.writer.add_scalar("Total train time", self.train_total_time, i) 
            self.lr_scheduler.step()
            if self.kfac_scheduler is not None:
                self.kfac_scheduler.step()
            if i % 10 == 0 or i >= 0.9 * self.epochs:
                self.test_all(epoch=i)
            self.save_checkpoint(epoch=i)
            """
            train_total_time = torch.tensor(self.train_total_time, dtype=torch.int, device="cuda")  # 或者"cpu"
            dist.all_reduce(train_total_time)
            if train_total_time.item() / self.world_size > 950:
                break
            """

        self.writer.close()

    def rpc_train_and_test(self):
        writer_path = self.log_dir
        writer_name = os.path.join(writer_path, str(self.rank))
        self.writer = SummaryWriter(
            log_dir=writer_name)
        self.rpc_communicator.writer = self.writer
        dist.barrier()
        print(f"rpc OK? {rpc_distributed.rpc.is_available()} ,dist OK? {dist.is_initialized()} in rank {self.rank}")

        for i in range(self.start_epoch+1, self.epochs+1):
            start_time = time.time()
            if self.preconditioner is None:
                loss = self.ad_sgd_train(epoch=i)
            else:
                loss = self.ad_kfac_train(epoch=i)
            self.train_total_time += time.time() - start_time
            if self.writer is not None:
                self.writer.add_scalar("Total train time", self.train_total_time,i)
                self.writer.add_scalar('Loss/train', loss, i)
                self.writer.add_scalar('LR/train', self.optimizer.param_groups[0]['lr'], i)
            self.lr_scheduler.step()
            if self.kfac_scheduler is not None:
                self.kfac_scheduler.step(step=i) 
            if i % 10 == 0 or i >= 0.9 * self.epochs:
                self.test_local(epoch=i)
            self.save_checkpoint(epoch=i)

        self.writer.close()
        """
        if self.preconditioner is not None:
            for factor_type, stat in self.rpc_communicator.execution_times_statistic.items():
                with open(os.path.join(self.log_dir, f"execution_time_{factor_type}_{self.rank}.csv"), 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(["shape", "avg_time"])
                    for shape, time_stat in stat.items():
                        avg_time = time_stat[0] / time_stat[1]
                        csv_writer.writerow([shape, avg_time])"
        """

        print(f"Rank {self.rank} : total train time: {self.train_total_time}")
        print(f"Rank {self.rank} : {self.rpc_communicator.com_statistic} at iteration {self.rpc_communicator.current_t()}")
        print(f"Rank {self.rank} : real fault rate {fault_simulator.fault_total_time / self.train_total_time}")

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
                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda",enabled=self.scaler is not None):
                    output = self.model(data)
                    loss = self.loss_func(output, target)

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    if self.args.clip_grad_norm is not None:
                        # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    if self.preconditioner is not None:
                        self.preconditioner.step()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.args.clip_grad_norm is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    if self.preconditioner is not None:
                        self.preconditioner.step()
                    self.optimizer.step()
                t.update() 
        if self.writer is not None:
            self.writer.add_scalar('Loss/train', loss.item(), epoch)
            self.writer.add_scalar('LR/train', self.optimizer.param_groups[0]["lr"], epoch)

    def random_delay(self, delay_num,delay_time):
        if self.fault_in_last_iteraion :
            self.fault_in_last_iteraion = False
            return
        delay_ranks = random.choices(range(self.world_size), k=delay_num)
        if self.rank in delay_ranks:
            time.sleep(delay_time)
            self.fault_in_last_iteraion = True

    def fault_simulation(self):
        fault_simulator.update_fault_status()
        while fault_simulator.is_fault():
            time.sleep(0.1)
            fault_simulator.update_fault_status()

        if fault_simulator.recover_flg and self.preconditioner is not None:
            self.rpc_communicator.task_reassign_rpc.resurrection_declaration()
            fault_simulator.recover_flg = False

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

                rpc_distributed.global_communicator.update_self_t()
                self.optimizer.zero_grad()

                with torch.amp.autocast(device_type="cuda",enabled=self.scaler is not None):
                    output = self.model(data)
                    loss = self.loss_func(output, target)
                    self.rpc_communicator.model_avg_rpc.set_loss(loss.item())

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    if self.args.clip_grad_norm is not None:
                        # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    if self.preconditioner is not None:
                        self.preconditioner.step()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.args.clip_grad_norm is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    if self.preconditioner is not None:
                        self.preconditioner.step()
                    self.optimizer.step()

                self.rpc_communicator.send_model_param()
                
                if com.current_t() % 50 == 40:
                    rpc_distributed.global_communicator.factor_computation_lazy_rebalance()
                    rpc_distributed.global_communicator.task_reassign_rpc.electing_new_leader_loop()
                
                if rpc_distributed.global_communicator.current_t() % 100 == 90:
                    rpc_distributed.global_communicator.task_reassign_rpc.check_and_reassign()

                if com.task_reassign_rpc.reassign_task_callback is not None:
                    com.task_reassign_rpc.reassign_task_callback()
                if com.update_assignment_callback is not None:
                    com.update_assignment_callback()
                    
                if rpc_distributed.global_communicator.current_t() % 200 == 0:
                    rpc_distributed.global_communicator.print_rpc_state()

                t.update()
        return loss.item()
    
    def ad_sgd_train(self, epoch):
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
                rpc_distributed.global_communicator.update_self_t()
                self.optimizer.zero_grad()

                with torch.amp.autocast(device_type="cuda",enabled=self.scaler is not None):
                    output = self.model(data)
                    loss = self.loss_func(output, target)
                    self.rpc_communicator.model_avg_rpc.set_loss(loss.item())

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    if self.args.clip_grad_norm is not None:
                        # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.args.clip_grad_norm is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    self.optimizer.step()

                self.rpc_communicator.send_model_param()

                if rpc_distributed.global_communicator.current_t() % 200 == 0:
                    rpc_distributed.global_communicator.print_rpc_state()
                t.update()
        
        return loss.item()

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
            'lr_scheduler': self.lr_scheduler.state_dict()
        }
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()

        try:
            temp_path = self.checkpoint_file_path + ".temp"
            torch.save(state, temp_path)
            os.rename(temp_path,self.checkpoint_file_path)   
        except Exception as e:
            print(f"Save checkpoint error: {e} in rank {self.rank} at epoch {epoch} file path {self.checkpoint_file_path}")

    def get_total_training_time(self):
        return self.train_total_time