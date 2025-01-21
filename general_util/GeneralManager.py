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
from  kfac.rpc_util.fault_sim import fault_simulator
from kfac.rpc_util.common_util import get_model_total_l2_norm
from general_util.consts import CHECK_POINT_PATH, DATA_DIR, LOG_DIR, SHARE_FILES_DIR


class GeneralManager:
    def __init__(self,experiment_name:str, dataset_name, model, sampler_func = None, train_com_method="ddp", is_2nd_order =True, epochs=100, batch_size =64, device=torch.device("cuda:0"), timestamp="",transform_train=None, transform_test=None, precondtioner=None ,recover = False):
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
        self.optimizer = torch.optim.SGD(params=model.parameters(),lr=0.001, momentum = 0.9) #torch.optim.Adam(model.parameters())
        #self.optimizer = torch.optim.Adam(model.parameters(),lr=0.0008)

        if is_2nd_order:
            if precondtioner is not None:
                self.preconditioner = precondtioner
            else:
                self.preconditioner = kfac.preconditioner.KFACPreconditioner(model=model)
            if train_com_method == "rpc":
                self.rpc_communicator:rpc_distributed.KFacRPCCommunicator = rpc_distributed.KFacRPCCommunicator(world_size=world_size, rank=rank,
                                                                                                                preconditioner=self.preconditioner, model=model,
                                                                                                                share_file_path=SHARE_FILES_DIR, timestamp=timestamp,
                                                                                                                log_dir = log_dir, device=device)
        else:
            self.preconditioner = None

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

    """"
    '''
    Departure from the original code
    '''
    
    def init_mischief(self,disconnect_ratio=0,max_sick_iter_ratio=0.2,max_disconnected_node_num = 2, possible_disconnect_node = None):
        max_disconnect_iter = int(len(self.data_manager.train_dataset) / self.batch_size / self.world_size * max_sick_iter_ratio)
        mischief.mischief_init(world_size=self.world_size, possible_disconnect_node=possible_disconnect_node,
                           max_disconnect_iter=max_disconnect_iter, disconnect_ratio=disconnect_ratio,
                            max_disconnected_node_num=max_disconnected_node_num,
                           ddp_trigger=True, factor_comm_trigger=True, inverse_comm_trigger=True)
        self.is_fault = True
        self.experiment_name_detail = f"mdn{max_disconnected_node_num}_dr{disconnect_ratio}_mdi{max_disconnect_iter}_ws{self.world_size}"
        if self.train_com_method == "rpc":
            mischief.recover_func = self.rpc_communicator.restart_sick_node
    """

    def train_and_test(self):
        writer_path = self.log_dir
        if self.experiment_name_detail is not None:
            writer_path = os.path.join(writer_path,self.experiment_name_detail)
        writer_name = os.path.join(writer_path,str(self.rank))
        self.writer = SummaryWriter(
            log_dir=writer_name)

        for i in range(0, self.epochs):
            self.train(epoch=i)
            self.test_all_top_1_and_top_n(epoch=i)
            self.save_checkpoint(epoch=i)

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
            self.simple_rpc_train(epoch=i)
            self.test_local_top(epoch=i, topk=(1, 3))
            self.save_checkpoint(epoch=i)

        self.writer.close()
        print(f"Rank {self.rank} : real fault rate {fault_simulator.fault_total_time / self.train_total_time}")
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
                self.optimizer.zero_grad()
                print(f"rank {self.rank} : {data.size()}")
                output = self.model(data)
                loss = self.loss_func(output, target)
                loss.backward()

                #if self.rank not in [0,6,9,15]:
                #    time.sleep(0.5)
                #time.sleep(delay_list_dict[0][self.rank])

                if self.preconditioner is not None:
                    self.preconditioner.step()
                self.optimizer.step()
                t.update()
        self.train_total_time += time.time() - start_time
        if self.writer is not None:
            self.writer.add_scalar('Loss/train', loss.item(), epoch)
            self.writer.add_scalar('Time/train',time.time() - start_time, epoch)

    def simple_rpc_train(self, epoch):
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
                rpc_distributed.global_communicator.update_self_t()

                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                
                with self.rpc_communicator.model_avg_rpc.local_model_store.lock:
                    output = self.model(data)
                    loss = self.loss_func(output, target)
                    loss.backward()
                    self.optimizer.step()
                self.rpc_communicator.model_avg_rpc.set_loss(loss.item())
                
                if self.preconditioner is not None:
                    self.preconditioner.step()

                #if batch_idx % 10 == 9:
                #    self.rpc_communicator.model_avg_rpc.process2_5()
                self.rpc_communicator.send_model_param()

                if rpc_distributed.global_communicator.current_t() % 200 == 0:
                    rpc_distributed.global_communicator.print_rpc_state()
                t.update()
            self.train_total_time += time.time() - start_time
            if self.writer is not None:
                self.writer.add_scalar('Iteration Variance',
                                       rpc_distributed.global_communicator.compute_iter_variance(),
                                       self.train_total_time)
                self.writer.add_scalar('Loss/train', loss.item(), epoch)

    def rpc_train(self, epoch):
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
                rpc_distributed.global_communicator.update_self_t()

                """
                if epoch > 0:
                    fault_simulator.update_fault_status()
                    if fault_simulator.is_fault():
                        time.sleep(fault_simulator.fault_over_time - time.time())
                    elif fault_simulator.recover_flg:
                        rpc_distributed.global_communicator.task_reassign_rpc.resurrection_declaration()
                        fault_simulator.recover_flg = False
                """
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

                self.rpc_communicator.model_avg_rpc.set_loss(loss.item())
                self.rpc_communicator.model_avg_rpc.broadcast_model()
                self.rpc_communicator.model_avg_rpc.avg_model_with_neighbors()

                if self.preconditioner is not None:
                    self.preconditioner.step()

                self.optimizer.step()
                #self.scheduler.step()

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
                """

                '''
                if self.writer is not None and batch_idx % 30 == 0:
                    process = psutil.Process(os.getpid())
                    self.writer.add_scalar('Memory', process.memory_info().rss / 1024**3, (epoch+1)*batch_idx)
                    allocated_memory = torch.cuda.memory_allocated(0)  # 0 表示 GPU 0
                    cached_memory = torch.cuda.memory_reserved(0)  # 0 表示 GPU 0
                    self.writer.add_scalar('Memory/GPU_Allocated', allocated_memory / 1024**3, (epoch+1)*batch_idx)
                    self.writer.add_scalar('Memory/GPU_Cached', cached_memory / 1024**3, (epoch+1)*batch_idx)
                '''

                t.update()
            self.train_total_time += time.time() - start_time
            if self.writer is not None:
                self.writer.add_scalar('Iteration Variance', rpc_distributed.global_communicator.compute_iter_variance(), self.train_total_time)
                self.writer.add_scalar('Loss/train', loss.item(), epoch)
                self.writer.add_scalar('Time/train',time.time() - start_time, epoch)

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

    def test_all_top_1_and_top_n(self, epoch, top_n=3):
        self.model.eval()
        
        # 初始化 Top-1 和 Top-N 精度的计数
        correct_top_1 = 0
        correct_top_n = 0
        total = 0

        with torch.no_grad():
            for data, target in self.data_manager.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                # Top-1 精度：获取预测的最大值的索引
                pred_top_1 = output.argmax(dim=1, keepdim=True)  # 获取最大值索引
                correct_top_1 += pred_top_1.eq(target.view_as(pred_top_1)).sum().item()
                
                # Top-N 精度：获取前 top_n 个预测的索引
                _, pred_top_n = output.topk(top_n, dim=1, largest=True, sorted=True)
                
                # 将 target 从 [batch_size] 形状扩展为 [batch_size, 1] 以便与 top_n 的预测比较
                target_expanded = target.view(-1, 1)

                # 检查前 top_n 个预测是否包含目标类别
                correct_top_n += pred_top_n.eq(target_expanded).sum().item()

                # 累加总样本数
                total += target.size(0)

        # 把 correct_top_1, correct_top_n 和 total 转换成 tensor 以便进行分布式计算
        correct_total_tensor = torch.tensor([correct_top_1, correct_top_n, total]).to(self.device)

        # 使用 dist.all_reduce 把所有节点的 correct_top_1, correct_top_n 和 total 累加到 rank 0 节点
        dist.all_reduce(correct_total_tensor)

        # 只在 rank 0 上计算最终的 Top-1 和 Top-N 精度并记录
        if self.writer is not None and self.rank == 0:  # 假设 self.rank 存储了当前进程的 rank
            correct_top_1_sum, correct_top_n_sum, total_sum = correct_total_tensor.unbind()
            top_1_accuracy = correct_top_1_sum.item() / total_sum.item()
            top_n_accuracy = correct_top_n_sum.item() / total_sum.item()
            
            # 记录 Top-1 和 Top-N 精度到 TensorBoard
            time_as_step = round(self.train_total_time * 1000)  # 使用训练迭代次数作为 x 轴
            self.writer.add_scalar('Top-1 Accuracy/test', top_1_accuracy, time_as_step)
            self.writer.add_scalar(f'Top-{top_n} Accuracy/test', top_n_accuracy, time_as_step)

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
        self.writer.add_scalar('Accuracy/test', accuracy, epoch)

    def test_local_top(self, epoch, topk=(1,)):  # 添加 topk 参数，支持多种 Top-N 精度
        self.model.eval()
        topk_correct = {k: 0 for k in topk}  # 初始化每个 Top-N 精度的正确计数
        total = 0
        with torch.no_grad():
            for data, target in self.data_manager.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # 获取前 topk 个类别及其对应的索引
                _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)  # pred 形状为 (batch_size, max(topk))
                pred = pred.t()  # 转置使 pred 的形状为 (max(topk), batch_size)
                
                # target shape: (batch_size), pred shape: (max(topk), batch_size)
                correct = pred.eq(target.view(1, -1).expand_as(pred))  # shape: (max(topk), batch_size)
                
                for k in topk:
                    topk_correct[k] += correct[:k].reshape(-1).float().sum(0).item()  # 计算 Top-k 精度的正确数
                
                total += target.size(0)

        # 计算每个 Top-k 精度
        topk_accuracies = {f'Top-{k} Accuracy': topk_correct[k] / total for k in topk}
        
        # 将 Top-N 精度输出到日志
        time_as_step = round(self.train_total_time * 1000)  # 使用训练time作为 x 轴
        for k, acc in topk_accuracies.items():
            self.writer.add_scalar(f'{k}/test', acc, time_as_step)

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

    def save_checkpoint(self, epoch):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'preconditioner': self.preconditioner.state_dict() if self.preconditioner is not None else None,
            'epoch': epoch,
            'train_total_time': self.train_total_time,
        }
        try:
            temp_path = self.checkpoint_file_path + ".temp"
            torch.save(state, temp_path)
            os.rename(temp_path,self.checkpoint_file_path)   
        except Exception as e:
            print(f"Save checkpoint error: {e} in rank {self.rank} at epoch {epoch} file path {self.checkpoint_file_path}")

    def get_total_training_time(self):
        return self.train_total_time