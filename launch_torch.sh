#!/bin/bash

which python
export CUDA_VISIBLE_DEVICES=0
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

SHAREDFILE=/work/NBB/yu_mingzhe/kfac-pytorch/data/master_addr
OMPI_SIZE=$OMPI_COMM_WORLD_SIZE
OMPI_RANK=$OMPI_COMM_WORLD_RANK

if [ "$OMPI_COMM_WORLD_RANK" -eq 0 ]; then
    # 获取主节点的网络地址
    NET_ADDR=$(hostname -I | awk '{print $1}')
    # 将网络地址写入共享文件
    echo $NET_ADDR > $SHAREDFILE
else
    # 等待主节点地址文件被创建
    while [ ! -f $SHAREDFILE ]
    do
      sleep 1
    done
    # 读取网络地址
    NET_ADDR=$(cat $SHAREDFILE)
fi

torchrun --nproc_per_node=1 --nnodes=$OMPI_SIZE --node_rank=$OMPI_RANK \
 --rdzv-id=001 --rdzv-backend=c10d \
 --rdzv_endpoint=$NET_ADDR:29511 \
 /work/NBB/yu_mingzhe/kfac-pytorch/torch_cifar10_resnet.py \
 --epochs=200 \
 --warmup-epochs=5 \
 --seed=42 \
 --kfac-inv-method \
 --disconnect-node-number=4 \
 --connect-terms=9 \
 --disconnect-terms=1 \
 --times=0

sleep 3

torchrun --nproc_per_node=1 --nnodes=$OMPI_SIZE --node_rank=$OMPI_RANK \
 --rdzv-id=002 --rdzv-backend=c10d \
 --rdzv_endpoint=$NET_ADDR:29512 \
 /work/NBB/yu_mingzhe/kfac-pytorch/torch_cifar10_resnet.py \
 --epochs=200 \
 --warmup-epochs=5 \
 --seed=42 \
 --kfac-inv-method \
 --disconnect-node-number=4 \
 --connect-terms=8 \
 --disconnect-terms=2 \
 --times=0

sleep 3

torchrun --nproc_per_node=1 --nnodes=$OMPI_SIZE --node_rank=$OMPI_RANK \
 --rdzv-id=003 --rdzv-backend=c10d \
 --rdzv_endpoint=$NET_ADDR:29513 \
 /work/NBB/yu_mingzhe/kfac-pytorch/torch_cifar10_resnet.py \
 --epochs=200 \
 --warmup-epochs=5 \
 --seed=42 \
 --kfac-inv-method \
 --disconnect-node-number=4 \
 --connect-terms=6 \
 --disconnect-terms=4 \
 --times=0

sleep 3

torchrun --nproc_per_node=1 --nnodes=$OMPI_SIZE --node_rank=$OMPI_RANK \
 --rdzv-id=004 --rdzv-backend=c10d \
 --rdzv_endpoint=$NET_ADDR:29514 \
 /work/NBB/yu_mingzhe/kfac-pytorch/torch_cifar10_resnet.py \
 --epochs=200 \
 --warmup-epochs=5 \
 --seed=42 \
 --kfac-inv-method \
 --disconnect-node-number=4 \
 --connect-terms=4 \
 --disconnect-terms=6 \
 --times=0

sleep 3

torchrun --nproc_per_node=1 --nnodes=$OMPI_SIZE --node_rank=$OMPI_RANK \
 --rdzv-id=005 --rdzv-backend=c10d \
 --rdzv_endpoint=$NET_ADDR:29515 \
 /work/NBB/yu_mingzhe/kfac-pytorch/torch_cifar10_resnet.py \
 --epochs=200 \
 --warmup-epochs=5 \
 --seed=42 \
 --kfac-inv-method \
 --disconnect-node-number=4 \
 --connect-terms=2 \
 --disconnect-terms=8 \
 --times=0

sleep 3

torchrun --nproc_per_node=1 --nnodes=$OMPI_SIZE --node_rank=$OMPI_RANK \
 --rdzv-id=006 --rdzv-backend=c10d \
 --rdzv_endpoint=$NET_ADDR:29516 \
 /work/NBB/yu_mingzhe/kfac-pytorch/torch_cifar10_resnet.py \
 --epochs=200 \
 --warmup-epochs=5 \
 --seed=42 \
 --kfac-inv-method \
 --disconnect-node-number=4 \
 --connect-terms=0 \
 --disconnect-terms=10 \
 --times=0

sleep 3

# 如果是主节点，删除共享文件
if [ "$OMPI_COMM_WORLD_RANK" -eq 0 ]; then
    rm $SHAREDFILE
fi