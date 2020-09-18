#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=01:00:00
#$ -N resnet_4
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

module load cuda/10.0/10.0.130 cudnn/7.6/7.6.4 nccl/2.4/2.4.8-1 openmpi/2.1.6


mpiexec -n 4 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_imagenet.py  --model=resnet50  --batchsize=64  --epochs=100  --out="results/resnet/4"

mpiexec -n 4 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_imagenet.py  --model=resnet101  --batchsize=128  --epochs=100  --out="results/resnet/4"


mpiexec -n 4 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_imagenet.py  --model=resnet152  --batchsize=128  --epochs=100  --out="results/resnet/4"










