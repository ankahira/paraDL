#!/bin/bash
#$ -cwd
#$ -l rt_F=8
#$ -l h_rt=10:00:00
#$ -N resnet_full
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

mpiexec -n 32 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_imagenet_full.py  --model=resnet  --batchsize=64  --epochs=70  --out="results/resnet/full"








