#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=40:00:00
#$ -N resnet_1
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

mpiexec -n 1 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_imagenet.py  --model=resnet  --batchsize=64  --epochs=10  --out="results/resnet/1"








