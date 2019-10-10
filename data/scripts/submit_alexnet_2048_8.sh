#!/bin/bash
#$ -cwd
#$ -l rt_F=2
#$ -l h_rt=02:00:00
#$ -N alexnet_2048_8
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

mpiexec -n 8 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_imagenet.py  --model=alexnet  --batchsize=2048  --epochs=1  --out="results/alexnet/2048_8"








