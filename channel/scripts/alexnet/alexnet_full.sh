#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=10:00:00
#$ -N alexnet_full
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

mpiexec -n 4 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_imagenet_full.py  --model=alexnet  --batchsize=2048  --epochs=40  --out="results/alexnet/full"








