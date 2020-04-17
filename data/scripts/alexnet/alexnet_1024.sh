#!/bin/bash
#$ -cwd
#$ -l rt_F=256
#$ -l h_rt=00:30:00
#$ -N alexnet_1024
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

mpiexec -n 1024 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_imagenet.py  --model=alexnet  --batchsize=512  --epochs=1  --out="results/alexnet/1024"








