#!/bin/bash
#$ -cwd
#$ -l rt_F=8
#$ -l h_rt=05:00:00
#$ -N alexnet_32
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

mpiexec -n 32 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_imagenet.py  --model=alexnet  --batchsize=512  --epochs=10  --out="results/alexnet/32"








