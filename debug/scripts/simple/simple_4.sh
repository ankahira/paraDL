#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=00:20:00
#$ -N alexnet_4
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

mpiexec -n 4 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_imagenet.py  --model=simple  --batchsize=512  --epochs=20  --out="results/simple/4"









