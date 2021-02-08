#!/bin/bash
#$ -cwd
#$ -l rt_F=2
#$ -l h_rt=02:00:00
#$ -N alexnet_8
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

mpiexec -n 8 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_imagenet_filter.py  --model=filter_simple  --batchsize=10  --epochs=1000  --out="results/filter_simple/8"







