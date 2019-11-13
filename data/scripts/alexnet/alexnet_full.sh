#!/bin/bash
#$ -cwd
#$ -l rt_F=4
#$ -l h_rt=40:00:00
#$ -N alexnet_full
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
conda activate

source ~/.bash_profile

source ~/.bash_profile

mpiexec -n 16 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_imagenet.py  --model=alexnet  --batchsize=1024  --epochs=70  --out="results/alexnet/1"








