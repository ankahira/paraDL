#!/bin/bash
#$ -cwd
#$ -l rt_F=64
#$ -l h_rt=02:00:00
#$ -N alexnet_256
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

mpiexec -n 256 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_imagenet.py  --model=alexnet  --batchsize=2048  --epochs=10  --out="results/alexnet/256"








