#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=00:20:00
#$ -N alexnet_2
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
conda activate

source ~/.bash_profile

source ~/.bash_profile

mpiexec -n 2 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_imagenet.py  --model=alexnet  --batchsize=512  --epochs=10  --out="results/alexnet/2"








