#!/bin/bash
#$ -cwd
#$ -l rt_F=128
#$ -l h_rt=01:00:00
#$ -N alexnet_512
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

mpiexec -n 512 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_imagenet.py  --model=alexnet  --batchsize=512  --epochs=100  --out="results/alexnet/512"








