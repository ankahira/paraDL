#!/bin/bash
#$ -cwd
#$ -l rt_F=2
#$ -l h_rt=03:00:00
#$ -N resnet_8
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

mpiexec -n 8 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_imagenet.py  --model=resnet  --batchsize=128  --epochs=100  --out="results/resnet/8"







