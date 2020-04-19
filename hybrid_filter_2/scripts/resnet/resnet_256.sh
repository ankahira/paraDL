#!/bin/bash
#$ -cwd
#$ -l rt_F=128
#$ -l h_rt=01:00:00
#$ -N resnet_256
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

mpiexec -n 256 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_imagenet.py  --model=resnet  --batchsize=64  --epochs=1  --out="results/resnet/256"








