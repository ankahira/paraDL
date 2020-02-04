#!/bin/bash
#$ -cwd
#$ -l rt_F=512
#$ -l h_rt=02:00:00
#$ -N resnet_2048
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

mpiexec -n 2048 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_imagenet.py  --model=resnet  --batchsize=64  --epochs=10  --out="results/resnet/2048"








