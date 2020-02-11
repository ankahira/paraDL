#!/bin/bash
#$ -cwd
#$ -l rt_F=8
#$ -l h_rt=05:00:00
#$ -N resnet_256_32
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

mpiexec -n 32 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_imagenet.py  --model=resnet  --batchsize=256  --epochs=1  --out="results/resnet/256_32"








