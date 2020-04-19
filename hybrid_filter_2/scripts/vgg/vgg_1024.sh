#!/bin/bash
#$ -cwd
#$ -l rt_F=512
#$ -l h_rt=01:00:00
#$ -N vgg_1024
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

mpiexec -n 1024 --hostfile $SGE_JOB_HOSTLIST --oversubscribe python train_imagenet.py  --model=vgg  --batchsize=32  --epochs=1  --out="results/vgg/1024"







