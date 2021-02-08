#!/bin/bash
#$ -cwd
#$ -l rt_F=64
#$ -l h_rt=01:00:00
#$ -N vgg_256
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

mpiexec -n 256 --hostfile $SGE_JOB_HOSTLIST --oversubscribe python train_imagenet.py  --model=vgg  --batchsize=32  --epochs=100  --out="results/vgg/256"







