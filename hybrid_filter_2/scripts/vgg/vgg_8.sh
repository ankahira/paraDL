#!/bin/bash
#$ -cwd
#$ -l rt_F=4
#$ -l h_rt=03:00:00
#$ -N vgg_8
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

mpiexec -n 8 --hostfile $SGE_JOB_HOSTLIST --oversubscribe python train_imagenet.py  --model=vgg  --batchsize=32  --epochs=1  --out="results/vgg/8"







