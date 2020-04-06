#!/bin/bash
#$ -cwd
#$ -l rt_F=4
#$ -l h_rt=05:00:00
#$ -N data_parallel_vgg_256_16
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

mpiexec -n 16 --hostfile $SGE_JOB_HOSTLIST --oversubscribe python train_imagenet.py  --model=vgg  --batchsize=256  --epochs=1  --out="results/vgg/256_16"








