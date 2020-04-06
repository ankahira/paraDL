#!/bin/bash
#$ -cwd
#$ -l rt_F=2
#$ -l h_rt=05:00:00
#$ -N train_data_parallel_vgg_128_8
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

mpiexec -n 8 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_imagenet.py  --model=vgg  --batchsize=128  --epochs=1  --out="results/vgg/128_8"








