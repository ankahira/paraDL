#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=30:00:00
#$ -N train_channel_parallel_alexnet_64
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

mpirun  -n 4  python train_imagenet.py  --model=alexnet  --batchsize=256  --epochs=1  --out="results/alexnet/256"








