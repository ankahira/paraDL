#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=30:00:00
#$ -N train_channel_parallel_vgg
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

mpirun  -n 4  python train_imagenet.py  --model=vgg  --batchsize=32  --epochs=1  --out="results/vgg"








