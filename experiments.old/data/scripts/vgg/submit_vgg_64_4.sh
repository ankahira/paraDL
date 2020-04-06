#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=05:00:00
#$ -N train_data_vgg_parallel_64_4
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

mpirun  -n 4  python train_imagenet.py  --model=vgg  --batchsize=64  --epochs=1  --out="results/vgg/64_4"








