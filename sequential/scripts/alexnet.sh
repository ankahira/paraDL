#!/bin/bash
#$ -cwd
#$ -l rt_G.small=1
#$ -l h_rt=40:00:00
#$ -N train_alexnet
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

job_name=train_alexnet_1_1


python train_imagenet.py --model=alexnet --batchsize=128 --epochs=10 --gpu=0 --out="results/alexnet/"




