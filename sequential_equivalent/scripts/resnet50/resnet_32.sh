#!/bin/bash
#$ -cwd
#$ -l rt_F=8
#$ -l h_rt=01:00:00
#$ -N resnet50_32
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

python train_imagenet.py  --model=resnet50  --batchsize=64  --epochs=100  --out="results/resnet50/32"







