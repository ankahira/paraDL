#!/bin/bash
#$ -cwd
#$ -l rt_F=2
#$ -l h_rt=10:00:00
#$ -N vgg_8
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

python train_imagenet.py  --model=vgg  --batchsize=32  --epochs=100  --out="results/vgg/8"







