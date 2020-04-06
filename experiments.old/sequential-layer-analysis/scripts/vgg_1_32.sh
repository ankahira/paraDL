#!/bin/bash
#$ -cwd
#$ -l rt_G.small=1
#$ -l h_rt=05:00:00
#$ -N train_vgg_1_32
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

job_name=train_vgg_1_32

python  train_imagenet.py --model=vgg --batchsize=64 --epochs=1 --gpu=0 --out="results/vgg/vgg_1_64" >> results/vgg/vgg_1_64


