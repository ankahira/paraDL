#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=20:00:00
#$ -N vgg_full
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile


mpirun  -n 4  python train_imagenet_full.py  --model=vgg  --batchsize=128  --epochs=40  --out="results/vgg/full"








