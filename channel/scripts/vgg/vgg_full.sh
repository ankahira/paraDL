#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=10:00:00
#$ -N vgg_full
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile


mpirun  -n 3  python train_imagenet.py  --model=vgg  --batchsize=4  --epochs=50  --out="results/vgg/full"








