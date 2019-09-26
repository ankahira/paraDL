#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=05:00:00
#$ -N train_alexnet_2_64
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

job_name=train_alexnet_2_64



mpirun  -n 2  python train_imagenet.py  --model=alexnet  --batchsize=64  --epochs=1  --out="results/alexnet_2"  >> results/alexnet_2/alexnet_2_64



