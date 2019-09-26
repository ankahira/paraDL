#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=05:00:00
#$ -N train_resnet_4_64
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

job_name=train_resnet_4_64



mpirun  -n 4  python train_imagenet.py  --model=resnet  --batchsize=64  --epochs=1  --out="results/resnet_4"  >> results/resnet_4/resnet_4_64



