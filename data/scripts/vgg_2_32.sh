#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=05:00:00
#$ -N train_vgg_2_32
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

job_name=train_vgg_2_32



mpirun  -n 2  python train_imagenet.py  --model=vgg  --batchsize=32  --epochs=1  --out="results/vgg_2"  >> results/vgg_2/vgg_2_32



