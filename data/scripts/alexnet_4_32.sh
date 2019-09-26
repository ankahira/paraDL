#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=05:00:00
#$ -N train_alexnet_4_32
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

job_name=train_alexnet_4_32



mpirun  -n 4  python train_imagenet.py  --model=alexnet  --batchsize=32  --epochs=1  --out="results/alexnet_4"  >> results/alexnet_4/alexnet_4_32








