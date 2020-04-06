#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=40:00:00
#$ -N alexnet_3
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

mpirun  -n 3  python train_imagenet.py  --model=alexnet  --batchsize=512  --epochs=1  --out="results/alexnet/3"








