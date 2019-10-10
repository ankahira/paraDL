#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=02:00:00
#$ -N alexnet_2048_4
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

mpiexec -n 4 python train_imagenet.py  --model=alexnet  --batchsize=2048  --epochs=1  --out="results/alexnet/2048_4"








