#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=15:00:00
#$ -N train_pipeline_parallel_alexnet_128_4
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

python train_imagenet.py  --model=alexnet  --batchsize=128  --epochs=1  --out="results/alexnet/128_4"








