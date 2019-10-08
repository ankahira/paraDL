#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=15:00:00
#$ -N train_pipeline_parallel
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

job_name=train_pipeline_parallel


python train_imagenet.py  --model=alexnet  --batchsize=8  --epochs=1  --out="results"





