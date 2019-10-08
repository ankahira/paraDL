#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=10:00:00
#$ -N train_data_parallel
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

job_name=train_data_parallel



mpirun  -n 4  python train_imagenet.py  --model=alexnet  --batchsize=8  --epochs=10  --out="results"  >> results








