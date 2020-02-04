#!/bin/bash
#$ -cwd
#$ -l rt_G.small=1
#$ -l h_rt=05:00:00
#$ -N measure_alexnet
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

job_name=train_alexnet_1_1


python alexnet_measurements.py --model=alexnet --batchsize=1 --epochs=1 --gpu=0 --out="results/alexnet/"




