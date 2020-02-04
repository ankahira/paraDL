#!/bin/bash
#$ -cwd
#$ -l rt_G.small=1
#$ -l h_rt=01:00:00
#$ -N meausure_resnet
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

job_name=train_resnet_1_1


python resnet_measurements.py --model=resnet --batchsize=1 --epochs=1 --gpu=0 --out="results/resnet/resnet"


