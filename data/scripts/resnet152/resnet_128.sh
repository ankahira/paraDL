#!/bin/bash
#$ -cwd
#$ -l rt_F=64
#$ -l h_rt=03:00:00
#$ -N resnet_128
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile


NUM_PROCESSES_PER_NODE=4
NUM_PROCESSES=128


mpiexec -n ${NUM_PROCESSES} -map-by ppr:${NUM_PROCESSES_PER_NODE}:node  python train_imagenet.py  --model=resnet152  --batchsize=64  --epochs=100  --out="results/resnet152/128"








