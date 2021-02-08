#!/bin/bash
#$ -cwd
#$ -l rt_F=128
#$ -l h_rt=02:00:00
#$ -N vgg_512
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

mpiexec -n 512 --hostfile $SGE_JOB_HOSTLIST --oversubscribe python train_imagenet.py  --model=vgg  --batchsize=128  --epochs=10  --out="results/vgg/512"






