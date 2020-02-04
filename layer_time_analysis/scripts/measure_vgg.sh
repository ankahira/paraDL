#!/bin/bash
#$ -cwd
#$ -l rt_G.small=1
#$ -l h_rt=01:00:00
#$ -N measure_vgg
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

python  vgg_measurements.py --model=vgg --batchsize=1 --epochs=1 --gpu=0 --out="results/vgg"


