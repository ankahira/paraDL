#!/bin/bash
#$ -cwd
#$ -l rt_F=16
#$ -l h_rt=02:00:00
#$ -N cosmo_64
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

mpiexec -n 64 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_hybrid_cosmoflow.py  --batchsize=1  --epochs=5  --out="results/hybrid/64"








