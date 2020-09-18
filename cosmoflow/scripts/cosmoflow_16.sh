#!/bin/bash
#$ -cwd
#$ -l rt_F=4
#$ -l h_rt=05:00:00
#$ -N cosmo_16
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y
#$ -p -400


source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

# mpiexec -n 16 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_hybrid_cosmoflow.py  --batchsize=1  --epochs=20  --out="results/16"

mpiexec -n 16 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_hybrid_cosmoflow_256.py  --batchsize=1  --epochs=20  --out="results_256/16"








