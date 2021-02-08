#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=10:00:00
#$ -N cosmoflow_4
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile

cp -r /groups2/gaa50004/cosmoflow_data_256/ $SGE_LOCALDIR

mpiexec -n 4  python train_hybrid_cosmoflow.py  --batchsize=1  --epochs=20  --out="results_256/4"







