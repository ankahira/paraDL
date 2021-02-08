#!/bin/bash
#$ -cwd
#$ -l rt_F=42
#$ -l h_rt=04:00:00
#$ -N cosmo_all
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y


source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate
source ~/.bash_profile

NUM_PROCESSES_PER_NODE=4
NUM_PROCESSES=128



echo $SGE_JOB_HOSTLIST
cat $SGE_JOB_HOSTLIST > ./$JOB_ID.$JOB_NAME.nodes.list
cp $SGE_JOB_HOSTLIST  ./$JOB_ID.$JOB_NAME.hostlist
sed -i '/g0206 /d' ./$JOB_ID.$JOB_NAME.nodes.list
sed -i '/g0504 /d' ./$JOB_ID.$JOB_NAME.nodes.list
sed -i '/g0592 /d' ./$JOB_ID.$JOB_NAME.nodes.list
sed -i '/g0614 /d' ./$JOB_ID.$JOB_NAME.nodes.list
sed -i '/g0855 /d' ./$JOB_ID.$JOB_NAME.nodes.list
sed -i '/g0643 /d' ./$JOB_ID.$JOB_NAME.nodes.list
sed -i '/g0273 /d' ./$JOB_ID.$JOB_NAME.nodes.list
sed -i '/g0329 /d' ./$JOB_ID.$JOB_NAME.nodes.list
sed -i '/g0275 /d' ./$JOB_ID.$JOB_NAME.nodes.list
sed -i '/g0418 /d' ./$JOB_ID.$JOB_NAME.nodes.list


cp -r /groups2/gaa50004/cosmoflow_data_256/ $SGE_LOCALDIR


mpiexec -n ${NUM_PROCESSES} --hostfile ./$JOB_ID.$JOB_NAME.nodes.list --oversubscribe  -map-by ppr:${NUM_PROCESSES_PER_NODE}:node python train_hybrid_cosmoflow.py  --batchsize=1  --epochs=20  --out="results_256/128"


sleep 1000

NUM_PROCESSES_PER_NODE=4
NUM_PROCESSES=256

mpiexec -n ${NUM_PROCESSES} --hostfile ./$JOB_ID.$JOB_NAME.nodes.list --oversubscribe  -map-by ppr:${NUM_PROCESSES_PER_NODE}:node python train_hybrid_cosmoflow.py  --batchsize=1  --epochs=20  --out="results_256/256"
#


