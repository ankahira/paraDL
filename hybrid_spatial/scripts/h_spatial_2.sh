#!/bin/bash
#$ -cwd
#$ -l rt_F=74
#$ -l h_rt=01:00:00
#$ -N h_spatial_all_2
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y

source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate

source ~/.bash_profile


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




NUM_PROCESSES_PER_NODE=4
NUM_PROCESSES=256
mpiexec -n ${NUM_PROCESSES} --hostfile ./$JOB_ID.$JOB_NAME.nodes.list --oversubscribe  -map-by ppr:${NUM_PROCESSES_PER_NODE}:node python train_imagenet.py  --model=vgg  --batchsize=32  --epochs=10  --out="results/vgg/256"
#sleep 1800


#
#NUM_PROCESSES_PER_NODE=4
#NUM_PROCESSES=512
#mpiexec -n ${NUM_PROCESSES} --hostfile ./$JOB_ID.$JOB_NAME.nodes.list --oversubscribe  -map-by ppr:${NUM_PROCESSES_PER_NODE}:node python train_imagenet.py  --model=vgg  --batchsize=32  --epochs=10  --out="results/vgg/512"
#
#sleep 1800
#
#NUM_PROCESSES_PER_NODE=4
#NUM_PROCESSES=512
#mpiexec -n ${NUM_PROCESSES} --hostfile ./$JOB_ID.$JOB_NAME.nodes.list --oversubscribe  -map-by ppr:${NUM_PROCESSES_PER_NODE}:node  python train_imagenet.py  --model=resnet152  --batchsize=64  --epochs=100  --out="results/resnet152/512"
#
#sleep 1800
#
#NUM_PROCESSES_PER_NODE=4
#NUM_PROCESSES=512
#mpiexec -n ${NUM_PROCESSES} --hostfile ./$JOB_ID.$JOB_NAME.nodes.list --oversubscribe  -map-by ppr:${NUM_PROCESSES_PER_NODE}:node python train_imagenet.py  --model=resnet50  --batchsize=64  --epochs=100  --out="results/resnet50/512"
