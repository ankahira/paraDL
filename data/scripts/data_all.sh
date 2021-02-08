#!/bin/bash
#$ -cwd
#$ -l rt_F=140
#$ -l h_rt=05:00:00
#$ -N data_all
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y


source /etc/profile.d/modules.sh
source ~/.bash_profile
conda activate
source ~/.bash_profile

cp -r /groups2/gaa50004/data/ILSVRC2012/pytorch/train/  $SGE_LOCALDIR

/groups2/gaa50004/data/ILSVRC2012/pytorch/train/


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


# ---------------------------ResNet152 128----------------------------------------------------------------------------------------------#

#
#NUM_PROCESSES_PER_NODE=4
#NUM_PROCESSES=128
#
#mpiexec -n ${NUM_PROCESSES} --hostfile ./$JOB_ID.$JOB_NAME.nodes.list --oversubscribe  -map-by ppr:${NUM_PROCESSES_PER_NODE}:node  python train_imagenet.py  --model=resnet152  --batchsize=64  --epochs=100  --out="results/resnet152/128"
#
#sleep 1600

# ---------------------------ResNet152 256----------------------------------------------------------------------------------------------#

NUM_PROCESSES_PER_NODE=4
NUM_PROCESSES=256

mpiexec -n ${NUM_PROCESSES} --hostfile ./$JOB_ID.$JOB_NAME.nodes.list --oversubscribe  -map-by ppr:${NUM_PROCESSES_PER_NODE}:node  python train_imagenet.py  --model=resnet152  --batchsize=64  --epochs=100  --out="results/resnet152/256"



# ---------------------------ResNet152 512----------------------------------------------------------------------------------------------#
#NUM_PROCESSES_PER_NODE=4
#NUM_PROCESSES=512
#
#mpiexec -n ${NUM_PROCESSES} --hostfile ./$JOB_ID.$JOB_NAME.nodes.list --oversubscribe  -map-by ppr:${NUM_PROCESSES_PER_NODE}:node  python train_imagenet.py  --model=resnet152  --batchsize=64  --epochs=100  --out="results/resnet152/512"