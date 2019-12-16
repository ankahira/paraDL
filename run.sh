#!/bin/bash

BATCH=$1
EPOCH_SQ=$2
EPOCH_SP=$3

#
# ( cd sequential && python train_imagenet.py  --model=alexnet  --batchsize=$BATCH  --epochs=$EPOCH_SQ  --out="results/alexnet/debug" )


( cd spatial && mpirun  -n 4  python train_imagenet.py  --model=alexnet  --batchsize=$BATCH  --epochs=$EPOCH_SP  --out="results/alexnet/debug")

# echo "Diff forward prop outputs"

# diff sequential/sequential_forward_prop.txt spatial/spatial_forward_prop.txt

#python verification.py


