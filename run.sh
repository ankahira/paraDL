#!/bin/bash

BATCH=$1
EPOCH_SQ=$2
EPOCH_SP=$3


( cd sequential && python train_imagenet.py  --model=alexnet  --batchsize=$BATCH  --epochs=$EPOCH_SQ  --out="results/alexnet/debug" )


( cd ../spatial && mpirun  -n 4  python train_imagenet.py  --model=alexnet  --batchsize=$BATCH  --epochs=$EPOCH_SP  --out="results/alexnet/debug")


python check_weights.py


