#!/bin/bash

BATCH=$1
EPOCH_Q=$2
EPOCH_S=$3


cd sequential
python train_imagenet.py  --model=alexnet  --batchsize=$BATCH  --epochs=$EPOCH_Q  --out="results/alexnet/4"

cd ../spatial

mpirun  -n 4  python train_imagenet.py  --model=alexnet  --batchsize=$BATCH  --epochs=$EPOCH_S  --out="results/alexnet/4"

cd ..

diff spatial/spatial_output.txt sequential/sequential_output.txt
