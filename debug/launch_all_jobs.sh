#!/usr/bin/env bash

python  train_imagenet_sequential.py --model=alexnet --batchsize=128 --epochs=100 --gpu=0 --out="results"

python  train_imagenet_sequential.py --model=vgg --batchsize=32 --epochs=100 --gpu=0 --out="results"



mpiexec -n 2 --hostfile $SGE_JOB_HOSTLIST --oversubscribe  python train_imagenet.py  --model=alexnet  --batchsize=512  --epochs=100  --out="results/alexnet/2"



