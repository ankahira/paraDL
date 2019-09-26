#!/usr/bin/env bash


python train_alexnet.py
    ../datasets/imagenet/train.txt
    ../datasets/imagenet/val.txt
    --batchsize 64
    --epoch 10
    --gpu 1
    --mean datasets/imagenet/mean.npy
    --out results/alexnet
    --root /gpfs/projects/bsc19/bsc19992/datasets/imagenet/train
    --val_batchsize 32
