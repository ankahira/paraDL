#!/usr/bin/env bash

#Hybrid parallelism jobs


qsub -g gaa50004 scripts/resnet/resnet_8.sh
sleep 60
qsub -g gaa50004 scripts/resnet/resnet_16.sh
sleep 60
qsub -g gaa50004 scripts/resnet/resnet_32.sh
sleep 60
qsub -g gaa50004 scripts/resnet/resnet_64.sh
sleep 60
qsub -g gaa50004 scripts/resnet/resnet_128.sh
sleep 60
qsub -g gaa50004 scripts/resnet/resnet_256.sh
sleep 60
qsub -g gaa50004 scripts/resnet/resnet_512.sh
sleep 60
qsub -g gaa50004 scripts/resnet/resnet_1024.sh







