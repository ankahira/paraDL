#!/usr/bin/env bash


qsub -g gaa50004 scripts/alexnet/alexnet_4.sh

qsub -g gaa50004 scripts/vgg/vgg_4.sh

qsub -g gaa50004 scripts/resnet/resnet_4.sh



