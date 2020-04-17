#!/usr/bin/env bash


qsub -g gaa50004 ./scripts/alexnet.sh


qsub -g gaa50004 ./scripts/resnet.sh


qsub -g gaa50004 ./scripts/vgg.sh

