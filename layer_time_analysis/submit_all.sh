#!/usr/bin/env bash


# Alexnet jobs


qsub -g gaa50004 ./scripts/alexnet_1_1.sh

qsub -g gaa50004 ./scripts/alexnet_1_32.sh

qsub -g gaa50004 ./scripts/alexnet_1_64.sh

qsub -g gaa50004 ./scripts/alexnet_1_128.sh


sleep 5

# Resnet Jobs


qsub -g gaa50004 ./scripts/resnet_1_1.sh

qsub -g gaa50004 ./scripts/resnet_1_32.sh

qsub -g gaa50004 ./scripts/resnet_1_64.sh

qsub -g gaa50004 ./scripts/resnet_1_128.sh


sleep 5

# VGG Jobs


qsub -g gaa50004 ./scripts/vgg_1_1.sh

qsub -g gaa50004 ./scripts/vgg_1_32.sh

qsub -g gaa50004 ./scripts/vgg_1_64.sh

qsub -g gaa50004 ./scripts/vgg_1_128.sh

