#!/usr/bin/env bash


# Alexnet jobs


qsub -g gaa50004 ./scripts/alexnet_2_32.sh

qsub -g gaa50004 ./scripts/alexnet_2_64.sh

qsub -g gaa50004 ./scripts/alexnet_4_32.sh

qsub -g gaa50004 ./scripts/alexnet_4_64.sh


sleep 5

# Resnet Jobs


qsub -g gaa50004 ./scripts/resnet_2_32.sh

qsub -g gaa50004 ./scripts/resnet_2_64.sh

qsub -g gaa50004 ./scripts/resnet_4_32.sh

qsub -g gaa50004 ./scripts/resnet_4_64.sh


sleep 5

# VGG Jobs


qsub -g gaa50004 ./scripts/vgg_2_32.sh

qsub -g gaa50004 ./scripts/vgg_2_64.sh

qsub -g gaa50004 ./scripts/vgg_4_32.sh

qsub -g gaa50004 ./scripts/vgg_4_64.sh

