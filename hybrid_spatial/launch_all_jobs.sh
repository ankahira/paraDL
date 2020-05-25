#!/usr/bin/env bash

#Hybrid parallelism jobs

# Alexnet

#qsub -g gaa50004 scripts/alexnet/alexnet_8.sh
#qsub -g gaa50004 scripts/alexnet/alexnet_16.sh
#qsub -g gaa50004 scripts/alexnet/alexnet_32.sh
#qsub -g gaa50004 scripts/alexnet/alexnet_64.sh

#qsub -g gaa50004 scripts/alexnet/alexnet_128.sh
#qsub -g gaa50004 scripts/alexnet/alexnet_256.sh
#qsub -g gaa50004 scripts/alexnet/alexnet_512.sh
#qsub -g gaa50004 scripts/alexnet/alexnet_1024.sh

## VGG

#qsub -g gaa50004 scripts/vgg/vgg_8.sh
#qsub -g gaa50004 scripts/vgg/vgg_16.sh
#qsub -g gaa50004 scripts/vgg/vgg_32.sh
#qsub -g gaa50004 scripts/vgg/vgg_64.sh

#qsub -g gaa50004 scripts/vgg/vgg_128.sh
#qsub -g gaa50004 scripts/vgg/vgg_256.sh
#qsub -g gaa50004 scripts/vgg/vgg_512.sh
#qsub -g gaa50004 scripts/vgg/vgg_1024.sh



# ## Resnet

# qsub -g gaa50004 scripts/resnet/resnet_8.sh
# sleep 200
# qsub -g gaa50004 scripts/resnet/resnet_16.sh
# sleep 200
# qsub -g gaa50004 scripts/resnet/resnet_32.sh
# sleep 200
# qsub -g gaa50004 scripts/resnet/resnet_64.sh
# sleep 200
# qsub -g gaa50004 scripts/resnet/resnet_128.sh
# sleep 200
# qsub -g gaa50004 scripts/resnet/resnet_256.sh
# sleep 300
# qsub -g gaa50004 scripts/resnet/resnet_512.sh
# sleep 700
# qsub -g gaa50004 scripts/resnet/resnet_1024.sh

## Resnet

# conda activate
# module load cuda/10.0/10.0.130 cudnn/7.6/7.6.4 nccl/2.4/2.4.8-1 openmpi/2.1.6 
# qsub -g gaa50004 no_comm_scripts/resnet/resnet_8.sh
# sleep 200
# qsub -g gaa50004 no_comm_scripts/resnet/resnet_16.sh
# sleep 200
# qsub -g gaa50004 no_comm_scripts/resnet/resnet_32.sh
# sleep 200
# qsub -g gaa50004 no_comm_scripts/resnet/resnet_64.sh
# sleep 200
# qsub -g gaa50004 no_comm_scripts/resnet/resnet_128.sh
# sleep 200





