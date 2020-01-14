#!/usr/bin/env python
import os
import sys
import numpy
import argparse
import random

import numpy as np
import cupy as cp
import chainer.backends.cuda
from chainer import training
from chainer.training import extensions
import chainermnx


import chainermn

# Local Imports
from models.alexnet import AlexNet

from models.vgg import VGG
from models.resnet50 import ResNet50
numpy.set_printoptions(threshold=sys.maxsize)

# Global
TRAIN = "/groups2/gaa50004/data/ILSVRC2012/train_256x256/10_image_train.txt"
VAL = "/groups2/gaa50004/data/ILSVRC2012/val_256x256/val.txt"
TRAINING_ROOT = "/groups2/gaa50004/data/ILSVRC2012/train_256x256/"
VALIDATION_ROOT = "/groups2/gaa50004/data/ILSVRC2012/val_256x256"
MEAN_FILE = "/groups2/gaa50004/data/ILSVRC2012/train_256x256/mean.npy"


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=False):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype(np.float32)
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label


def split_comm(comm):
    """Create a local communicator from the main communicator
    :arg: comm
    :return local comm
    """
    hs = comm.mpi_comm.allgather(os.uname()[1])
    host_list = []
    for h in hs:
        if h not in host_list:
            host_list.append(h)

    hosts = {k: v for v, k in enumerate(host_list)}

    small_comm = comm.split(hosts[os.uname()[1]], comm.intra_rank)
    return small_comm


def create_data_comm(comm):
    """Create a data communicator from the main communicator
    :arg: comm
    :return local comm
    """
    hs = comm.mpi_comm.allgather(os.uname()[1])
    host_list = []
    for h in hs:
        if h not in host_list:
            host_list.append(h)

    hosts = {k: v for v, k in enumerate(host_list)}

    small_comm = comm.split(hosts[os.uname()[1]], comm.intra_rank)
    return small_comm


def main():
    chainer.disable_experimental_feature_warning = True
    models = {
        'alexnet': AlexNet,
        'resnet': ResNet50,
        'vgg': VGG,
    }

    parser = argparse.ArgumentParser(description='Train ImageNet From Scratch')
    parser.add_argument('--model', '-M', choices=models.keys(), default='AlexNet', help='Convnet model')
    parser.add_argument('--batchsize', '-B', type=int, default=32, help='Learning minibatch size')
    parser.add_argument('--epochs', '-E', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--out', '-o', default='results', help='Output directory')
    args = parser.parse_args()

    batch_size = args.batchsize
    epochs = args.epochs
    out = args.out

    # Prepare ChainerMN communicator.
    comm = chainermn.create_communicator("pure_nccl")
    local_comm = split_comm(comm)

    device = comm.intra_rank

    if comm.rank == 0:
        print('==========================================')
        print('Num of GPUs : {}'.format(comm.size))
        print('Model :  {}'.format(args.model))
        print('Minibatch-size: {}'.format(batch_size))
        print('Epochs: {}'.format(args.epochs))
        print('==========================================')

    # model = L.Classifier(models[args.model]())
    model = models[args.model](local_comm)
    # chainer.backends.cuda.get_device_from_id(device).use()  # Make the GPU current
    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu(device)
    mean = np.load(MEAN_FILE)

    # All ranks load the data


    # Split and distribute the dataset. Only worker 0 loads the whole dataset.
    # Datasets of worker 0 are evenly split and distributed to all workers.
    # This is the tricky part
    mean = np.load(MEAN_FILE)

    if comm.rank == 0:
        train = PreprocessedDataset(TRAIN, TRAINING_ROOT, mean, 226)
        val = PreprocessedDataset(VAL, VALIDATION_ROOT, mean, 226, False)
    else:
        train = None
        val = None
    # model inputs come from datasets, and each process takes different mini-batches
    train_iter = chainermn.scatter_dataset(train, comm, shuffle=True)

    train_iter = chainermn.iterators.create_multi_node_iterator(
        chainer.iterators.MultithreadIterator(train_iter, args.batchsize, n_threads=40, shuffle=True), comm)



    val_iter = chainermn.scatter_dataset(val, comm, shuffle=True)


    # Create a multi node optimizer from a standard Chainer optimizer.
    # For Hybrid, we modify the multinode optimizer such that it takes two communicators. The first main communicator
    # is for global all reduce and the second one is to do a local all reduce
    optimizer = chainermnx.create_multi_node_optimizer(chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9), comm)

    optimizer.setup(model)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (epochs, 'epoch'), out)

    val_interval = (100, 'epoch')
    log_interval = (1, 'epoch')

    # Create a multi node evaluator from an evaluator.
    evaluator = extensions.Evaluator(val_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator, trigger=val_interval)
    if comm.rank == 0:
        trainer.extend(extensions.DumpGraph('main/loss'))
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'], 'epoch', filename='loss.png'))
        trainer.extend(extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'], 'epoch', filename='accuracy.png'))
        trainer.extend(extensions.ProgressBar())

    if comm.rank == 0:
        print("Starting training .....")

    trainer.run()
    # serializers.save_npz('spatial_model_rank_{}.npz'.format(comm.rank), model)


if __name__ == '__main__':
    main()
