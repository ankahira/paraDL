#!/usr/bin/env python

from __future__ import print_function
import argparse
import multiprocessing
import random
import os

import numpy as np
import cupy as cp
import shutil

import chainer.backends.cuda
from chainer import training
from chainer.training import extensions
from chainer.function_hooks import TimerHook, CupyMemoryProfileHook
from datetime import datetime

import chainermn
import chainermnx

import chainer.links as L

# Local Imports
from models.alexnet import AlexNet
from models.vgg import VGG
from models.resnet50 import ResNet50


# Global Variables
TRAIN = "/groups2/gaa50004/data/ILSVRC2012/train_256x256/train.txt"
VAL = "/groups2/gaa50004/data/ILSVRC2012/val_256x256/val.txt"
TRAINING_ROOT = "/groups2/gaa50004/data/ILSVRC2012/train_256x256/"
VALIDATION_ROOT = "/groups2/gaa50004/data/ILSVRC2012/val_256x256"
MEAN_FILE = "/groups2/gaa50004/data/ILSVRC2012/train_256x256/mean.npy"


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
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


def create_local_comm(comm):
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

    local_comm = comm.split(hosts[os.uname()[1]], comm.intra_rank)
    return local_comm


def create_data_comm(comm):
    """Create a data communicator from the main communicator

    :arg: comm
    :return data comm
    """
    colour = comm.rank % 4

    # if comm.rank % 4 == 0:
    #     colour = 0
    # else:
    #     colour = 1
    data_comm = comm.split(colour, comm.rank)
    return data_comm


def main():
    # These two lines help with memory. If they are not included training runs out of memory.
    # Use them till you the real reason why its running out of memory

    # pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    # cp.cuda.set_allocator(pool.malloc)
    chainer.disable_experimental_feature_warning = True

    models = {
        'alexnet': AlexNet,
        'resnet': ResNet50,
        'vgg': VGG,
    }

    parser = argparse.ArgumentParser(description='Train ImageNet From Scratch')
    parser.add_argument('--model', '-M', choices=models.keys(), default='AlexNet', help='Convnet model')
    parser.add_argument('--batchsize', '-B', type=int, default=32, help='Learning minibatch size')
    parser.add_argument('--epochs', '-E', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--out', '-o', default='results', help='Output directory')
    args = parser.parse_args()

    batch_size = args.batchsize
    epochs = args.epochs
    out = args.out

    # Start method of multiprocessing module need to be changed if we are using InfiniBand and MultiprocessIterator.
    multiprocessing.set_start_method('forkserver')
    p = multiprocessing.Process()
    p.start()
    p.join()

    # Directories are created later by the reporter.

    try:
        shutil.rmtree(out)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    # Create new output dirs
    try:
        os.makedirs(out)
    except OSError:
        pass

    # Prepare ChainerMN communicator.
    comm = chainermnx.create_communicator("filter_nccl", out)
    local_comm = create_local_comm(comm)
    data_comm = create_data_comm(comm)
    device = comm.intra_rank

    if comm.rank == 0:
        print('==========================================')
        print('Num of GPUs : {}'.format(comm.size))
        print('Model :  {}'.format(args.model))
        print('Minibatch-size: {}'.format(batch_size))
        print('Epochs: {}'.format(args.epochs))
        print('==========================================')

    model = L.Classifier(models[args.model](local_comm))

    chainer.backends.cuda.get_device_from_id(device).use()  # Make the GPU current
    model.to_gpu()
    comm._init_comms()
    local_comm._init_comms()
    data_comm._init_comms()

    # Split and distribute the dataset. Only worker 0 loads the whole dataset.
    # Datasets of worker 0 are evenly split and distributed to all workers.
    mean = np.load(MEAN_FILE)

    if local_comm.rank == 0:
        if data_comm.rank == 0:
            train = PreprocessedDataset(TRAIN, TRAINING_ROOT, mean, 224)
            val = PreprocessedDataset(VAL, VALIDATION_ROOT, mean, 224, False)
        else:
            train = None
            val = None
        train = chainermn.scatter_dataset(train, data_comm, shuffle=True)
        val = chainermn.scatter_dataset(val, data_comm, shuffle=True)
    else:
        train = PreprocessedDataset(TRAIN, TRAINING_ROOT, mean, 224)
        val = PreprocessedDataset(VAL, VALIDATION_ROOT, mean, 224, False)
        train = chainermn.datasets.create_empty_dataset(train)
        val = chainermn.datasets.create_empty_dataset(val)
    train_iter = chainermn.iterators.create_multi_node_iterator(
        chainer.iterators.MultithreadIterator(train, args.batchsize, n_threads=20, shuffle=True), local_comm)
    val_iter = chainermn.iterators.create_multi_node_iterator(
        chainer.iterators.MultithreadIterator(val, args.batchsize, repeat=False, shuffle=False, n_threads=20),
        local_comm)

    # optimizer = chainer.optimizers.Adam()
    # We need a multinode optimiser so that we can perform gradient allreduce like for data parallelism
    # Using chainmermnx in order to log the allreduce times
    # optimizer = chainermnx.create_multi_node_optimizer(chainer.optimizers.Adam(), data_comm)
    optimizer = chainermnx.create_multi_node_optimizer(chainer.optimizers.Adam(), data_comm, out)
    optimizer.setup(model)

    #TODO
    # Remember to change this updater to the stardard updater not chainermnx
    # You put this in oder to measure compute and data load time
    updater = chainermnx.training.StandardUpdater(train_iter, optimizer, comm, out=out, device=device)
    trainer = training.Trainer(updater, (epochs, 'epoch'), out)

    val_interval = (1, 'epoch')
    log_interval = (1, 'iteration')

    # Create an evaluator
    evaluator = extensions.Evaluator(val_iter, model, device=device)
    trainer.extend(evaluator, trigger=val_interval)

    # Some display and output extensions are necessary only for one worker.
    filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log"

    # Some display and output extensions are necessary only for one worker.
    if comm.rank == 0:
        # trainer.extend(extensions.DumpGraph('main/loss'))
        trainer.extend(extensions.LogReport(trigger=log_interval, filename=filename))
        trainer.extend(extensions.observe_lr(), trigger=(1, 'epoch'))
        trainer.extend(extensions.ProgressBar(update_interval=10))
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'], 'epoch', filename='loss.png'))
        trainer.extend(extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'], 'epoch', filename='accuracy.png'))
        trainer.extend(extensions.ProgressBar(update_interval=10))

    if comm.rank == 0:
        print("Starting training .....")

    trainer.run()


if __name__ == '__main__':
    main()
