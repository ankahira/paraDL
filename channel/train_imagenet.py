#!/usr/bin/env python

from __future__ import print_function
import argparse
import multiprocessing
import random

import numpy as np


import chainer.backends.cuda
from chainer import training
from chainer.training import extensions

from chainer.function_hooks import CupyMemoryProfileHook



import chainermn

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


def main():
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

    # Prepare ChainerMN communicator.
    comm = chainermn.create_communicator("pure_nccl")
    device = comm.intra_rank

    if comm.rank == 0:
        print('==========================================')
        print('Num of GPUs : {}'.format(comm.size))
        print('Model :  {}'.format(args.model))
        print('Minibatch-size: {}'.format(batch_size))
        print('Epochs: {}'.format(args.epochs))
        print('==========================================')

    model = models[args.model](comm)

    chainer.backends.cuda.get_device_from_id(device).use()  # Make the GPU current
    model.to_gpu()

    # Split and distribute the dataset. Only worker 0 lo1898687ads the whole dataset.
    # Datasets of worker 0 are evenly split and distributed to all workers.
    mean = np.load(MEAN_FILE)

    # All ranks load the data
    train = PreprocessedDataset(TRAIN, TRAINING_ROOT, mean, 226)
    val = PreprocessedDataset(VAL, VALIDATION_ROOT, mean, 226, False)

    # Create a multinode iterator such that each rank gets the same batch
    if comm.rank != 0:
        train = chainermn.datasets.create_empty_dataset(train)
        val = chainermn.datasets.create_empty_dataset(val)
    # Same dataset in all nodes
    train_iter = chainermn.iterators.create_multi_node_iterator(
        chainer.iterators.MultithreadIterator(train, args.batchsize,  n_threads=40), comm)
    val_iter = chainermn.iterators.create_multi_node_iterator(
        chainer.iterators.MultithreadIterator(val, args.batchsize, repeat=False, shuffle=False,  n_threads=40), comm)

    # We dont use a multinode optimizer here as we dont do all reduce on final weights
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (epochs, 'iteration'), out)

    val_interval = (1, 'epoch')
    log_interval = (1, 'epoch')

    # Create an evaluator
    evaluator = extensions.Evaluator(val_iter, model, device=device)
    # Since I need to measure timer per epoch, I avoid evaluation and just train the model
    # By setting the evaluation epoch high, this will not be triggered when i am running few epochs
    trainer.extend(evaluator, trigger=val_interval)

    # Some display and output extensions are necessary only for one worker.
    if comm.rank == 0:
        trainer.extend(extensions.DumpGraph('main/loss'))
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=(1, 'epoch'))
        trainer.extend(extensions.ProgressBar(update_interval=10))
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'], 'epoch', filename='loss.png'))
        trainer.extend(extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'], 'epoch', filename='accuracy.png'))
        trainer.extend(extensions.ProgressBar())

    # TODO : Figure out how to send this report to a file

    if comm.rank == 0:
        print("Starting training .....")

    hook = CupyMemoryProfileHook()
    with hook:
        trainer.run()

    if comm.rank == 0:
        hook.print_report()


if __name__ == '__main__':
    main()
