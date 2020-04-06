#!/usr/bin/env python

from __future__ import print_function
import argparse
import multiprocessing
import random
import gc

import numpy as np


import chainer.backends.cuda
from chainer import training
from chainer.training import extensions
from chainer.function_hooks import CupyMemoryProfileHook
from chainer.function_hooks import TimerHook

import cupy
import numpy

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
        gc.collect()
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
    # multiprocessing.set_start_method('forkserver')
    # p = multiprocessing.Process()
    # p.start()
    # p.join()

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

    # model = L.Classifier(models[args.model]())
    model = models[args.model]()
    model_0 = model.copy()
    model_1 = model.copy()
    model_2 = model.copy()
    model_3 = model.copy()

    model_0.to_gpu(0)
    model_1.to_gpu(1)
    model_2.to_gpu(2)
    model_3.to_gpu(3)



    chainer.backends.cuda.get_device_from_id(device).use()  # Make the GPU current
    model.to_gpu()

    # Split and distribute the dataset. Only worker 0 loads the whole dataset.
    # Datasets of worker 0 are evenly split and distributed to all workers.
    mean = np.load(MEAN_FILE)
    if comm.rank == 0:
        train = PreprocessedDataset(TRAIN, TRAINING_ROOT, mean, 226)
        val = PreprocessedDataset(VAL, VALIDATION_ROOT, mean, 226, False)
    else:
        train = None
        val = None
    # model inputs come from datasets, and each process takes different mini-batches
    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    val = chainermn.scatter_dataset(val, comm, shuffle=True)

    train_iter = chainer.iterators.MultithreadIterator(train, batch_size, n_threads=20)
    val_iter = chainer.iterators.MultithreadIterator(val, batch_size, n_threads=20, repeat=False)

    # Create a multi node optimizer from a standard Chainer optimizer.
    optimizer = chainermn.create_multi_node_optimizer(chainer.optimizers.Adam(), comm)
    optimizer.setup(model)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (epochs, 'epoch'), out)

    # Create a multi node evaluator from an evaluator.
    evaluator = extensions.Evaluator(val_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator, trigger=(1, 'epoch'))
    # Some display and output extensions are necessary only for one worker.
    # (Otherwise, there would just be repeated outputs.)
    if comm.rank == 0:
        # trainer.extend(extensions.DumpGraph('main/loss'))
        trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
        trainer.extend(extensions.observe_lr(), trigger=(1, 'epoch'))
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.ProgressBar(update_interval=10))

    # TODO : Figure out how to send this report to a file
    if comm.rank == 0:
        print("Starting training .....")

    trainer.run()


if __name__ == '__main__':
    main()