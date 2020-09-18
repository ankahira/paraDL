#!/usr/bin/env python

from __future__ import print_function
import argparse
import random
import numpy as np
import multiprocessing
import shutil

import chainer.backends.cuda
from chainer import training
from chainer.training import extensions
import chainermn
import chainermnx
import chainer.links as L
from datetime import datetime


import matplotlib

# Local Imports
from models.alexnet import AlexNet
from models.vgg import VGG
from models.resnet50 import ResNet50, ResNet101, ResNet152
# from models.resnet import ResNet

matplotlib.use('Agg')

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
        'resnet50': ResNet50,
        'resnet101': ResNet101,
        'resnet152': ResNet152,
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

    #Start method of multiprocessing module need to be changed if we are using InfiniBand and MultiprocessIterator.
    multiprocessing.set_start_method('forkserver')
    p = multiprocessing.Process()
    p.start()
    p.join()
    # Prepare ChainerMN communicator.
    comm = chainermnx.create_communicator("pure_nccl")
    device = comm.intra_rank

    if comm.rank == 0:
        print('==========================================')
        print('Num of GPUs : {}'.format(comm.size))
        print('Model :  {}'.format(args.model))
        print('Minibatch-size: {}'.format(batch_size))
        print('Epochs: {}'.format(args.epochs))
        print('==========================================')
        # Clean up logs and directories from previous runs. This is temporary. In the future just add time stamps to logs
        # Directories are created later by the reporter.
        #TODO
        # Change and use pathlib for this part
        try:
            shutil.rmtree(out)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
    # model = models[args.model]()
    model = L.Classifier(models[args.model]())
    # model = L.Classifier(ResNet152(152))


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

    train_iter = chainer.iterators.MultithreadIterator(train, batch_size, n_threads=80, shuffle=True)
    val_iter = chainer.iterators.MultithreadIterator(val, batch_size, n_threads=80, repeat=False)

    # Create a multi node optimizer from a standard Chainer optimizer.
    # This optimiser is modified to take two comms are its the same used for other parallelism strategies.
    optimizer = chainermnx.create_multi_node_optimizer(chainer.optimizers.Adam(), comm, comm, out)
    optimizer.setup(model)
    # Set up a trainer
    #TODO
    # Remember to change this updater to the stardard updater not chainermnx
    # You put this in oder to measure compute and data load time

    updater = chainermnx.training.StandardUpdater(train_iter, optimizer, comm, out=out, device=device)
    # updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (epochs, 'epoch'), out)

    val_interval = (100, 'epoch')
    log_interval = (1, 'epoch')

    # Create a multi node evaluator from an evaluator.
    evaluator = extensions.Evaluator(val_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator, trigger=val_interval)

    # Give file names data and time to prevent loosing information during reruns

    filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log"

    if comm.rank == 0:
        trainer.extend(extensions.DumpGraph('main/loss'))
        trainer.extend(extensions.LogReport(trigger=log_interval, filename=filename))
        # trainer.extend(extensions.observe_lr(), trigger=log_interval)
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

    if comm.rank == 0:
        print("Finished")


if __name__ == '__main__':
    main()
