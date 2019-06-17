import tensorflow as tf
import math
import numpy as np
import chainer as ch
from chainer import backend
from chainer import backends
from chainer.backends import cuda
from chainer import Function, FunctionNode, gradient_check, report, training, utils, Variable
from chainer import datasets, initializers, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import pprint

# Local Modules
from models.CosmoNet import CosmoNet

import matplotlib

matplotlib.use('Agg')


def data_prep():
    X = np.random.rand(32, 1, 64, 64, 64).astype(np.float32)

    Y = np.random.rand(32, 2).astype(np.float32)

    return X, Y


def main():

    # Input data and label

    x, y = data_prep()

    train, test = datasets.split_dataset_random(datasets.TupleDataset(x, y), first_size=20)

    train_iterator = ch.iterators.SerialIterator(train, 10)
    test_iter = ch.iterators.SerialIterator(test, 5, repeat=False, shuffle=False)

    model = L.Linear(CosmoNet())

    optimizer = ch.optimizers.Adam()

    optimizer.setup(model)

    updater = training.StandardUpdater(train_iterator, optimizer)

    trainer = training.Trainer(updater, (5, 'epoch'), out='results')
    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model))
    # trainer.extend(extensions.DumpGraph('main/loss'))

    trainer.extend(extensions.snapshot(), trigger=(20, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    #  Run the training
    trainer.run()


if __name__ == "__main__":

    main()


