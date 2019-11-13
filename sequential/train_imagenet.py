import random
import argparse
import numpy as np
import chainer
from chainer import training
from chainer.training import extensions
import sys
import numpy
from chainer import dataset

from chainer import serializers


# Local Imports
from models.alexnet import AlexNet
from models.vgg import VGG
from models.resnet50 import ResNet50

import matplotlib
matplotlib.use('Agg')

# Global Variables
numpy.set_printoptions(threshold=sys.maxsize)

TRAIN = "/groups2/gaa50004/data/ILSVRC2012/train_256x256/50_image_train.txt"
VAL = "/groups2/gaa50004/data/ILSVRC2012/val_256x256/val.txt"
TRAINING_ROOT = "/groups2/gaa50004/data/ILSVRC2012/train_256x256/"
VALIDATION_ROOT = "/groups2/gaa50004/data/ILSVRC2012/val_256x256"
MEAN_FILE = "/groups2/gaa50004/data/ILSVRC2012/train_256x256/mean.npy"


class PreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, path, root, mean, crop_size, random=False):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype(chainer.get_dtype())
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
    parser.add_argument('--gpu', '-g', type=int,  default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='results', help='Output directory')
    args = parser.parse_args()

    device = args.gpu
    batch_size = args.batchsize
    epochs = args.epochs
    out = args.out

    # print('Device: {}'.format(device))
    # print('Dtype: {}'.format(chainer.config.dtype))
    print('Model: {} '.format(args.model))
    print('Minibatch-size: {}'.format(batch_size))
    print('Epochs: {}'.format(epochs))
    print('')

    # Initialize the model to train
    model = models[args.model]()
    chainer.backends.cuda.get_device_from_id(device).use()
    model.to_gpu()

    # Load the mean file
    mean = np.load(MEAN_FILE)

    # Load the dataset files
    train = PreprocessedDataset(TRAIN, TRAINING_ROOT, mean, 256)
    val = PreprocessedDataset(VAL, VALIDATION_ROOT, mean, 256, False)
    # These iterators load the images with subprocesses running in parallel
    # to the training/validation.
    train_iter = chainer.iterators.MultiprocessIterator(
        train, batch_size, shuffle=False)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, batch_size, repeat=False)
    converter = dataset.concat_examples

    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=converter, device=device)
    trainer = training.Trainer(updater, (epochs, 'epoch'), out)

    val_interval = (100, 'epoch')
    log_interval = (1, 'epoch')

    trainer.extend(extensions.Evaluator(val_iter, model, converter=converter,
                                        device=device), trigger=val_interval)

    trainer.extend(extensions.DumpGraph('main/loss'))
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time', 'lr']), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=100))

    trainer.run()
    serializers.save_npz('sequential_model.npz', model)


if __name__ == '__main__':
    main()
