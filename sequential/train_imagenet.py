import os
import random
import argparse
import numpy as np
import chainer
from chainer import training
from chainer.training import extensions
import sys
import numpy
from chainer import dataset
import cupy as cp
from datetime import datetime
import chainermn
import chainermnx
from chainer.function_hooks import TimerHook


from chainer import serializers
import chainer.links as L

# Local Imports
from models.alexnet import AlexNet
from models.vgg import VGG
from models.resnet import ResNet50, ResNet101, ResNet152


import matplotlib
matplotlib.use('Agg')

# Global Variables
numpy.set_printoptions(threshold=sys.maxsize)

TRAIN = "/groups2/gaa50004/data/ILSVRC2012/pytorch/train/train.txt"
VAL = "/groups2/gaa50004/data/ILSVRC2012/val_256x256/val.txt"
TRAINING_ROOT = "/groups2/gaa50004/data/ILSVRC2012/pytorch/train/"
VALIDATION_ROOT = "/groups2/gaa50004/data/ILSVRC2012/val_256x256"
MEAN_FILE = "/groups2/gaa50004/data/ILSVRC2012/pytorch/train/mean.npy"

TIME = 0


class PreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, path, root, mean, crop_size, random=True):
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

    # These two lines help with memory. If they are not included training runs out of memory.
    # Use them till you the real reason why its running out of memory
    # pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    # cp.cuda.set_allocator(pool.malloc)

    models = {
        'alexnet': AlexNet,
        'resnet50': ResNet50,
        'vgg': VGG,
        'resnet152': ResNet152,
    }

    parser = argparse.ArgumentParser(description='Train ImageNet Sequential')
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


    print('Model: {} '.format(args.model))
    print('Minibatch-size: {}'.format(batch_size))
    print('Epochs: {}'.format(epochs))
    print('')

    # Initialize the model to train
    # model = models[args.model]()
    model = L.Classifier(models[args.model]())

    chainer.backends.cuda.get_device_from_id(device).use()
    model.to_gpu()

    # Load the mean file
    mean = np.load(MEAN_FILE)

    # Load the dataset files
    train = PreprocessedDataset(TRAIN, TRAINING_ROOT, mean, 226, True)
    val = PreprocessedDataset(VAL, VALIDATION_ROOT, mean, 226, False)

    train_iter = chainer.iterators.MultithreadIterator(
        train, batch_size, n_threads=80, shuffle=True)
    val_iter = chainer.iterators.MultithreadIterator(
        val, batch_size, n_threads=80, repeat=False)
    converter = dataset.concat_examples

    optimizer =chainer.optimizers.Adam()
    optimizer.setup(model)

    # Set up a trainer
    updater = chainer.training.updaters.StandardUpdater(
        train_iter, optimizer, converter=converter, device=device)
    trainer = training.Trainer(updater, (epochs, 'iteration'), out)

    val_interval = (100, 'epoch')
    log_interval = (1, 'epoch')

    evaluator = extensions.Evaluator(val_iter, model, device=device)
    trainer.extend(evaluator, trigger=val_interval)

    # trainer.extend(extensions.DumpGraph('main/loss'))
    filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log"

    trainer.extend(extensions.LogReport(trigger=log_interval, filename=filename))

    # trainer.extend(extensions.observe_lr(), trigger=log_interval)
    # trainer.extend(extensions.PrintReport(
    #     ['epoch', 'main/loss', 'validation/main/loss',
    #      'main/accuracy', 'validation/main/accuracy', 'elapsed_time', 'lr']), trigger=log_interval)
    trainer.extend(extensions.ProgressBar())

    print("Starting training")

    # trainer.run()
    hook = TimerHook()
    time_hook_results_file = open(os.path.join(args.out, "function_times.txt"), "a")
    with hook:
        trainer.run()

    hook.print_report()
    hook.print_report(file=time_hook_results_file)


if __name__ == '__main__':
    main()
