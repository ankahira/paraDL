import random
import argparse
import numpy as np
import chainer
from chainer import dataset
from chainer import training
from chainer.training import extensions
from chainer.function_hooks import CupyMemoryProfileHook
import chainer.links as L


# Local Imports
from models.alexnet import AlexNet
from models.vgg import VGG
from models.resnet50 import ResNet50

import matplotlib
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

    model = L.Classifier(models[args.model]())
    chainer.backends.cuda.get_device_from_id(device).use()
    model.to_gpu()

    # TODO :  Implement some code for checkpointing to restart

    # Load the mean file
    mean = np.load(MEAN_FILE)

    # Load the dataset files
    train = PreprocessedDataset(TRAIN, TRAINING_ROOT, mean, 226)
    val = PreprocessedDataset(VAL, VALIDATION_ROOT, mean, 226, False)
    train_iter = chainer.iterators.MultiprocessIterator(train, batch_size)
    val_iter = chainer.iterators.MultiprocessIterator(val, batch_size, repeat=False)
    converter = dataset.concat_examples

    # Set up an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(train_iter, optimizer, converter=converter, device=device)
    trainer = training.Trainer(updater, (epochs, 'epoch'), out=out)

    # Extensions and Reporting
    trainer.extend(extensions.Evaluator(val_iter, model, converter=converter, device=device), trigger=(20, 'epoch'))

    trainer.extend(extensions.DumpGraph('main/loss'))
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(extensions.observe_lr(), trigger=(1, 'epoch'))
    trainer.extend(extensions.PrintReport(['epoch', 'elapsed_time', ]), trigger=(1, 'epoch'))
    # trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
    # 'epoch', filename='loss.png'))
    # trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],
    # 'epoch', filename='accuracy.png'))
    trainer.extend(extensions.ProgressBar())

    trainer.run()


if __name__ == '__main__':
    main()
