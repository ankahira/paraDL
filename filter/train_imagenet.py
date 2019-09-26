import random
import argparse
import numpy as np
import chainer
from chainer import dataset
from chainer import training
from chainer.training import extensions


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
BATCH_SIZE = 32
EPOCHS = 10


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
        'alex': AlexNet,
        'resnet': ResNet50,
        'vgg': VGG,
    }

    parser = argparse.ArgumentParser(description='Train ImageNet From Scratch')
    parser.add_argument('--model', '-m', choices=models.keys(), default='AlexNet', help='Convnet model')
    parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--logdir', '-l', default='results', help='log dir')
    args = parser.parse_args()

    device = args.gpu

    print('Device: {}'.format(device))
    print('Dtype: {}'.format(chainer.config.dtype))
    print('# Minibatch-size: {}'.format(BATCH_SIZE))
    print('# epoch: {}'.format(EPOCHS))
    print('')

    # Initialize the model to train

    model = models[args.model]()

    # TODO :  Implement some code for checkpointing to restart
    chainer.backends.cuda.get_device_from_id(device).use()
    model.to_gpu()

    # Load the mean file
    mean = np.load(MEAN_FILE)

    # Load the dataset files
    train = PreprocessedDataset(TRAIN, TRAINING_ROOT, mean, model.insize)
    val = PreprocessedDataset(VAL, VALIDATION_ROOT, mean, model.insize, False)
    train_iter = chainer.iterators.MultiprocessIterator(train, BATCH_SIZE)
    val_iter = chainer.iterators.MultiprocessIterator(val, BATCH_SIZE, repeat=False)
    converter = dataset.concat_examples

    # Set up an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(train_iter, optimizer, converter=converter, device=device)
    trainer = training.Trainer(updater, (EPOCHS, 'epoch'), out=args.logdir)

    # Extensions and Reporting
    trainer.extend(extensions.Evaluator(val_iter, model, converter=converter, device=device), trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot(filename='trainer_checkpoint'), trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, 'model_checkpoint'), trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(extensions.observe_lr(), trigger=(1, 'epoch'))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr'
    ]), trigger=(1, 'epoch'))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', filename='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', filename='accuracy.png'))
    trainer.extend(extensions.ProgressBar())

    trainer.run()


if __name__ == '__main__':
    main()
