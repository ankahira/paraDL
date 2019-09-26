import sys

import numpy as np

import chainer


TRAIN = "/groups2/gaa50004/data/ILSVRC2012/train_256x256/train.txt"
VAL = "/groups2/gaa50004/data/ILSVRC2012/val_256x256/val.txt"
TRAINING_ROOT = "/groups2/gaa50004/data/ILSVRC2012/train_256x256/"
VALIDATION_ROOT = "/groups2/gaa50004/data/ILSVRC2012/val_256x256"
MEAN_FILE = "/groups2/gaa50004/data/ILSVRC2012/train_256x256/mean.npy"
BATCH_SIZE = 32
EPOCHS = 10
RESULTS_DIR = "results/alexnet"


def compute_mean(dataset):
    print('compute mean image')
    sum_image = 0
    N = len(dataset)
    for i, (image, _) in enumerate(dataset):
        sum_image += image
        sys.stderr.write('{} / {}\r'.format(i, N))
        sys.stderr.flush()
    sys.stderr.write('\n')
    return sum_image / N


def main():

    dataset = chainer.datasets.LabeledImageDataset(TRAIN, TRAINING_ROOT)
    mean = compute_mean(dataset)
    np.save(MEAN_FILE, mean)


if __name__ == '__main__':
    main()
