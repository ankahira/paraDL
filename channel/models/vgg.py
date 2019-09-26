from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L


# Local Imports
from .parallel_convolution import ParallelConvolution2D


class Block(chainer.Chain):
    def __init__(self, comm, in_channels, out_channels, ksize, pad=1):
        super(Block, self).__init__()
        with self.init_scope():
            if comm.size <= in_channels:
                self.conv = ParallelConvolution2D(comm,
                                                  in_channels,
                                                  out_channels,
                                                  ksize,
                                                  pad=pad,
                                                  nobias=True)
            else:
                self.conv = chainer.links.Convolution2D(in_channels,
                                                        out_channels,
                                                        ksize,
                                                        pad=pad,
                                                        nobias=True)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return F.relu(h)


class VGG(chainer.Chain):
    def __init__(self, comm, class_labels=1000):
        super(VGG, self).__init__()
        self.comm = comm

        with self.init_scope():
            self.block1_1 = Block(comm, 3,   64,  3)
            self.block1_2 = Block(comm, 64,  64,  3)
            self.block2_1 = Block(comm, 64,  128, 3)
            self.block2_2 = Block(comm, 128, 128, 3)
            self.block3_1 = Block(comm, 128, 256, 3)
            self.block3_2 = Block(comm, 256, 256, 3)
            self.block3_3 = Block(comm, 256, 256, 3)
            self.block4_1 = Block(comm, 256, 512, 3)
            self.block4_2 = Block(comm, 512, 512, 3)
            self.block4_3 = Block(comm, 512, 512, 3)
            self.block5_1 = Block(comm, 512, 512, 3)
            self.block5_2 = Block(comm, 512, 512, 3)
            self.block5_3 = Block(comm, 512, 512, 3)
            self.fc1 = L.Linear(None, 512, nobias=True)
            self.bn_fc1 = L.BatchNormalization(512)
            self.fc2 = L.Linear(None, class_labels, nobias=True)

    def __call__(self, x):
        # 64 channel blocks:
        h = self.block1_1(x)
        h = F.dropout(h, ratio=0.3)
        h = self.block1_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 128 channel blocks:
        h = self.block2_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block2_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 256 channel blocks:
        h = self.block3_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block3_2(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block3_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block4_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block4_2(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block4_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block5_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block5_2(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block5_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.dropout(h, ratio=0.5)
        h = self.fc1(h)
        h = self.bn_fc1(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)
        h = self.fc2(h)

        return h
