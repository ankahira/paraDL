import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from .spatial_convolution import SpatialConvolution2D, SpatialConvolution2DGather


class AlexNet(chainer.Chain):

    insize = 226

    def __init__(self, comm):
        super(AlexNet, self).__init__()
        self.comm = comm
        self.n_proc = self.comm.size
        with self.init_scope():
            self.conv1 = SpatialConvolution2DGather(comm, None, 96, 11, stride=4)
            self.conv2 = L.Convolution2D(None, 256, 5, pad=2)
            self.conv3 = L.Convolution2D(None, 384, 3, pad=1)
            self.conv4 = L.Convolution2D(None, 384, 3, pad=1)
            self.conv5 = L.Convolution2D(None, 256, 3, pad=1)
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, 1000)

    def forward(self, x):
        partions = np.array_split(x, self.n_proc, 3)

        if self.comm.rank == 0:
            x = partions[0]
        elif self.comm.rank == 1:
            x = partions[1]
        elif self.comm.rank == 2:
            x = partions[2]
        elif self.comm.rank == 3:
            x = partions[3]
        else:
            print("Rank does not exist")

        h = F.relu(self.conv1(x))

        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)

        # TODO Implement all reduce here

        h = F.relu(self.fc6(h))
        h = F.dropout(h, ratio=0.5)
        h = F.relu(self.fc7(h))
        h = F.dropout(h, ratio=0.5)
        h = self.fc8(h)
        return h


