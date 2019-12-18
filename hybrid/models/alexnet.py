import chainer
import chainer.functions as F
import chainermnx.functions as FX
import chainer.links as L
import chainermnx.links as LX
import chainermn
import cupy as cp
from chainer.initializers import Constant
from chainermnx.functions.halo_exchange import halo_exchange


class AlexNet(chainer.Chain):

    def __init__(self, comm):
        super(AlexNet, self).__init__()
        self.comm = comm
        self.n_proc = self.comm.size
        with self.init_scope():
            # self.conv1 = L.Convolution2D(None, 96, 11, stride=4)
            self.conv1 = LX.Convolution2D(self.comm, 1, None, 3,  3, pad=(0, 1))
            self.conv2 = L.Convolution2D(None, 256, 5, pad=(0, 2))
            self.conv3 = L.Convolution2D(None, 384, 3, pad=(0, 1))
            self.conv4 = L.Convolution2D(None, 384, 3, pad=(0, 1))
            self.conv5 = L.Convolution2D(None, 256, 3, pad=(0, 1))
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, 1000)

    def forward(self, x, t):

        partions = cp.array_split(x, self.comm.size, -2)
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
        h = halo_exchange(self.comm, x, k_size=11, index=1, pad=0)
        h = F.relu(self.conv1(h))
        h = halo_exchange(self.comm, h, k_size=5, index=2, pad=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=4, stride=4)

        h = halo_exchange(self.comm, h, k_size=3, index=3, pad=1)
        h = F.relu(self.conv3(h))

        h = halo_exchange(self.comm, h, k_size=3, index=4, pad=1)
        h = F.relu(self.conv4(h))

        h = halo_exchange(self.comm, h, k_size=3, index=5, pad=1)
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(h, ksize=4, stride=4)
        hs = chainermn.functions.allgather(self.comm, h)
        h = F.concat(hs, -2)

        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss




