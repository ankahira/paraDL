import chainer
import chainermnx
import chainer.functions as F
import chainermnx.functions as FX
import chainer.links as L
import chainermnx.links as LX
import chainermn
import cupy as cp
from chainer.initializers import Constant
from chainermnx.functions.halo_exchange import halo_exchange
from chainermnx.functions.checker import checker
from chainer import serializers


class AlexNet(chainer.Chain):

    def __init__(self, comm):
        super(AlexNet, self).__init__()
        self.comm = comm
        self.n_proc = self.comm.size
        with self.init_scope():
            self.conv1 = LX.Convolution2D(self.comm, 1, None, 3, 3, pad=(0, 1), nobias=True)
            self.conv2 = LX.Convolution2D(self.comm, 2, None, 3, 3, pad=(0, 1), nobias=True)
            self.conv3 = LX.Convolution2D(self.comm, 3, None, 3, 3, pad=(0, 1), nobias=True)

            self.conv4 = L.Convolution2D(None, 3, 3, pad=1, nobias=True)
            self.conv5 = L.Convolution2D(None, 3, 3, pad=(0, 1), nobias=True, initialW=Constant(cp.random.rand()))
            self.conv6 = L.Convolution2D(None, 3, 3, pad=(0, 1), nobias=True, initialW=Constant(cp.random.rand()))
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

        h = halo_exchange(self.comm, x, k_size=3, index=1, pad=1)
        h = F.relu(self.conv1(h))
        h = halo_exchange(self.comm, h, k_size=3, index=2, pad=1)
        h = F.relu(self.conv2(h))
        h = halo_exchange(self.comm, h, k_size=3, index=3, pad=1)
        h = F.relu(self.conv3(h))
        h = chainermnx.functions.allgather(self.comm, h)
        h = FX.concat(h, -2)
        h = F.relu(self.conv4(h))
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss




