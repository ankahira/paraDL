import chainer
import chainer.functions as F
import chainermnx.functions as FX
from chainermnx.links import ChannelParallelConvolution2D, ChannelParallelFC
import chainer.links as L


class AlexNet(chainer.Chain):

    def __init__(self, comm):
        self.comm = comm
        super(AlexNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 96, 11, stride=4)
            self.conv2 = L.Convolution2D(None, 256, 5, pad=2)
            self.conv3 = L.Convolution2D(None, 384, 3, pad=1)
            self.conv4 = L.Convolution2D(None, 384, 3, pad=1)
            self.conv5 = L.Convolution2D(None, 256, 3, pad=1)
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, 1000)

    def forward(self, x):
        h = FX.split(self.comm, x)
        h = FX.allreduce(self.conv1(h), self.comm)
        h = F.relu(h)

        h = F.max_pooling_2d(h, ksize=3, stride=2)

        h = FX.split(self.comm, h)
        h = FX.allreduce(self.conv2(h), self.comm)
        h = F.relu(h)

        h = F.max_pooling_2d(h, ksize=3, stride=2)

        h = FX.split(self.comm, h)
        h = FX.allreduce(self.conv3(h), self.comm)
        h = F.relu(h)

        h = FX.split(self.comm, h)
        h = FX.allreduce(self.conv4(h), self.comm)
        h = F.relu(h)

        h = FX.split(self.comm, h)
        h = FX.allreduce(self.conv5(h), self.comm)
        h = F.relu(h)

        h = F.max_pooling_2d(h, ksize=3, stride=2)

        h = F.relu(self.fc6(h))
        h = F.dropout(h, ratio=0.5)

        h = F.relu(self.fc7(h))
        h = F.dropout(h, ratio=0.5)
        h = self.fc8(h)
        return h
