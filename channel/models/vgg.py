import chainer
import chainer.functions as F
import chainermnx.functions as FX
from chainermnx.links import ChannelParallelConvolution2D, ChannelParallelFC
import chainer.links as L


class VGG(chainer.Chain):

    def __init__(self, comm):
        super(VGG, self).__init__()
        self.comm = comm
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 64, 3, 1, 1)
            self.conv1_2 = L.Convolution2D(None, 64, 3, 1, 1)

            self.conv2_1 = L.Convolution2D(None, 128, 3, 1, 1)
            self.conv2_2 = L.Convolution2D(None, 128, 3, 1, 1)

            self.conv3_1 = L.Convolution2D(None, 256, 3, 1, 1)
            self.conv3_2 = L.Convolution2D(None, 256, 3, 1, 1)
            self.conv3_3 = L.Convolution2D(None, 256, 3, 1, 1)

            self.conv4_1 = L.Convolution2D(None, 512, 3, 1, 1)
            self.conv4_2 = L.Convolution2D(None, 512, 3, 1, 1)
            self.conv4_3 = L.Convolution2D(None, 512, 3, 1, 1)

            self.conv5_1 = L.Convolution2D(None, 512, 3, 1, 1)
            self.conv5_2 = L.Convolution2D(None, 512, 3, 1, 1)
            self.conv5_3 = L.Convolution2D(None, 512, 3, 1, 1)

            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(4096, 4096)
            self.fc8 = L.Linear(4096, 1000)

    def forward(self, x):
        h = FX.split(self.comm, x)
        h = F.relu(FX.allreduce(self.conv1_1(h), self.comm))

        h = FX.split(self.comm, h)
        h = F.relu(FX.allreduce(self.conv1_2(h), self.comm))

        h = F.max_pooling_2d(h, 2, 2)

        h = FX.split(self.comm, h)
        h = F.relu(FX.allreduce(self.conv2_1(h), self.comm))

        h = FX.split(self.comm, h)
        h = F.relu(FX.allreduce(self.conv2_2(h), self.comm))

        h = F.max_pooling_2d(h, 2, 2)

        h = FX.split(self.comm, h)
        h = F.relu(FX.allreduce(self.conv3_1(h), self.comm))

        h = FX.split(self.comm, h)
        h = F.relu(self.conv3_2(h))
        h = FX.allreduce(h, self.comm)

        h = FX.split(self.comm, h)
        h = F.relu(FX.allreduce(self.conv3_3(h), self.comm))

        h = F.max_pooling_2d(h, 2, 2)

        h = FX.split(self.comm, h)
        h = F.relu(FX.allreduce(self.conv4_1(h), self.comm))

        h = FX.split(self.comm, h)
        h = F.relu(FX.allreduce(self.conv4_2(h), self.comm))

        h = FX.split(self.comm, h)
        h = F.relu(FX.allreduce(self.conv4_3(h), self.comm))

        h = F.max_pooling_2d(h, 2, 2)

        h = FX.split(self.comm, h)
        h = F.relu(FX.allreduce(self.conv5_1(h), self.comm))

        h = FX.split(self.comm, h)
        h = F.relu(FX.allreduce(self.conv5_2(h), self.comm))

        h = FX.split(self.comm, h)
        h = F.relu(FX.allreduce(self.conv5_3(h), self.comm))

        h = F.max_pooling_2d(h, 2, 2)

        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)

        return h















