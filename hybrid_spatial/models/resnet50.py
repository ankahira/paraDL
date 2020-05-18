# Original author: yasunorikudo
# (https://github.com/yasunorikudo/chainer-ResNet)

import chainer
import chainermnx
import chainer.functions as F
from chainer import initializers
import chainer.links as L
import chainermnx.functions as FX
import chainermnx.links as LX
import cupy as cp

"""

Fix the resnet model padding
some pad regions were modified, make sure they are exactly the same as data parallel

"""

INDICES = list(range(1, 60))


class BottleNeckA(chainer.Chain):
    def __init__(self, original_comm, comm, out, in_size, ch, out_size, stride=2):
        super(BottleNeckA, self).__init__()
        self.comm = comm
        self.original_comm = original_comm
        self.out = out
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = LX.SpatialConvolution2D(self.original_comm, self.comm, self.out, INDICES[0], in_size, ch, 1, stride, 0, initialW=initialW, nobias=True)
            del INDICES[0]
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = LX.SpatialConvolution2D(self.original_comm, self.comm, self.out, INDICES[0], ch, ch, 3, 1, (0, 1), initialW=initialW, nobias=True)
            del INDICES[0]
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = LX.SpatialConvolution2D(self.original_comm, self.comm, self.out, INDICES[0], ch, out_size, 1, 1, 0, initialW=initialW, nobias=True)
            del INDICES[0]
            self.bn3 = L.BatchNormalization(out_size)

            self.conv4 = LX.SpatialConvolution2D(self.original_comm, self.comm, self.out, INDICES[0], in_size, out_size, 1, stride, 0, initialW=initialW, nobias=True)
            del INDICES[0]
            self.bn4 = L.BatchNormalization(out_size)

    def __call__(self, x):
        h1 = FX.halo_exchange(self.original_comm, self.comm, x, k_size=1, index=1, pad=0, out=self.out)
        h1 = F.relu(self.bn1(self.conv1(h1)))
        h1 = FX.halo_exchange(self.original_comm, self.comm, h1, k_size=3, index=2, pad=1, out=self.out)
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = FX.halo_exchange(self.original_comm, self.comm, h1, k_size=1, index=3, pad=0, out=self.out)
        h1 = self.bn3(self.conv3(h1))

        h2 = FX.halo_exchange(self.original_comm, self.comm, x, k_size=1, index=4, pad=0, out=self.out)
        h2 = self.bn4(self.conv4(h2))
        return F.relu(h1 + h2)


class BottleNeckB(chainer.Chain):

    def __init__(self, original_comm,  comm, out, in_size, ch):
        super(BottleNeckB, self).__init__()
        self.comm = comm
        self.original_comm = original_comm
        self.out = out
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = LX.SpatialConvolution2D(self.original_comm, self.comm, self.out,  INDICES[0], in_size, ch, 1, 1, 0, initialW=initialW, nobias=True)
            del INDICES[0]
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = LX.SpatialConvolution2D(self.original_comm, self.comm, self.out, INDICES[0], ch, ch, 3, 1, (0, 1), initialW=initialW, nobias=True)
            del INDICES[0]
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = LX.SpatialConvolution2D(self.original_comm, self.comm, self.out, INDICES[0], ch, in_size, 1, 1, 0, initialW=initialW, nobias=True)
            del INDICES[0]
            self.bn3 = L.BatchNormalization(in_size)

    def __call__(self, x):
        x = FX.halo_exchange(self.original_comm, self.comm, x, k_size=1, index=5, pad=0, out=self.out)
        h = F.relu(self.bn1(self.conv1(x)))
        h = FX.halo_exchange(self.original_comm, self.comm, h, k_size=3, index=6, pad=1, out=self.out)
        h = F.relu(self.bn2(self.conv2(h)))
        h = FX.halo_exchange(self.original_comm, self.comm, h, k_size=1, index=7, pad=0, out=self.out)
        h = self.bn3(self.conv3(h))

        return F.relu(h + x)


class Block(chainer.ChainList):

    def __init__(self, original_comm, comm, out, layer, in_size, ch, out_size, stride=2):
        super(Block, self).__init__()
        self.comm = comm
        self.original_comm = original_comm
        self.out = out
        self.add_link(BottleNeckA(self.original_comm, self.comm, self.out, in_size, ch, out_size, stride))
        for i in range(layer - 1):
            self.add_link(BottleNeckB(self.original_comm, self.comm, self.out, out_size, ch))

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x


class ResNet50(chainer.Chain):
    def __init__(self, original_comm, local_comm, out):
        super(ResNet50, self).__init__()
        self.comm = local_comm
        self.original_comm = original_comm
        self.out = out
        with self.init_scope():
            self.conv1 = LX.SpatialConvolution2D(self.original_comm, self.comm, self.out, INDICES[0], 3, 64, 7, 2, 3, initialW=initializers.HeNormal())
            del INDICES[0]
            self.bn1 = L.BatchNormalization(64)
            self.res2 = Block(self.original_comm, self.comm, self.out, 3, 64, 64, 256, 1)
            self.res3 = Block(self.original_comm, self.comm, self.out, 4, 256, 128, 512)
            self.res4 = Block(self.original_comm, self.comm, self.out, 6, 512, 256, 1024)
            self.res5 = Block(self.original_comm, self.comm, self.out, 3, 1024, 512, 2048)
            self.fc = L.Linear(None, 1000)

    def __call__(self, x):
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

        h = FX.halo_exchange(self.original_comm, self.comm, x, k_size=7, index=9, pad=3, out=self.out)
        h = self.bn1(self.conv1(h))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        hs = chainermnx.functions.spatialallgather(self.original_comm, self.comm, h, self.out)
        h = F.concat(hs, -2)
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.fc(h)
        return h