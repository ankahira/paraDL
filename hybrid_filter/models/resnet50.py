import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainermnx.links import FilterParallelConvolution2D,  FilterParallelFC


class BottleNeckA(chainer.Chain):

    def __init__(self, original_comm, comm, out, in_size, ch, out_size, stride=2):
        self.comm = comm
        self.original_comm = original_comm
        self.out = out
        super(BottleNeckA, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, in_size, ch, 1, stride, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, ch, out_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(out_size)
            self.conv4 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, in_size, out_size, 1, stride, 0, initialW=initialW, nobias=True)
            self.bn4 = L.BatchNormalization(out_size)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))

        return F.relu(h1 + h2)


class BottleNeckB(chainer.Chain):

    def __init__(self, original_comm, comm, out, in_size, ch):
        super(BottleNeckB, self).__init__()
        self.comm = comm
        self.original_comm = original_comm
        self.out = out
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, in_size, ch, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, ch, in_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(in_size)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))

        return F.relu(h + x)


class Block(chainer.ChainList):

    def __init__(self,  original_comm, comm, out, layer, in_size, ch, out_size, stride=2):
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

    insize = 226

    def __init__(self, original_comm, comm, out):
        self.comm = comm
        self.original_comm = original_comm
        self.out = out
        super(ResNet50, self).__init__()
        with self.init_scope():
            self.conv1 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, 3, 64, 7, 2, 3, initialW=initializers.HeNormal())
            self.bn1 = L.BatchNormalization(64)
            self.res2 = Block(self.original_comm, self.comm, self.out, 3, 64, 64, 256, 1)
            self.res3 = Block(self.original_comm, self.comm, self.out,  4, 256, 128, 512)
            self.res4 = Block(self.original_comm, self.comm, self.out,  6, 512, 256, 1024)
            self.res5 = Block(self.original_comm, self.comm, self.out,  3, 1024, 512, 2048)

            self.fc = L.Linear(2048, 1000)

    def __call__(self, x):
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.fc(h)
        return h


class ResNet101(chainer.Chain):

    insize = 226

    def __init__(self, original_comm, comm, out):
        self.comm = comm
        self.original_comm = original_comm
        self.out = out
        super(ResNet50, self).__init__()
        with self.init_scope():
            self.conv1 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, 3, 64, 7, 2, 3, initialW=initializers.HeNormal())
            self.bn1 = L.BatchNormalization(64)
            self.res2 = Block(self.original_comm, self.comm, self.out, 3, 64, 64, 256, 1)
            self.res3 = Block(self.original_comm, self.comm, self.out,  4, 256, 128, 512)
            self.res4 = Block(self.original_comm, self.comm, self.out,  23, 512, 256, 1024)
            self.res5 = Block(self.original_comm, self.comm, self.out,  3, 1024, 512, 2048)

            self.fc = L.Linear(2048, 1000)

    def __call__(self, x):
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.fc(h)
        return h


class ResNet152(chainer.Chain):

    insize = 226

    def __init__(self, original_comm, comm, out):
        self.comm = comm
        self.original_comm = original_comm
        self.out = out
        super(ResNet50, self).__init__()
        with self.init_scope():
            self.conv1 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, 3, 64, 7, 2, 3, initialW=initializers.HeNormal())
            self.bn1 = L.BatchNormalization(64)
            self.res2 = Block(self.original_comm, self.comm, self.out, 3, 64, 64, 256, 1)
            self.res3 = Block(self.original_comm, self.comm, self.out,  4, 256, 128, 512)
            self.res4 = Block(self.original_comm, self.comm, self.out,  36, 512, 256, 1024)
            self.res5 = Block(self.original_comm, self.comm, self.out,  3, 1024, 512, 2048)

            self.fc = L.Linear(2048, 1000)

    def __call__(self, x):
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.fc(h)
        return h
