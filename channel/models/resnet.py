
import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L
import chainermnx.functions as FX


class BottleNeckA(chainer.Chain):

    def __init__(self, comm, in_size, ch, out_size, stride=2):
        self.comm = comm
        super(BottleNeckA, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, ch, 1, stride, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(None, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(None, out_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(out_size)

            self.conv4 = L.Convolution2D(None, out_size, 1, stride, 0, initialW=initialW, nobias=True)
            self.bn4 = L.BatchNormalization(out_size)

    def __call__(self, x):
        h = FX.split(self.comm, x)
        h1 = F.relu(self.bn1(FX.allreduce(self.conv1(h), self.comm)))
        h1 = FX.split(self.comm, h1)
        h1 = F.relu(self.bn2(FX.allreduce(self.conv2(h1), self.comm)))
        h1 = FX.split(self.comm, h1)
        h1 = self.bn3(FX.allreduce(self.conv3(h1),  self.comm))
        h2 = FX.split(self.comm, x)
        h2 = self.bn4(FX.allreduce(self.conv4(h2),  self.comm))
        return F.relu(h1 + h2)


class BottleNeckB(chainer.Chain):

    def __init__(self, comm, in_size, ch):
        self.comm = comm
        super(BottleNeckB, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, ch, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(None, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(None, in_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(in_size)

    def __call__(self, x):
        h = FX.split(self.comm, x)
        h = F.relu(self.bn1(FX.allreduce(self.conv1(h), self.comm)))
        h = FX.split(self.comm, h)
        h = F.relu(self.bn2(FX.allreduce(self.conv2(h), self.comm)))
        h = FX.split(self.comm, h)
        h = self.bn3(FX.allreduce(self.conv3(h), self.comm))

        return F.relu(h + x)


class Block(chainer.ChainList):

    def __init__(self, comm, layer, in_size, ch, out_size, stride=2):
        self.comm = comm
        super(Block, self).__init__()
        self.add_link(BottleNeckA(self.comm, in_size, ch, out_size, stride))
        for i in range(layer - 1):
            self.add_link(BottleNeckB(self.comm, out_size, ch))

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x


class ResNet50(chainer.Chain):
    insize = 224

    def __init__(self, comm):
        self.comm = comm
        super(ResNet50, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 7, 2, 3, initialW=initializers.HeNormal())
            self.bn1 = L.BatchNormalization(64)
            self.res2 = Block(self.comm, 3, 64, 64, 256, 1)
            self.res3 = Block(self.comm, 4, 256, 128, 512)
            self.res4 = Block(self.comm, 6, 512, 256, 1024)
            self.res5 = Block(self.comm, 3, 1024, 512, 2048)
            self.fc = L.Linear(2048, 1000)

    def __call__(self, x):
        h = FX.split(self.comm, x)
        h = self.bn1(FX.allreduce(self.conv1(h), self.comm))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.fc(h)
        return h


class ResNet101(chainer.Chain):
    insize = 224

    def __init__(self, comm):
        self.comm = comm
        super(ResNet101, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 7, 2, 3, initialW=initializers.HeNormal())
            self.bn1 = L.BatchNormalization(64)
            self.res2 = Block(self.comm, 3, 64, 64, 256, 1)
            self.res3 = Block(self.comm, 4, 256, 128, 512)
            self.res4 = Block(self.comm, 23, 512, 256, 1024)
            self.res5 = Block(self.comm, 3, 1024, 512, 2048)
            self.fc = L.Linear(2048, 1000)

    def __call__(self, x):
        h = FX.split(self.comm, x)
        h = self.bn1(FX.allreduce(self.conv1(h), self.comm))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.fc(h)
        return h


class ResNet152(chainer.Chain):
    insize = 224

    def __init__(self, comm):
        self.comm = comm
        super(ResNet152, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 7, 2, 3, initialW=initializers.HeNormal())
            self.bn1 = L.BatchNormalization(64)
            self.res2 = Block(self.comm, 3, 64, 64, 256, 1)
            self.res3 = Block(self.comm, 8, 256, 128, 512)
            self.res4 = Block(self.comm, 36, 512, 256, 1024)
            self.res5 = Block(self.comm, 3, 1024, 512, 2048)
            self.fc = L.Linear(2048, 1000)

    def __call__(self, x):
        h = FX.split(self.comm, x)
        h = self.bn1(FX.allreduce(self.conv1(h), self.comm))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.fc(h)
        return h