import chainer
import chainer.functions as F
import chainer.links as L
P= 2

class ResNet(chainer.Chain):
    def __init__(self, n_layers):
        super(ResNet, self).__init__()
        w = chainer.initializers.HeNormal()

        # expected insize = 224
        if n_layers == 18:
            block = [2, 2, 2, 2]
        elif n_layers == 34:
            block = [3, 4, 6, 3]
        elif n_layers == 50:
            block = [3, 4, 6, 3]
        elif n_layers == 101:
            block = [3, 4, 23, 3]
        elif n_layers == 152:
            block = [3, 4, 36, 3]
        # expected insize = 32
        elif n_layers == 20:
            block = [3, 3, 3]
        elif n_layers == 32:
            block = [5, 5, 5]
        elif n_layers == 44:
            block = [7, 7, 7]
        elif n_layers == 56:
            block = [9, 9, 9]
        elif n_layers == 110:
            block = [18, 18, 18]
        elif n_layers == 1202:
            block = [200, 200, 200]
        else:
            raise

        with self.init_scope():
            if n_layers in [18, 34]:
                self.conv1 = L.Convolution2D(3, 64//P, 7, 2, 3, initialW=w)
                self.bn1 = L.BatchNormalization(64)
                self.res2 = BasicBlock(block[0], 64, 1)
                self.res3 = BasicBlock(block[1], 128)
                self.res4 = BasicBlock(block[2], 256)
                self.res5 = BasicBlock(block[3], 512)
            elif n_layers in [20, 32, 44, 56, 110, 1202]:
                self.conv1 = L.Convolution2D(3, 16//P, 7, 2, 3, initialW=w)
                self.bn1 = L.BatchNormalization(16)
                self.res2 = BasicBlock(block[0], 16, 1)
                self.res3 = BasicBlock(block[1], 32)
                self.res4 = BasicBlock(block[2], 64)
            elif n_layers in [50, 101, 152]:
                self.conv1 = L.Convolution2D(3, 64//P, 7, 2, 3, initialW=w)
                self.bn1 = L.BatchNormalization(64)
                self.res2 = BottleNeckBlock(block[0], 64, 64, 256, 1)
                self.res3 = BottleNeckBlock(block[1], 256, 128, 512)
                self.res4 = BottleNeckBlock(block[2], 512, 256, 1024)
                self.res5 = BottleNeckBlock(block[3], 1024, 512, 2048)
            self.fc = L.Linear(None, 1000)

    def __call__(self, x):
        h = self.predict(x)
        return h
        # loss = F.softmax_cross_entropy(h, t)
        # chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        # return loss

    def predict(self, x):
        h = self.bn1(self.conv1(x))
        h = F.concat((h, h, h, h), axis=1)
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        if hasattr(self, 'res5'):
            h = self.res5(h)
            h = F.average_pooling_2d(h, 7, stride=1)
        else:
            h = F.average_pooling_2d(h, 1, stride=1)
        h = self.fc(h)

        return h


class BasicBlock(chainer.ChainList):

    def __init__(self, n_blocks, ch, stride=2):
        super(BasicBlock, self).__init__()
        with self.init_scope():
            self.add_link(BasicA(ch, stride))
            for i in range(n_blocks - 1):
                self.add_link(BasicB(ch))

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x


class BottleNeckBlock(chainer.ChainList):

    def __init__(self, n_blocks, in_size, ch, out_size, stride=2):
        super(BottleNeckBlock, self).__init__()
        self.add_link(BottleNeckA(in_size, ch, out_size, stride))
        for i in range(n_blocks - 1):
            self.add_link(BottleNeckB(out_size, ch))

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x


class BasicA(chainer.Chain):

    def __init__(self, ch, stride):
        super(BasicA, self).__init__()
        w = chainer.initializers.HeNormal()

        with self.init_scope():
            self.conv1a = L.Convolution2D(None, ch//P, 3, stride, 1, initialW=w, nobias=True)
            self.bn1a = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(None, ch//P, 3, 1, 1, initialW=w, nobias=True)
            self.bn2 = L.BatchNormalization(ch)

            self.conv1b = L.Convolution2D(None, ch//P, 3, stride, 1, initialW=w, nobias=True)
            self.bn1b = L.BatchNormalization(ch)

    def __call__(self, x):
        h1 = F.relu(self.bn1a(self.conv1a(x)))
        h1 = F.concat((h1, h1, h1, h1), axis=1)

        h1 = self.bn2(self.conv2(h1))
        h2 = self.bn1b(self.conv1b(x))
        h2 = F.concat((h2, h2, h2, h2), axis=1)

        return F.relu(h1 + h2)


class BasicB(chainer.Chain):

    def __init__(self, ch):
        super(BasicB, self).__init__()
        w = chainer.initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, ch, 3, 1, 1, initialW=w, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(None, ch, 3, 1, 1, initialW=w, nobias=True)
            self.bn2 = L.BatchNormalization(ch)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))

        return F.relu(h + x)


class BottleNeckA(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2):
        super(BottleNeckA, self).__init__()
        w = chainer.initializers.HeNormal()

        with self.init_scope():
            self.conv1a = L.Convolution2D(
                in_size, ch, 1, stride, 0, initialW=w, nobias=True)
            self.bn1a = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=w, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(
                ch, out_size, 1, 1, 0, initialW=w, nobias=True)
            self.bn3 = L.BatchNormalization(out_size)

            self.conv1b = L.Convolution2D(
                in_size, out_size, 1, stride, 0,
                initialW=w, nobias=True)
            self.bn1b = L.BatchNormalization(out_size)

    def __call__(self, x):
        h1 = F.relu(self.bn1a(self.conv1a(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn1b(self.conv1b(x))

        return F.relu(h1 + h2)


class BottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch):
        super(BottleNeckB, self).__init__()
        w = chainer.initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, 1, 0, initialW=w, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=w, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(
                ch, in_size, 1, 1, 0, initialW=w, nobias=True)
            self.bn3 = L.BatchNormalization(in_size)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))

        return F.relu(h + x)