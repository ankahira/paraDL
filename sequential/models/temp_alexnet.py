import chainer
import chainer.functions as F
import chainer.links as L
import cupy as cp
from chainermnx.functions.halo_exchange import halo_exchange
from chainermnx.functions.checker import checker



from chainer.initializers import Constant


class AlexNet(chainer.Chain):
    def __init__(self):
        super(AlexNet, self).__init__()
        with self.init_scope():
            cp.random.seed(0)
            self.conv1 = L.Convolution2D(None, 3,  3, pad=1, nobias=True, initialW=Constant(cp.random.rand()))
            cp.random.seed(1)
            self.conv2 = L.Convolution2D(None, 3,  3, pad=1, nobias=True, initialW=Constant(cp.random.rand()))
            cp.random.seed(2)
            self.conv3 = L.Convolution2D(None, 3,  3, pad=1, nobias=True, initialW=Constant(cp.random.rand()))
            cp.random.seed(3)
            self.conv4 = L.Convolution2D(None, 3,  3, pad=1, nobias=True, initialW=Constant(cp.random.rand()))
            cp.random.seed(4)
            self.conv5 = L.Convolution2D(None, 3,  3, pad=1, nobias=True, initialW=Constant(cp.random.rand()))
            cp.random.seed(5)
            self.fc6 = L.Linear(None, 4096, nobias=True, initialW=Constant(cp.random.rand()))
            cp.random.seed(6)
            self.fc7 = L.Linear(None, 4096, nobias=True, initialW=Constant(cp.random.rand()))
            cp.random.seed(7)
            self.fc8 = L.Linear(None, 1000, nobias=True, initialW=Constant(cp.random.rand()))

    def forward(self, x, t):
        h = F.relu(self.conv1(x))
        h = checker(comm=None, index=2, x=h)
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)
        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss


