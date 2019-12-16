import chainer
import chainer.functions as F
import chainer.links as L
import cupy as cp
from chainermnx.functions.halo_exchange import halo_exchange


from chainer.initializers import Constant


class AlexNet(chainer.Chain):
    def __init__(self):
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

    def forward(self, x, t):
        h = F.relu(self.conv1(x))
        h = halo_exchange(self.comm, x, k_size=11, index=1, pad=0)

        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=4, stride=4)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(h, ksize=4, stride=4)

        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss


