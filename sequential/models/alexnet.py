import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import One
import numpy as np
import cupy as cp

from utils.initializers import Own
from chainer.initializers import Constant


class AlexNet(chainer.Chain):

    def __init__(self):
        super(AlexNet, self).__init__()
        with self.init_scope():
            cp.random.seed(0)
            self.conv1 = L.Convolution2D(None,  1, 3, stride=1, pad=1, nobias=True, initialW=Constant(cp.random.rand()))
            cp.random.seed(1)
            self.conv2 = L.Convolution2D(None, 256,  5, pad=2, nobias=True)
            cp.random.seed(2)
            self.conv3 = L.Convolution2D(None, 384,  3, pad=1, nobias=True)
            cp.random.seed(3)
            self.conv4 = L.Convolution2D(None, 384,  3, pad=1, nobias=True)
            cp.random.seed(4)
            self.conv5 = L.Convolution2D(None, 256,  3, pad=1, nobias=True)
            cp.random.seed(5)
            self.fc6 = L.Linear(None, 4096, nobias=True)
            cp.random.seed(6)
            self.fc7 = L.Linear(None, 4096, nobias=True)
            cp.random.seed(7)
            self.fc8 = L.Linear(None, 1000, nobias=True)

    def forward(self, x, t):
        h = self.conv1(x)
        with open('sequential.txt', 'w') as f:
            for i in range(h.shape[-2]):
                for j in range(h.shape[-1]):
                    print("%01.5f" % h[:, :, i, j].array, file=f)

                    # h = F.relu(h)
        # h = F.max_pooling_2d(h, ksize=3, stride=2)
        # h = F.relu(self.conv2(h))
        # h = F.max_pooling_2d(h, ksize=3, stride=2)
        #
        # h = F.relu(self.conv3(h))
        # h = F.relu(self.conv4(h))
        # h = F.relu(self.conv5(h))
        # h = F.max_pooling_2d(h, ksize=3, stride=2)
        #
        # h = F.relu(self.fc6(h))
        # h = F.dropout(h, ratio=0.5)
        # h = F.relu(self.fc7(h))
        # h = F.dropout(h, ratio=0.5)
        # h = self.fc8(h)
        #
        # loss = F.softmax_cross_entropy(h, t)
        # chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        # return loss

        # return h





