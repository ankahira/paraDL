import chainer
import chainer.functions as F
import chainer.links as L


class Simple(chainer.Chain):
    def __init__(self):
        super(Simple, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 96, 11, stride=4)
            self.fc1 = L.Linear(None, 1000)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = self.fc1(h)
        return h



