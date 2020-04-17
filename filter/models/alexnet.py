import chainer
import chainer.functions as F
import chainer.links as L

# Local Imports
from chainermnx.links import FilterParallelConvolution2D,  FilterParallelFC


class AlexNet(chainer.Chain):
    def __init__(self, comm):
        super(AlexNet, self).__init__()
        with self.init_scope():
            self.conv1 = FilterParallelConvolution2D(comm, None, 96, 11, stride=4)
            self.conv2 = FilterParallelConvolution2D(comm, None, 256, 5, pad=2)
            self.conv3 = FilterParallelConvolution2D(comm, None, 384, 3, pad=1)
            self.conv4 = FilterParallelConvolution2D(comm, None, 384, 3, pad=1)
            self.conv5 = FilterParallelConvolution2D(comm, None, 256, 3, pad=1)

            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, 1000)
            # Removed temporary
            # self.fc6 = FilterParallelFC(comm, None, 4096)
            # self.fc7 = FilterParallelFC(comm, None, 4096)
            # self.fc8 = FilterParallelFC(comm, None, 1000)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.relu(self.fc6(h))
        h = F.dropout(h, ratio=0.5)
        h = F.relu(self.fc7(h))
        h = F.dropout(h, ratio=0.5)
        h = self.fc8(h)
        return h
