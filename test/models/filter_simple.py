
import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainermnx.links import FilterParallelConvolution2D,  FilterParallelFC


class FilterSimple(chainer.Chain):
    def __init__(self, original_comm, comm, out):
        self.comm = comm
        self.original_comm = original_comm
        self.out = out
        super(FilterSimple, self).__init__()
        with self.init_scope():
            self.conv1 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, 3, 96, ksize=11, stride=4, initialW=initializers.HeNormal())
            self.fc = L.Linear(None, 1000)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.fc(h)
        return h


