import chainer
import chainer.functions as F
import chainer.links as L


# Local Imports
from chainermnx.links import FilterParallelConvolution2D,  FilterParallelFC


import chainer
import chainer.functions as F
import chainer.links as L


class VGG(chainer.Chain):
    def __init__(self,  original_comm, comm, out):
        super(VGG, self).__init__()
        self.comm = comm
        self.original_comm = original_comm
        self.out = out
        with self.init_scope():
            self.conv1_1 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, 3, 64, 3, 1, 1)
            self.conv1_2 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, 64, 64, 3, 1, 1)

            self.conv2_1 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, 64, 128, 3, 1, 1)
            self.conv2_2 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, 128, 128, 3, 1, 1)

            self.conv3_1 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, 128, 256, 3, 1, 1)
            self.conv3_2 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, 256, 256, 3, 1, 1)
            self.conv3_3 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, 256, 256, 3, 1, 1)

            self.conv4_1 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, 256, 512, 3, 1, 1)
            self.conv4_2 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, 512, 512, 3, 1, 1)
            self.conv4_3 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, 512, 512, 3, 1, 1)

            self.conv5_1 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, 512, 512, 3, 1, 1)
            self.conv5_2 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, 512, 512, 3, 1, 1)
            self.conv5_3 = FilterParallelConvolution2D(self.original_comm, self.comm, self.out, 512, 512, 3, 1, 1)

            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(4096, 4096)
            self.fc8 = L.Linear(4096, 1000)

    def forward(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, 2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, 2)

        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)

        return h


