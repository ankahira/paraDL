import cupy as cp
import chainer
import chainer.functions as F
import chainer.links as L

from chainermnx.links import SpatialConvolution2D


class VGG(chainer.Chain):
    def __init__(self, comm):
        super(VGG, self).__init__()
        self.comm = comm
        self.n_proc = self.comm.size
        with self.init_scope():
            self.conv1_1 = SpatialConvolution2D(comm, 1, 3, 64, 3, 1, 1)
            self.conv1_2 = SpatialConvolution2D(comm, 2, 64, 64, 3, 1, 1)

            self.conv2_1 = SpatialConvolution2D(comm, 3, 64, 128, 3, 1, 1)
            self.conv2_2 = SpatialConvolution2D(comm, 4, 128, 128, 3, 1, 1)

            self.conv3_1 = SpatialConvolution2D(comm, 5, 128, 256, 3, 1, 1)
            self.conv3_2 = SpatialConvolution2D(comm, 6, 256, 256, 3, 1, 1)
            self.conv3_3 = SpatialConvolution2D(comm, 7, 256, 256, 3, 1, 1)

            self.conv4_1 = SpatialConvolution2D(comm, 8, 256, 512, 3, 1, 1)
            self.conv4_2 = SpatialConvolution2D(comm, 9, 512, 512, 3, 1, 1)
            self.conv4_3 = SpatialConvolution2D(comm, 10, 512, 512, 3, 1, 1)

            self.conv5_1 = SpatialConvolution2D(comm, 11, 512, 512, 3, 1, 1)
            self.conv5_2 = SpatialConvolution2D(comm, 12, 512, 512, 3, 1, 1)
            self.conv5_3 = SpatialConvolution2D(comm, 13, 512, 512, 3, 1, 1)

            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(4096, 4096)
            self.fc8 = L.Linear(4096, 1000)

    def forward(self, x):
        partions = cp.array_split(x, self.n_proc, 3)
        x = partions[self.comm.rank]

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)

        # h = spatial_max_pooling(self.comm, h, 15, 2, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)

        # h = spatial_max_pooling(self.comm, h, 16, 2, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)

        # h = spatial_max_pooling(self.comm, h, 17, 2, 2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)

        # h = spatial_max_pooling(self.comm, h, 18, 2, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)


        # h = spatial_max_pooling(self.comm, h, 19, 2, 2)

        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)
        print("My rank is ", self.comm.rank, "with shape", h.shape)

        return h

