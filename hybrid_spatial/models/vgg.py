import cupy as cp
import chainer
import chainer.functions as F
import chainer.links as L
import chainermnx.functions as FX
import chainermnx.links as LX

import chainermnx


class VGG(chainer.Chain):
    def __init__(self, comm, out):
        super(VGG, self).__init__()
        self.comm = comm
        self.out = out
        self.n_proc = self.comm.size
        with self.init_scope():
            self.conv1_1 = LX.Convolution2D(comm, self.out, 1, 3, 64, 3, pad=(0, 1))
            self.conv1_2 = LX.Convolution2D(comm, self.out, 2, 64, 64, 3, pad=(0, 1))

            self.conv2_1 = LX.Convolution2D(comm, self.out, 3, 64, 128, 3, pad=(0, 1))
            self.conv2_2 = LX.Convolution2D(comm, self.out, 4, 128, 128, 3, pad=(0, 1))

            self.conv3_1 = LX.Convolution2D(comm, self.out, 5, 128, 256, 3, pad=(0, 1))
            self.conv3_2 = LX.Convolution2D(comm, self.out, 6, 256, 256, 3, pad=(0, 1))
            self.conv3_3 = LX.Convolution2D(comm, self.out, 7, 256, 256, 3, pad=(0, 1))

            self.conv4_1 = LX.Convolution2D(comm, self.out, 8, 256, 512, 3, pad=(0, 1))
            self.conv4_2 = LX.Convolution2D(comm, self.out, 9, 512, 512, 3, pad=(0, 1))
            self.conv4_3 = LX.Convolution2D(comm, self.out, 10, 512, 512, 3, pad=(0, 1))

            self.conv5_1 = LX.Convolution2D(comm, self.out, 11, 512, 512, 3, pad=(0, 1))
            self.conv5_2 = LX.Convolution2D(comm, self.out, 12, 512, 512, 3, pad=(0, 1))
            self.conv5_3 = LX.Convolution2D(comm, self.out, 13, 512, 512, 3, pad=(0, 1))

            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(4096, 4096)
            self.fc8 = L.Linear(4096, 1000)

    def forward(self, x):
        partions = cp.array_split(x, self.comm.size, -2)
        if self.comm.rank == 0:
            x = partions[0]
        elif self.comm.rank == 1:
            x = partions[1]
        elif self.comm.rank == 2:
            x = partions[2]
        elif self.comm.rank == 3:
            x = partions[3]
        else:
            print("Rank does not exist")

        h = FX.halo_exchange(self.comm, x, k_size=3, index=11, pad=1,  out=self.out)
        h = F.relu(self.conv1_1(h))
        h = FX.halo_exchange(self.comm, h, k_size=3, index=12, pad=1 , out=self.out)
        h = F.relu(self.conv1_2(h))
        h = FX.pooling_halo_exchange(self.comm, h, k_size=2, index=11)
        h = F.max_pooling_2d(h, 2, 2)

        h = FX.halo_exchange(self.comm, h, k_size=3, index=21, pad=1, out=self.out)
        h = F.relu(self.conv2_1(h))
        h = FX.halo_exchange(self.comm, h, k_size=3, index=22, pad=1, out=self.out)
        h = F.relu(self.conv2_2(h))
        h = FX.pooling_halo_exchange(self.comm, h, k_size=2, index=12)
        h = F.max_pooling_2d(h, 2, 2)

        h = FX.halo_exchange(self.comm, h, k_size=3, index=31, pad=1, out=self.out)
        h = F.relu(self.conv3_1(h))
        h = FX.halo_exchange(self.comm, h, k_size=3, index=32, pad=1, out=self.out)
        h = F.relu(self.conv3_2(h))
        h = FX.halo_exchange(self.comm, h, k_size=3, index=33, pad=1, out=self.out)
        h = F.relu(self.conv3_3(h))
        h = FX.pooling_halo_exchange(self.comm, h, k_size=2, index=13)
        h = F.max_pooling_2d(h, 2, 2)

        h = FX.halo_exchange(self.comm, h, k_size=3, index=41, pad=1, out=self.out)
        h = F.relu(self.conv4_1(h))
        h = FX.halo_exchange(self.comm, h, k_size=3, index=42, pad=1, out=self.out)
        h = F.relu(self.conv4_2(h))
        h = FX.halo_exchange(self.comm, h, k_size=3, index=43, pad=1, out=self.out)
        h = F.relu(self.conv4_3(h))
        h = FX.pooling_halo_exchange(self.comm, h, k_size=2, index=14)
        h = F.max_pooling_2d(h, 2, 2)

        h = FX.halo_exchange(self.comm, h, k_size=3, index=51, pad=1, out=self.out)
        h = F.relu(self.conv5_1(h))
        h = FX.halo_exchange(self.comm, h, k_size=3, index=52, pad=1, out=self.out)
        h = F.relu(self.conv5_2(h))
        h = FX.halo_exchange(self.comm, h, k_size=3, index=53, pad=1, out=self.out)
        h = F.relu(self.conv5_3(h))
        h = FX.pooling_halo_exchange(self.comm, h, k_size=2, index=15)
        h = F.max_pooling_2d(h, 2, 2)

        hs = chainermnx.functions.spatialallgather(self.comm, h, self.out)
        h = F.concat(hs, -2)

        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)

        return h

