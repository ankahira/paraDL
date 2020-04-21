import chainer
from chainer import Chain
import chainer.functions as F
import chainermnx.functions as FX
import chainer.links as L
from chainermnx.links import SpatialConvolution3D
from chainer.links import Convolution3D
import cupy as cp
import chainermnx
import chainermn


class CosmoFlow(Chain):
    def __init__(self, comm, out):
        self.comm = comm
        self.n_proc = self.comm.size
        self.out = out
        super(CosmoFlow, self).__init__()
        with self.init_scope():
            self.Conv1 = SpatialConvolution3D(comm=comm, out=self.out, index=1, in_channels=4, out_channels=16, ksize=3, stride=1, nobias=True)
            self.Conv2 = SpatialConvolution3D(comm=comm, out=self.out, index=2, in_channels=16, out_channels=32, ksize=3, stride=1, nobias=True)

            # Only the first 2 layers are in parallel
            self.Conv3 = Convolution3D(in_channels=32, out_channels=64, ksize=3, stride=1, pad=1, nobias=True)
            self.Conv4 = Convolution3D(in_channels=64, out_channels=128, ksize=3, stride=2, pad=1, nobias=True)
            self.Conv5 = Convolution3D(in_channels=128, out_channels=256, ksize=2, stride=1, pad=1, nobias=True)
            self.Conv6 = Convolution3D(in_channels=256, out_channels=256, ksize=2, stride=1, pad=1, nobias=True)
            self.Conv7 = Convolution3D(in_channels=256, out_channels=128, ksize=2, stride=1, pad=1, nobias=True)
            self.FC1 = L.Linear(None, 2048)
            self.FC2 = L.Linear(None, 256)
            self.Output = L.Linear(None, 4)

    def forward(self, x, y):
        # I think we need to do bcast because only one rank loads data and others dont have y values.        #
        partions = cp.array_split(x, self.comm.size, -2)
        # This part needs fixing. Probably all conditions are not checked
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
        # y = chainermn.functions.bcast(self.comm, y, 0)
        h = FX.halo_exchange_3d(self.comm, x, k_size=3, index=1, pad=0, out=self.out)
        h = F.leaky_relu(self.Conv1(h))
        h = F.average_pooling_3d(h, ksize=2, stride=2)
        h = FX.halo_exchange_3d(self.comm, h, k_size=3, index=2, pad=0, out=self.out)
        h = F.leaky_relu(self.Conv2(h))
        h = F.average_pooling_3d(h, ksize=2, stride=2)
        hs = chainermnx.functions.spatialallgather(self.comm, h, self.out)
        h = F.concat(hs, -2)
        h = F.leaky_relu(self.Conv3(h))
        h = F.average_pooling_3d(h, ksize=2, stride=2)
        h = F.leaky_relu(self.Conv4(h))
        h = F.average_pooling_3d(h, ksize=2, stride=2)
        h = F.leaky_relu(self.Conv5(h))
        h = F.average_pooling_3d(h, ksize=2, stride=2)
        h = F.leaky_relu(self.Conv6(h))
        h = F.average_pooling_3d(h, ksize=2, stride=2)
        h = F.leaky_relu(self.Conv7(h))
        h = F.leaky_relu(self.FC1(h))
        h = F.leaky_relu(self.FC2(h))
        h = self.Output(h)
        loss = F.mean_squared_error(h, y)
        chainer.report({'loss': loss}, self)
        return loss



