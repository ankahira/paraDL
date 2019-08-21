import chainer
import chainermn
from chainer import Chain
import chainer.functions as F
import chainer.links as L


class CosmoFlowMP(Chain):
    def __init__(self, comm):
        self.comm = comm
        super().__init__()
        with self.init_scope():
            self.Conv1 = L.Convolution3D(in_channels=1, out_channels=4, ksize=3, stride=1)
            self.Conv2 = L.Convolution3D(in_channels=16, out_channels=32, ksize=4, stride=1)
            self.Conv3 = L.Convolution3D(in_channels=32, out_channels=64, ksize=4, stride=2)
            self.Conv4 = L.Convolution3D(in_channels=64, out_channels=128, ksize=3, stride=1)
            self.Conv5 = L.Convolution3D(in_channels=128, out_channels=256, ksize=2, stride=1)
            self.Conv6 = L.Convolution3D(in_channels=256, out_channels=256, ksize=2, stride=1)
            self.Conv7 = L.Convolution3D(in_channels=256, out_channels=128, ksize=2, stride=1)
            self.FC1 = L.Linear(None, 2048)
            self.FC2 = L.Linear(None, 256)
            self.Output = L.Linear(None, 4)

    def forward(self, x, y):
        if self.comm.rank == 0:
            h = F.leaky_relu(self.Conv1(x[:, 0:1, :, :, :]))
        elif self.comm.rank == 1:
            h = F.leaky_relu(self.Conv1(x[:, 1:2, :, :, :]))
        elif self.comm.rank == 2:
            h = F.leaky_relu(self.Conv1(x[:, 2:3, :, :, :]))
        elif self.comm.rank == 3:
            h = F.leaky_relu(self.Conv1(x[:, 3:4, :, :, :]))

        hs = chainermn.functions.allgather(self.comm, h)
        h = F.concat(hs, axis=1)
        h = F.average_pooling_3d(h, ksize=2, stride=2)
        print(h.shape)
        h = F.leaky_relu(self.Conv2(h))
        h = F.average_pooling_3d(h, ksize=2, stride=2)
        h = F.leaky_relu(self.Conv3(h))
        h = F.leaky_relu(self.Conv4(h))
        h = F.leaky_relu(self.Conv5(h))
        h = F.leaky_relu(self.Conv6(h))
        h = F.leaky_relu(self.Conv7(h))
        h = F.leaky_relu(self.FC1(h))
        h = F.leaky_relu(self.FC2(h))
        h = self.Output(h)
        loss = F.mean_squared_error(h, y)
        chainer.report({'loss': loss}, self)
        return loss


