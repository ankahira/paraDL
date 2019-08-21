'''
Cosmoflow Single Node.

This is an attempt at some sort of pipeline model where several layers are placed in one GPU.
This model is for 4 GPUs and the model is spread across them all.
No communication of MPI implemented here. The layer parameters are simply copied to the next GPU.

This model doesnt work with the big dataset(4 * 512 * 512 * 512 )

'''

import chainer
from chainer import Chain
import chainer.functions as F
import chainer.links as L


class PipelineCosmoFlow(Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.Conv1 = L.Convolution3D(in_channels=4, out_channels=16, ksize=3, stride=1).to_gpu(0)

            self.Conv2 = L.Convolution3D(in_channels=16, out_channels=32, ksize=4, stride=1).to_gpu(1)
            self.Conv3 = L.Convolution3D(in_channels=32, out_channels=64, ksize=4, stride=2).to_gpu(1)
            self.Conv4 = L.Convolution3D(in_channels=64, out_channels=128, ksize=3, stride=1).to_gpu(1)

            self.Conv5 = L.Convolution3D(in_channels=128, out_channels=256, ksize=2, stride=1).to_gpu(2)
            self.Conv6 = L.Convolution3D(in_channels=256, out_channels=256, ksize=2, stride=1).to_gpu(2)
            self.Conv7 = L.Convolution3D(in_channels=256, out_channels=128, ksize=2, stride=1).to_gpu(2)

            self.FC1 = L.Linear(None, 2048).to_gpu(3)
            self.FC2 = L.Linear(None, 256).to_gpu(3)
            self.Output = L.Linear(None, 4).to_gpu(3)

    def forward(self, x, y):
        h = F.leaky_relu(self.Conv1(x))

        h = F.copy(h, 1)

        h = F.average_pooling_3d(h, ksize=2, stride=2)
        h = F.leaky_relu(self.Conv2(h))
        h = F.average_pooling_3d(h, ksize=2, stride=2)
        h = F.leaky_relu(self.Conv3(h))
        h = F.leaky_relu(self.Conv4(h))

        h = F.copy(h, 2)

        h = F.leaky_relu(self.Conv5(h))
        h = F.leaky_relu(self.Conv6(h))
        h = F.leaky_relu(self.Conv7(h))

        h = F.copy(h, 3)

        h = F.leaky_relu(self.FC1(h))
        h = F.leaky_relu(self.FC2(h))
        h = self.Output(h)

        h = F.copy(h, 0)
        loss = F.mean_squared_error(h, y)
        chainer.report({'loss': loss}, self)
        return loss


