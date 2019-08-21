import chainer
import chainermn
import numpy as np
from chainer import Chain
import chainer.functions as F
import chainer.links as L


class ParallelConvolution3D(chainer.links.Convolution3D):
    def __init__(self, comm, in_channels, out_channels, *args, **kwargs):
        self.comm = comm
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(self._in_channel_size, self._out_channel_size, *args, **kwargs)

    def _channel_size(self, n_channel):
        # Return the size of the corresponding channels.
        n_proc = self.comm.size
        i_proc = self.comm.rank
        return n_channel // n_proc + (1 if i_proc < n_channel % n_proc else 0)

    @property
    def _in_channel_size(self):
        return self._channel_size(self.in_channels)

    @property
    def _out_channel_size(self):
        return self._channel_size(self.out_channels)

    @property
    def _channel_indices(self):
        # Return the indices of the corresponding channel.
        indices = np.arange(self.in_channels)
        indices = indices[indices % self.comm.size == 0] + self.comm.rank
        return [i for i in indices if i < self.in_channels]

    def __call__(self, x):
        x = x[:, self._channel_indices, :, :, :]
        y = super(ParallelConvolution3D, self).__call__(x)
        ys = chainermn.functions.allgather(self.comm, y)
        return F.concat(ys, axis=1)


class ParallelConvolutionCosmoFlow(Chain):

    def __init__(self, comm):
        super().__init__()
        self.comm = comm
        with self.init_scope():
            self.Conv1 = ParallelConvolution3D(comm, in_channels=4, out_channels=16, ksize=3, stride=1)
            self.Conv2 = ParallelConvolution3D(comm, in_channels=16, out_channels=32, ksize=4, stride=1)
            self.Conv3 = ParallelConvolution3D(comm, in_channels=32, out_channels=64, ksize=4, stride=2)
            self.Conv4 = ParallelConvolution3D(comm, in_channels=64, out_channels=128, ksize=3, stride=1)
            self.Conv5 = ParallelConvolution3D(comm, in_channels=128, out_channels=256, ksize=2, stride=1)
            self.Conv6 = ParallelConvolution3D(comm, in_channels=256, out_channels=256, ksize=2, stride=1)
            self.Conv7 = ParallelConvolution3D(comm, in_channels=256, out_channels=128, ksize=2, stride=1)
            self.FC1 = L.Linear(None, 2048)
            self.FC2 = L.Linear(None, 256)
            self.Output = L.Linear(None, 4)

    def forward(self, x, y):
        y = y.astype(np.float32)
        h = F.leaky_relu(self.Conv1(x))
        h = F.average_pooling_3d(h, ksize=2, stride=2)
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


