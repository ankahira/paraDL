from __future__ import print_function

import chainer
import chainer.functions as F
import chainermn.functions
import numpy as np


class ParallelConvolution2D(chainer.links.Convolution2D):
    def __init__(self, comm, in_channels, out_channels, *args, **kwargs):
        self.comm = comm
        self.in_channels = in_channels
        self.out_channels = out_channels
        super(ParallelConvolution2D, self).__init__(
            self._in_channel_size, self._out_channel_size, *args, **kwargs)

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
        print(x.shape)
        print("***********************************We got HERE !!!*******************************************")
        x = x[:, self._channel_indices, :, :]
        y = super(ParallelConvolution2D, self).__call__(x)
        ys = chainermn.functions.allgather(self.comm, y)
        return F.concat(ys, axis=1)
