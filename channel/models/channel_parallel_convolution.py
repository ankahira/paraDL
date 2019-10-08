from __future__ import print_function

import chainer
import chainer.functions as F
import chainermn.functions
import numpy as np

# This is really messy but will be improved at a later date.(or maybe never)
# Doing a lot of stuff at __init__ so as to initialise class depending on channels.

# Each node receives 1 input channel, keep F filters but each filter only have dimension 1xKxK
# so that it create F output channel also with the size of FxW'xH'.


class ChannelParallelConvolution2D(chainer.links.Convolution2D):
    def __init__(self, comm, in_channels, out_channels, *args, **   kwargs):
        if in_channels <= comm.size:
            self.comm = comm
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.parallel_execute = False
            super(ChannelParallelConvolution2D, self).__init__(self.in_channels, self.out_channels, *args, **kwargs)
        else:
            self.comm = comm
            indices = np.arange(in_channels)
            indices = indices[indices % self.comm.size == 0] + self.comm.rank
            self._channel_indices = [i for i in indices if i < in_channels]
            self.in_channels = len(self._channel_indices)
            self.out_channels = out_channels
            self.parallel_execute = True
            super(ChannelParallelConvolution2D, self).__init__(self.in_channels, self.out_channels, *args, **kwargs)

    def __call__(self, x, *args, **kwargs):
        if self.parallel_execute:
            # Each process gets C/P channels
            x = x[:, self._channel_indices, :, :]
            y = super(ChannelParallelConvolution2D, self).__call__(x)
            temp_array = y.array
            ys = chainermn.communicators.mpi_communicator_base.MpiCommunicatorBase.allreduce(self.comm, temp_array)
            ys = chainer.Variable(ys)
            return ys
        else:
            y = super(ChannelParallelConvolution2D, self).__call__(x)
            return y

