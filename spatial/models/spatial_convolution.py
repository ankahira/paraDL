import cupy as cp
import chainer
import chainermn
import chainer.functions as F


# We dont do an all reduce at the end of each layer.
# All reduce is done as in the case of data parallelism using the multi-node optimiser.

class SpatialConvolution2D(chainer.links.Convolution2D):
    def __init__(self, comm, in_channels, out_channels, *args, **kwargs):
        self.comm = comm
        self.in_channels = in_channels
        self.out_channels = out_channels
        super(SpatialConvolution2D, self).__init__(self.in_channels, self.out_channels, *args, **kwargs)
        self.n_proc = self.comm.size
        self.i_proc = self.comm.rank
        self.k_size = self.ksize

    def __call__(self, x):
        # Check if halo region is required then perform halo exchange before convolution
        # R is number of rows to exchange
        y = super(SpatialConvolution2D, self).__call__(x)
        return y


