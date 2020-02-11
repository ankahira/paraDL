from __future__ import print_function

import chainer
import chainer.functions as F
import chainermn.functions
import numpy as np

# Assume there are 4 filters
# Each node receives C input channel, keeps 1 filters. Each filter have dimension CxKxK  create 1
# output channel (total p output channels) with the size 1xW'xH'
# That is, in the forward phase of each node k only calculate one value of y_j (where j=k) because it
# keep on 1 filter wij so that they have to perform an allgather operation to collect all
# the y_j before pass it to the next layer.


class FilterParallelConvolution2D(chainer.links.Convolution2D):
    def __init__(self, comm, in_channels, out_channels, *args, **kwargs):
        self.comm = comm
        self.in_channels = in_channels
        self.filters = out_channels
        indices = np.arange(self.filters)
        indices = indices[indices % self.comm.size == 0] + self.comm.rank
        self.filter_indices = [i for i in indices if i < self.filters]
        self.new_filters = len(self.filter_indices)
        super(FilterParallelConvolution2D, self).__init__(self.in_channels, self.new_filters, *args, **kwargs)

    def __call__(self, x):
        y = super(FilterParallelConvolution2D, self).__call__(x)
        ys = chainermn.functions.allgather(self.comm, y)
        # Backward will be invoked as well as the ordinary chainer functions,
        # where gradients are reduced to each process
        return F.concat(ys, axis=1)


class FilterParallelFC(chainer.links.Linear):
    def __init__(self, comm, in_size, out_size):
        self.comm = comm
        self.in_size = in_size
        self.out_size = out_size
        indices = np.arange(self.out_size)
        indices = indices[indices % self.comm.size == 0] + self.comm.rank
        self.out_indices = [i for i in indices if i < self.out_size]
        self.new_out_size = len(self.out_indices)
        super(FilterParallelFC, self).__init__(self.in_size, self.new_out_size)

    def __call__(self, x):
        y = super(FilterParallelFC, self).__call__(x)
        ys = chainermn.functions.allgather(self.comm, y)
        return F.concat(ys, axis=1)

