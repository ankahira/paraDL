import cupy as cp
import chainer
import chainermn
import chainer.functions as F


class SpatialConvolution2D(chainer.links.Convolution2D):
    def __init__(self, comm, in_channels, out_channels, *args, **kwargs):
        self.comm = comm
        self.in_channels = in_channels
        self.out_channels = out_channels
        super(SpatialConvolution2D, self).__init__(
            self.in_channels, self.out_channels, *args, **kwargs)
        self.n_proc = self.comm.size
        self.i_proc = self.comm.rank
        self.k_size = self.ksize

    def __call__(self, x):
        p_height = cp.asarray(x.shape)[-1]

        # Check if halo region is required then perform halo exchange before convolution
        # R is number of rows to exchange
        R = p_height % self.k_size

        if R != 0:
            if self.comm.rank == 0:
                # Rank 0 doesnt need to send anything
                # Receive halo region
                received_halo_region = chainermn.functions.recv(self.comm, rank=1)
                received_halo_region = received_halo_region.array

                # Concat to existing X
                if hasattr(x, "array"):
                    temp_x = x.array
                    x = chainer.Variable(cp.concatenate((temp_x,received_halo_region), axis=-1))
                else:
                    x = cp.concatenate((x, received_halo_region), axis=-1)

            elif self.comm.rank == 1:
                # Send Halo region
                halo_region_send = x[:, :, :, -R:]
                chainermn.functions.send(halo_region_send, self.comm, rank=0)

                # Receive halo region
                received_halo_region = chainermn.functions.recv(self.comm, rank=2)
                received_halo_region = received_halo_region.array

                # Concat to existing X
                # Concat to existing X
                if hasattr(x, "array"):
                    temp_x = x.array
                    x = chainer.Variable(cp.concatenate((temp_x, received_halo_region), axis=-1))
                else:
                    x = cp.concatenate((x, received_halo_region), axis=-1)

            elif self.comm.rank == 2:
                # Send Halo region
                halo_region_send = x[:, :, :, -R:]
                chainermn.functions.send(halo_region_send, self.comm, rank=1)

                # Receive halo region
                received_halo_region = chainermn.functions.recv(self.comm, rank=3)
                received_halo_region = received_halo_region.array

                # Concat to existing X
                # Concat to existing X
                if hasattr(x, "array"):
                    temp_x = x.array
                    x = chainer.Variable(cp.concatenate((temp_x, received_halo_region), axis=-1))
                else:
                    x = cp.concatenate((x, received_halo_region), axis=-1)

            elif self.comm.rank == 3:
                halo_region_send = x[:, :, :, -R:]
                chainermn.functions.send(halo_region_send, self.comm, rank=2)

                # Rank 3 doesnt need to receive

            else:
                print("Rank does not exist")

        y = super(SpatialConvolution2D, self).__call__(x)

        return y


class SpatialConvolution2DGather(chainer.links.Convolution2D):

    def __init__(self, comm, in_channels, out_channels, *args, **kwargs):
        self.comm = comm
        self.in_channels = in_channels
        self.out_channels = out_channels
        super(SpatialConvolution2DGather, self).__init__(
            self.in_channels, self.out_channels, *args, **kwargs)
        self.n_proc = self.comm.size
        self.i_proc = self.comm.rank
        self.k_size = self.ksize

    def __call__(self, x):
        p_height = cp.asarray(x.shape)[-1]

        # Check if halo region is required then perform halo exchange before convolution
        # R is number of rows to exchange
        R = p_height % self.k_size

        if R != 0:
            if self.comm.rank == 0:
                # Rank 0 doesnt need to send anything
                # Receive halo region
                received_halo_region = chainermn.functions.recv(self.comm, rank=1)
                received_halo_region = received_halo_region.array

                # Concat to existing X
                if hasattr(x, "array"):
                    temp_x = x.array
                    x = chainer.Variable(cp.concatenate((temp_x, received_halo_region), axis=-1))
                else:
                    x = cp.concatenate((x, received_halo_region), axis=-1)

            elif self.comm.rank == 1:
                # Send Halo region
                halo_region_send = x[:, :, :, -R:]
                chainermn.functions.send(halo_region_send, self.comm, rank=0)

                # Receive halo region
                received_halo_region = chainermn.functions.recv(self.comm, rank=2)
                received_halo_region = received_halo_region.array

                # Concat to existing X
                # Concat to existing X
                if hasattr(x, "array"):
                    temp_x = x.array
                    x = chainer.Variable(cp.concatenate((temp_x, received_halo_region), axis=-1))
                else:
                    x = cp.concatenate((x, received_halo_region), axis=-1)

            elif self.comm.rank == 2:
                # Send Halo region
                halo_region_send = x[:, :, :, -R:]
                chainermn.functions.send(halo_region_send, self.comm, rank=1)

                # Receive halo region
                received_halo_region = chainermn.functions.recv(self.comm, rank=3)
                received_halo_region = received_halo_region.array

                # Concat to existing X
                # Concat to existing X
                if hasattr(x, "array"):
                    temp_x = x.array
                    x = chainer.Variable(cp.concatenate((temp_x, received_halo_region), axis=-1))
                else:
                    x = cp.concatenate((x, received_halo_region), axis=-1)

            elif self.comm.rank == 3:
                halo_region_send = x[:, :, :, -R:]
                chainermn.functions.send(halo_region_send, self.comm, rank=2)

                # Rank 3 doesnt need to receive

            else:
                print("Rank does not exist")

        y = super(SpatialConvolution2DGather, self).__call__(x)
        ys = chainermn.functions.allgather(self.comm, y)
        return F.concat(ys, axis=-1)


