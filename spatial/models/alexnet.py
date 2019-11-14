import chainer
import chainer.functions as F
import chainer.links as L
import chainermn
import cupy as cp
from chainer.initializers import Constant
from chainermnx.functions.halo_exchange import halo_exchange


class AlexNet(chainer.Chain):

    def __init__(self, comm):
        super(AlexNet, self).__init__()
        self.comm = comm
        self.n_proc = self.comm.size
        with self.init_scope():
            cp.random.seed(0)
            self.conv1 = L.Convolution2D(None, 3, 3, pad=(0, 1), nobias=True, initialW=Constant(cp.random.rand()))
            cp.random.seed(1)
            self.conv2 = L.Convolution2D(None, 3, 3, pad=(0, 1), nobias=True, initialW=Constant(cp.random.rand()))
            cp.random.seed(2)
            self.conv3 = L.Convolution2D(None, 3, 3, pad=(0, 1), nobias=True, initialW=Constant(cp.random.rand()))
            cp.random.seed(3)
            self.conv4 = L.Convolution2D(None, 3, 3, pad=(0, 1), nobias=True, initialW=Constant(cp.random.rand()))
            cp.random.seed(4)
            self.conv5 = L.Convolution2D(None, 3, 3, pad=(0, 1), nobias=True, initialW=Constant(cp.random.rand()))
            cp.random.seed(5)
            self.fc6 = L.Linear(None, 4096, nobias=True, initialW=Constant(cp.random.rand()))
            cp.random.seed(6)
            self.fc7 = L.Linear(None, 4096, nobias=True, initialW=Constant(cp.random.rand()))
            cp.random.seed(7)
            self.fc8 = L.Linear(None, 1000, nobias=True, initialW=Constant(cp.random.rand()))

    def verification(self, h):
        hs = chainermn.functions.allgather(self.comm, h)
        h = F.concat(hs, -2)
        if self.comm.rank==0:
            print(h.shape)
            with open('spatial_output.txt', 'w') as file:
                for i in range(h.shape[-2]):
                    for j in range(h.shape[-1]):
                        print("%01.5f" % h[0, 0, i, j].array, file=file, end="")

                    print("\n", file=file)


    def forward(self, x, t):
        partions = cp.array_split(x, self.n_proc, -2)

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

        h = halo_exchange(self.comm, x, k_size=3, index=1)
        h = F.relu(self.conv1(h))
        self.verification(h)
        ## h = F.max_pooling_2d(h, ksize=4, stride=4)
        #h = halo_exchange(self.comm, h, k_size=3, index=2)
        #h = F.relu(self.conv2(h))
        ## h = F.max_pooling_2d(h, ksize=4, stride=4)
        #h = halo_exchange(self.comm, h, k_size=3, index=3)
        #h = F.relu(self.conv3(h))
        #h = halo_exchange(self.comm, h, k_size=3, index=4)
        #h = F.relu(self.conv4(h))
        #h = halo_exchange(self.comm, h, k_size=3, index=5)
        #h = F.relu(self.conv5(h))
        ## h = F.max_pooling_2d(h, ksize=4, stride=4)
        hs = chainermn.functions.allgather(self.comm, h)
        h = F.concat(hs, -2)

        #h = F.dropout(F.relu(self.fc6(h)))
        #h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss


