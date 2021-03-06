import chainer
import chainer.functions as F
from chainermnx.links import ChannelParallelConvolution2D, ChannelParallelFC


class AlexNet(chainer.Chain):

    def __init__(self, comm):
        super(AlexNet, self).__init__()
        self.comm = comm
        with self.init_scope():
            self.conv1 = ChannelParallelConvolution2D(comm, 3, 96, 11, stride=4)
            self.conv2 = ChannelParallelConvolution2D(comm, 96, 256, 5, pad=2)
            self.conv3 = ChannelParallelConvolution2D(comm, 256, 384, 3, pad=1)
            self.conv4 = ChannelParallelConvolution2D(comm, 384, 384, 3, pad=1)
            self.conv5 = ChannelParallelConvolution2D(comm, 384, 256, 3, pad=1)
            self.fc6 = ChannelParallelFC(comm, None, 4096)
            self.fc7 = ChannelParallelFC(comm, 4096, 4096)
            self.fc8 = ChannelParallelFC(comm, 4096, 1000)

    def forward(self, x, t):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(h, ksize=3, stride=2)

        h = F.relu(self.fc6(h))
        h = F.dropout(h, ratio=0.5)
        h = F.relu(self.fc7(h))
        h = F.dropout(h, ratio=0.5)
        h = self.fc8(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss
