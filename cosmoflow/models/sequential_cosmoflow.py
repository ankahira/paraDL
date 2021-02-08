import chainer
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from chainer.links import Convolution3D


class CosmoFlow(Chain):
    def __init__(self):
        super(CosmoFlow, self).__init__()
        with self.init_scope():
            self.Conv1 = Convolution3D(in_channels=4, out_channels=16, ksize=3, stride=1, nobias=True)
            self.Conv2 = Convolution3D(in_channels=16, out_channels=32, ksize=3, stride=1, nobias=True)
            self.Conv3 = Convolution3D(in_channels=32, out_channels=64, ksize=3, stride=1, pad=1, nobias=True)
            self.Conv4 = Convolution3D(in_channels=64, out_channels=128, ksize=3, stride=2, pad=1, nobias=True)
            self.Conv5 = Convolution3D(in_channels=128, out_channels=256, ksize=2, stride=1, pad=1, nobias=True)
            self.Conv6 = Convolution3D(in_channels=256, out_channels=256, ksize=2, stride=1, pad=1, nobias=True)
            self.Conv7 = Convolution3D(in_channels=256, out_channels=128, ksize=2, stride=1, pad=1, nobias=True)
            self.FC1 = L.Linear(None, 2048)
            self.FC2 = L.Linear(None, 256)
            self.Output = L.Linear(None, 4)

    def forward(self, x, y):
        h = F.leaky_relu(self.Conv1(x))
        h = F.average_pooling_3d(h, ksize=2, stride=2)
        h = F.leaky_relu(self.Conv2(h))
        h = F.average_pooling_3d(h, ksize=2, stride=2)
        h = F.leaky_relu(self.Conv3(h))
        h = F.average_pooling_3d(h, ksize=2, stride=2)
        h = F.leaky_relu(self.Conv4(h))
        h = F.average_pooling_3d(h, ksize=2, stride=2)
        h = F.leaky_relu(self.Conv5(h))
        h = F.average_pooling_3d(h, ksize=2, stride=2)
        h = F.leaky_relu(self.Conv6(h))
        h = F.average_pooling_3d(h, ksize=2, stride=2)
        h = F.leaky_relu(self.Conv7(h))
        h = F.leaky_relu(self.FC1(h))
        h = F.leaky_relu(self.FC2(h))
        h = self.Output(h)
        loss = F.mean_squared_error(h, y)
        chainer.report({'loss': loss}, self)
        return loss



