from chainer import Chain
import chainer.functions as F
import chainer.links as L


class CosmoNet(Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.Conv1 = L.Convolution3D(None, out_channels=2, ksize=3, stride=1)
            self.Conv2 = L.Convolution3D(None, out_channels=12, ksize=4, stride=1)
            self.Conv3 = L.Convolution3D(None, out_channels=64, ksize=4, stride=2)
            self.Conv4 = L.Convolution3D(None, out_channels=64, ksize=3, stride=1)
            self.Conv5 = L.Convolution3D(None, out_channels=128, ksize=2, stride=1)
            self.Conv6 = L.Convolution3D(None, out_channels=128, ksize=2, stride=1)
            self.FC1 = L.Linear(None, 1024)
            self.FC2 = L.Linear(None, 256)
            self.Output = L.Linear(None, 2)

    def forward(self, x):
        h = F.leaky_relu(self.Conv1(x))
        h = F.average_pooling_3d(h, ksize=2, stride=2)
        h = F.leaky_relu(self.Conv2(h))
        h = F.average_pooling_3d(h, ksize=2, stride=2)
        h = F.leaky_relu(self.Conv3(h))
        h = F.leaky_relu(self.Conv4(h))
        h = F.leaky_relu(self.Conv5(h))
        h = F.leaky_relu(self.Conv6(h))
        h = F.leaky_relu(self.FC1(h))
        h = F.leaky_relu(self.FC2(h))
        return self.Output(h)

