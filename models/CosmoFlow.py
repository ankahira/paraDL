import chainer
from chainer import Chain
import chainer.functions as F
import chainer.links as L


class CosmoNet(Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.Conv1 = L.Convolution3D(in_channels=1, out_channels=16, ksize=3, stride=1)
            self.Conv2 = L.Convolution3D(in_channels=16, out_channels=32, ksize=4, stride=1)
            self.Conv3 = L.Convolution3D(in_channels=32, out_channels=64, ksize=4, stride=2)
            self.Conv4 = L.Convolution3D(in_channels=64, out_channels=128, ksize=3, stride=1)
            self.Conv5 = L.Convolution3D(in_channels=128, out_channels=256, ksize=2, stride=1)
            self.Conv6 = L.Convolution3D(in_channels=256, out_channels=256, ksize=2, stride=1)
            self.Conv7 = L.Convolution3D(in_channels=256, out_channels=128, ksize=2, stride=1)
            self.FC1 = L.Linear(None, 2048)
            self.FC2 = L.Linear(None, 256)
            self.Output = L.Linear(None, 3)

    def forward(self, x):
        h = F.relu(self.Conv1(x))
        h = F.average_pooling_3d(h, ksize=2, stride=2)
        h = F.relu(self.Conv2(h))
        h = F.average_pooling_3d(h, ksize=2, stride=2)
        h = F.relu(self.Conv3(h))
        h = F.relu(self.Conv4(h))
        h = F.relu(self.Conv5(h))
        h = F.relu(self.Conv6(h))
        h = F.relu(self.FC1(h))
        h = F.relu(self.FC2(h))
        if chainer.config.train:
            return self.Output(h)
        return F.softmax(self.Output(h))

