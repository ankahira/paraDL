import chainer
import chainer.functions as F
import chainer.links as L

P = 16

class VGG(chainer.Chain):
    def __init__(self):
        super(VGG, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64//P, 3, 1, 1)
            self.conv1_2 = L.Convolution2D(None, 64//P, 3, 1, 1)

            self.conv2_1 = L.Convolution2D(None, 128//P, 3, 1, 1)
            self.conv2_2 = L.Convolution2D(None, 128//P, 3, 1, 1)

            self.conv3_1 = L.Convolution2D(None, 256//P, 3, 1, 1)
            self.conv3_2 = L.Convolution2D(None, 256//P, 3, 1, 1)
            self.conv3_3 = L.Convolution2D(None, 256//P, 3, 1, 1)

            self.conv4_1 = L.Convolution2D(None, 512//P, 3, 1, 1)
            self.conv4_2 = L.Convolution2D(None, 512//P, 3, 1, 1)
            self.conv4_3 = L.Convolution2D(None, 512//P, 3, 1, 1)

            self.conv5_1 = L.Convolution2D(None, 512//P, 3, 1, 1)
            self.conv5_2 = L.Convolution2D(None, 512//P, 3, 1, 1)
            self.conv5_3 = L.Convolution2D(None, 512//P, 3, 1, 1)

            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(4096, 4096)
            self.fc8 = L.Linear(4096, 1000)

    def forward(self, x):

        h = F.relu(self.conv1_1(x))
        h = F.concat((h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h), axis=1)

        h = F.relu(self.conv1_2(h))
        h = F.concat((h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h), axis=1)

        h = F.max_pooling_2d(h, 2, 2)

        h = F.relu(self.conv2_1(h))
        h = F.concat((h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h), axis=1)

        h = F.relu(self.conv2_2(h))
        h = F.concat((h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h), axis=1)

        h = F.max_pooling_2d(h, 2, 2)

        h = F.relu(self.conv3_1(h))
        h = F.concat((h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h), axis=1)

        h = F.relu(self.conv3_2(h))
        h = F.concat((h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h), axis=1)

        h = F.relu(self.conv3_3(h))
        h = F.concat((h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h), axis=1)

        h = F.max_pooling_2d(h, 2, 2)

        h = F.relu(self.conv4_1(h))
        h = F.concat((h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h), axis=1)

        h = F.relu(self.conv4_2(h))
        h = F.concat((h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h), axis=1)

        h = F.relu(self.conv4_3(h))
        h = F.concat((h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h), axis=1)

        h = F.max_pooling_2d(h, 2, 2)

        h = F.relu(self.conv5_1(h))
        h = F.concat((h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h), axis=1)

        h = F.relu(self.conv5_2(h))
        h = F.concat((h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h), axis=1)

        h = F.relu(self.conv5_3(h))
        h = F.concat((h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h), axis=1)

        h = F.max_pooling_2d(h, 2, 2)

        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)
        return h













