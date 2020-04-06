import chainer
import chainer.functions as F
import chainer.links as L

import chainer
import chainer.functions as F
import chainer.links as L

from chainer.function_hooks import TimerHook


class VGG(chainer.Chain):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1_1_hook = TimerHook()
        self.conv1_2_hook = TimerHook()

        self.conv2_1_hook = TimerHook()
        self.conv2_2_hook = TimerHook()

        self.conv3_1_hook = TimerHook()
        self.conv3_2_hook = TimerHook()
        self.conv3_3_hook = TimerHook()

        self.conv4_1_hook = TimerHook()
        self.conv4_2_hook = TimerHook()
        self.conv4_3_hook = TimerHook()

        self.conv5_1_hook = TimerHook()
        self.conv5_2_hook = TimerHook()
        self.conv5_3_hook = TimerHook()

        self.fc6_hook = TimerHook()
        self.fc7_hook = TimerHook()
        self.fc8_hook = TimerHook()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3, 1, 1)
            self.conv1_2 = L.Convolution2D(64, 64, 3, 1, 1)

            self.conv2_1 = L.Convolution2D(64, 128, 3, 1, 1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, 1, 1)

            self.conv3_1 = L.Convolution2D(128, 256, 3, 1, 1)
            self.conv3_2 = L.Convolution2D(256, 256, 3, 1, 1)
            self.conv3_3 = L.Convolution2D(256, 256, 3, 1, 1)

            self.conv4_1 = L.Convolution2D(256, 512, 3, 1, 1)
            self.conv4_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv4_3 = L.Convolution2D(512, 512, 3, 1, 1)

            self.conv5_1 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_3 = L.Convolution2D(512, 512, 3, 1, 1)

            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(4096, 4096)
            self.fc8 = L.Linear(4096, 1000)

    def forward(self, x):
        with self.conv1_1_hook:
            h = F.relu(self.conv1_1(x))
        with self.conv1_2_hook:
            h = F.relu(self.conv1_2(h))
            h = F.max_pooling_2d(h, 2, 2)

        with self.conv2_1_hook:
            h = F.relu(self.conv2_1(h))
        with self.conv2_2_hook:
            h = F.relu(self.conv2_2(h))
            h = F.max_pooling_2d(h, 2, 2)

        with self.conv3_1_hook:
            h = F.relu(self.conv3_1(h))
        with self.conv3_2_hook:
            h = F.relu(self.conv3_2(h))
        with self.conv3_3_hook:
            h = F.relu(self.conv3_3(h))
            h = F.max_pooling_2d(h, 2, 2)

        with self.conv4_1_hook:
            h = F.relu(self.conv4_1(h))
        with self.conv4_2_hook:
            h = F.relu(self.conv4_2(h))
        with self.conv4_3_hook:
            h = F.relu(self.conv4_3(h))
            h = F.max_pooling_2d(h, 2, 2)

        with self.conv5_1_hook:
            h = F.relu(self.conv5_1(h))
        with self.conv5_2_hook:
            h = F.relu(self.conv5_2(h))
        with self.conv5_3_hook:
            h = F.relu(self.conv5_3(h))
            h = F.max_pooling_2d(h, 2, 2)

        with self.fc6_hook:
            h = F.dropout(F.relu(self.fc6(h)))

        with self.fc7_hook:
            h = F.dropout(F.relu(self.fc7(h)))

        with self.fc8_hook:
            h = self.fc8(h)
        return h














