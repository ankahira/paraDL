import time
import chainer
import chainer.functions as F
import chainer.links as L

from chainer.function_hooks import TimerHook


class AlexNet(chainer.Chain):

    insize = 226

    def __init__(self):
        super(AlexNet, self).__init__()

        # For layer by layer
        self.conv1_hook = TimerHook()
        self.conv2_hook = TimerHook()
        self.conv3_hook = TimerHook()
        self.conv4_hook = TimerHook()
        self.conv5_hook = TimerHook()
        self.fc6_hook = TimerHook()
        self.fc7_hook = TimerHook()
        self.fc8_hook = TimerHook()

        # For blocks
        self.Block_1_hook = TimerHook()
        self.Block_2_hook = TimerHook()



        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 96, 11, stride=4)
            self.conv2 = L.Convolution2D(None, 256, 5, pad=2)
            self.conv3 = L.Convolution2D(None, 384, 3, pad=1)
            self.conv4 = L.Convolution2D(None, 384, 3, pad=1)
            self.conv5 = L.Convolution2D(None, 256, 3, pad=1)
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, 1000)

    # Block analysis
    def forward(self, x):
        with self.Block_1_hook:
            h = F.relu(self.conv1(x))
            h = F.max_pooling_2d(h, ksize=3, stride=2)
            h = F.relu(self.conv2(h))
            h = F.max_pooling_2d(h, ksize=3, stride=2)
            h = F.relu(self.conv3(h))
            h = F.relu(self.conv4(h))
        with self.Block_2_hook:
            h = F.relu(self.conv5(h))
            h = F.max_pooling_2d(h, ksize=3, stride=2)
            h = F.relu(self.fc6(h))
            h = F.dropout(h, ratio=0.5)
            h = F.relu(self.fc7(h))
            h = F.dropout(h, ratio=0.5)
            h = self.fc8(h)

        return h

    # Layer by layer
    # def forward(self, x):
    #     with self.conv1_hook:
    #         h = F.relu(self.conv1(x))
    #         h = F.max_pooling_2d(h, ksize=3, stride=2)
    #     with self.conv2_hook:
    #         h = F.relu(self.conv2(h))
    #         h = F.max_pooling_2d(h, ksize=3, stride=2)
    #     with self.conv3_hook:
    #         h = F.relu(self.conv3(h))
    #     with self.conv4_hook:
    #         h = F.relu(self.conv4(h))
    #     with self.conv5_hook:
    #         h = F.relu(self.conv5(h))
    #         h = F.max_pooling_2d(h, ksize=3, stride=2)
    #     with self.fc6_hook:
    #         h = F.relu(self.fc6(h))
    #         h = F.dropout(h, ratio=0.5)
    #     with self.fc7_hook:
    #         h = F.relu(self.fc7(h))
    #         h = F.dropout(h, ratio=0.5)
    #     with self.fc8_hook:
    #         h = self.fc8(h)
    #
    #     return h