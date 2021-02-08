import chainer
import cupy  as cp

import chainer.links as L

from chainer.function_hooks import TimerHook

x = cp.arange(1 * 3 * 10 * 10, dtype=cp.float32).reshape(1, 3, 10, 10)

l = L.Convolution2D(3, 100, 3).to_gpu(0)

hook = TimerHook()
with hook:
    for i in range(10):
        y = l(x)
hook.print_report()

