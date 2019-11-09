import chainer
from chainer import function_node

class DummyFunction(function_node.FunctionNode):

    def forward(self, inputs):
        x, = inputs

        return x,

    def backward(self, indixes, grad_outputs):
        gy, = grad_ouputs
        print("we got here")
        print(gy)
        return gy,

def dummy_function(x):
    func = DummyFunction()
    return func.apply(x,)[0]

