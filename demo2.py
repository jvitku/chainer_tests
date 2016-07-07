import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class MyProc(object):
    def __init__(self):
        self.l1 = L.Linear(4, 3)
        self.l2 = L.Linear(3, 2)

    def forward(self, x):
        h = self.l1(x)
        return self.l2(h)


class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(4, 3),
            l2=L.Linear(3, 2),
        )

    def __call__(self, x):
        h = self.l1(x)
        return self.l2(h)

x = Variable(np.array([[1, 2, 3, 4], [4, 5, 6, 7]], dtype=np.float32)) # data, minibatch of size 2 of data of D=4


model = MyChain()                  # instance of the network
optimizer = optimizers.SGD()    # given gradients, optimize weights by..
optimizer.setup(model)

############# way 1: compute gradients manually, run the optimizer

model.zerograds()
y = model(x)
y.grad = np.ones((2, 2), dtype=np.float32) # compute gradients here manually
y.backward()
optimizer.update()



