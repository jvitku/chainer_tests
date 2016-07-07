import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

# define array
##################################### links demo

lv = L.Linear(3, 2)  # f(x)=Wx+b         3 inputs 2 outputs
lv.zerograds()       # delete gradients

lvb = L.Linear(4, 2)
lvb.zerograds()

lh = L.Linear(2, 1)
lh.zerograds()

x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)) # data, minibatch of size 2 of data of D=3
x2 = Variable(np.array([[1, 2, 3, 4], [4, 5, 6, 7]], dtype=np.float32)) # data, minibatch of size 2 of data of D=4

##### use the network with lv layer
y = lh(lv(x))            # apply function
y.grad = np.ones((1, 1), dtype=np.float32)  # initialize gradients
y.backward()        # compute gradients of f

##### use the network with lvb layer on different data
y = lh(lvb(x2))            # apply function
y.grad = np.ones((1, 1), dtype=np.float32)  # initialize gradients
y.backward()        # compute gradients of f

print "y.data="
print y.data
