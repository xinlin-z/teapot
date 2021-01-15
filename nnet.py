import random
import numpy as np
import dataset as ds
import func


class mlp():
    """Feedforward Fully Connected Neural Network."""

    def __init__(self, sizes, neuron, output, costfunc):
        """ sizes: a tuple like (784,100,10) """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.b = [np.random.randn(x,1).astype(ds.DTYPE) for x in sizes[1:]]
        self.w = [np.random.randn(x,y).astype(ds.DTYPE)
                        for x,y in zip(sizes[1:],sizes[:-1])]
        self.a = []  # layered activation value, including input layer.
        self.z = []  # layered weighted input
        # config check for neuron, output, costfunc
        if neuron == 'sigmoid':
            self.af = func.sigmoid
            self.afdz = func.sigmoid_dz
            if costfunc == 'quadratic':
                assert output == 'sigmoid'
                self.cf = func.quadratic
                self.cfda = func.quadratic_da
            elif costfunc == 'cross_entropy':
                assert output == 'sigmoid'
                self.cf = func.cross_entropy
            elif costfunc == 'log_likelihood':
                assert output == 'softmax'
                self.cf = func.log_likelihood
            else:
                raise ValueError('costfunc not supported with sigmoid')
        elif neuron == 'tanh':
            self.af = func.tanh
            self.afdz = func.tanh_dz
            if costfunc == 'cross_entropy':
                assert output == 'sigmoid'
                self.cf = func.cross_entropy
            elif costfunc == 'log_likelihood':
                assert output == 'softmax'
                self.cf = func.log_likelihood
            else:
                raise ValueError('costfunc not supported with tanh')
        else:
            raise ValueError('neuron not supported')
        self.neuron = neuron
        self.costfunc = costfunc
        self.output = output

    def ff(self, a):
        """ feedforward """
        for w,b in zip(self.w[:-1], self.b[:-1]):
            a = self.af(func.z(w,a,b))
        z = func.z(self.w[-1], a, self.b[-1])
        if self.output == 'softmax':
            return func.softmax(z)
        else:  # sigmoid
            return func.sigmoid(z)

    def cost(self, y, x):
        """total averaged cost over data pairs"""
        return self.cf(y, self.ff(x))/y.shape[1]

    def anz(self, a):
        """compute layered a and z,
        self.a[0] is the data of input layer"""
        self.z.clear()
        self.a.clear()
        self.a.append(a)
        for w,b in zip(self.w[:-1], self.b[:-1]):
            self.z.append(func.z(w,self.a[-1],b))
            self.a.append(self.af(self.z[-1]))
        self.z.append(func.z(self.w[-1], self.a[-1], self.b[-1]))
        if self.output == 'softmax':
            self.a.append(func.softmax(self.z[-1]))
        else:
            self.a.append(self.af(self.z[-1]))

    def backprop(self, x, y):
        nabla_w = [np.zeros_like(w) for w in self.w]
        nabla_b = [np.zeros_like(b) for b in self.b]
        self.anz(x)
        # the output layer
        if ((self.output=='sigmoid' and self.costfunc=='cross_entropy') or
                (self.output=='softmax' and self.costfunc=='log_likelihood')):
            delta = self.a[-1] - y  # a little speed up
        else:
            delta = self.cfda(y, self.a[-1])*self.afdz(self.z[-1])
        nabla_b[-1] = np.sum(delta,axis=1).reshape(nabla_b[-1].shape)
        nabla_w[-1] = delta @ self.a[-2].T
        # the rest layer, backward
        for i in range(self.num_layers-2, 0, -1):
            delta = self.w[i].T @ delta * self.afdz(self.z[i-1])
            nabla_b[i-1] = np.sum(delta,axis=1,keepdims=True)
            nabla_w[i-1] = delta @ self.a[i-1].T
        return nabla_w, nabla_b

    def gd(self, x, y, eta):
        """ gradient descent, matrix-based  """
        nabla_w = [np.zeros_like(w) for w in self.w]
        nabla_b = [np.zeros_like(b) for b in self.b]
        num = x.shape[1]
        delta_w, delta_b = self.backprop(x, y)
        self.w = [x-eta*w/num for x,w in zip(self.w, delta_w)]
        self.b = [x-eta*b/num for x,b in zip(self.b, delta_b)]



