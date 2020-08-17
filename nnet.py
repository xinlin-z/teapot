import random
import numpy as np
import dataset as ds
import func


class fffnn():
    """FeedForward Fully Connected Neural Network."""

    def __init__(self, sizes):
        """ sizes: a tuple like (784, 15, 10),
            af: activation function,
            afdz: derivative of af for z, 
            cf: cost function for a single input,
            cfda: derivative of cf for a. """
        self.af = func.sigmoid
        self.afdz = func.sigmoid_dz
        self.cf = func.qcost_x
        self.cfda = func.qcost_x_da
        #
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.b = [np.random.randn(x,1).astype(ds.DTYPE) for x in sizes[1:]]
        self.w = [np.random.randn(x,y).astype(ds.DTYPE) 
                        for x,y in zip(sizes[1:],sizes[:-1])]
        self.a = []  # layered activation value, including input layer.
        self.z = []  # layered weighted input

    def ff(self, a):
        """ feedforward """
        for w,b in zip(self.w, self.b):
            a = self.af(func.weighted_input(w,a,b))
        return a 

    def anz(self, a):
        """compute layered a and z,
        self.a[0] is the data of input layer"""
        self.z.clear()
        self.a.clear()
        self.a.append(a)
        for w,b in zip(self.w, self.b):
            self.z.append(func.weighted_input(w,self.a[-1],b))
            self.a.append(self.af(self.z[-1]))

    def cost(self, data):
        """total averaged cost over data pairs"""
        return sum([self.cf(x[1],self.ff(x[0])) for x in data])/len(data)
    
    def backprop(self, x, y):
        """backprop algorithm, get gradient.
        x, y are a single pair of the known input and output
        return nabla w and b tuple with the same shape of nn."""  
        nabla_w = [np.zeros_like(w) for w in self.w]
        nabla_b = [np.zeros_like(b) for b in self.b]
        self.anz(x)
        # the output layer
        delta = self.cfda(y, self.a[-1])*self.afdz(self.z[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = delta @ self.a[-2].T
        # the rest layer, backward
        for i in range(self.num_layers-2, 0, -1):
            delta = self.w[i].T @ delta * self.afdz(self.z[i-1])
            nabla_b[i-1] = delta
            nabla_w[i-1] = delta @ self.a[i-1].T
        return nabla_w, nabla_b

    def gd(self, data, eta):
        """ gradient descent """    
        nabla_w = [np.zeros_like(w) for w in self.w]
        nabla_b = [np.zeros_like(b) for b in self.b]
        num = len(data)
        for x,y in data:
            delta_w, delta_b = self.backprop(x, y)
            nabla_w = [j+k for j,k in zip(nabla_w,delta_w)]
            nabla_b = [j+k for j,k in zip(nabla_b,delta_b)]
        self.w = [x-eta*w/num for x,w in zip(self.w, nabla_w)]
        self.b = [x-eta*b/num for x,b in zip(self.b, nabla_b)]

    def sgd(self, trd, mblen, eta):
        """ stochastic gradient descent for one epoch """
        random.shuffle(trd)
        mb = [trd[k:k+mblen] for k in range(0,len(trd),mblen)]
        for i in range(len(mb)):
            self.gd(mb[i], eta)


