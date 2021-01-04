import numpy as np
from dataset import DTYPE


def weighted_input(w, a, b):
    return w@a+b


z = weighted_input


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_dz(z):
    s = sigmoid(z)
    return s*(1-s)


def tanh(z):
    a = np.exp(z)
    b = np.exp(-z)
    return (a-b)/(a+b)


def tanh_dz(z):
    return 1-tanh(z)**2


def relu(z):
    return np.maximum(0.0, z)


def relu_dz(z):
    return (z>0).astype(DTYPE)


def lu(z):
    return z


def lu_dz(z):
    return np.ones_like(z, dtype=DTYPE)


def qcost_x(y, a):
    """Quadratic cost function only for a single input x,
    Cx = 1/2 * ||y(x)-a(x)||^2,
    output is float number."""
    return 0.5*np.linalg.norm(y-a)**2


def qcost_x_da(y, a):
    """ partial derivative of qcost_x for a """
    return a-y


def cross_entropy_x(y, a):
    """ cross-entropy cost
    Cx = -sum((y*ln(a) + (1-y)*ln(1-a))) """
    return -np.sum(y*np.log(a)+(1-y)*np.log(1-a))


def cross_entropy_x_da(y, a):
    """ partial derivative of corssentropy for a """
    return (a-y)/(a*(1-a))


