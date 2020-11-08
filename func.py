import time
import datetime
import numpy as np


def weighted_input(w, a, b):
    """ z """
    return w@a+b


def sigmoid(z):
    """ sigmoid """
    return 1.0/(1.0+np.exp(-z))


def sigmoid_dz(z):
    """ da/dz partial derivative of sigmoid """
    s = sigmoid(z)
    return s*(1-s)


def qcost_x(y, a):
    """Quadratic cost function only for a single input x,
    Cx = 1/2 * ||y(x)-a(x)||^2,
    output is float number."""
    return 0.5*(np.linalg.norm(y-a)**2)


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


