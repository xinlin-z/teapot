import time
import datetime
import numpy as np


def getMilliS():
    """get millisecond of now in string of length 3"""
    a = str(int(time.time()*1000)%1000)
    if len(a) == 1: return '00'+a
    if len(a) == 2: return '0'+a
    return a


def getTime():
    """get time in format HH:MM:SS.MS"""
    now = time.strftime('%H:%M:%S', time.localtime())
    return now+'.'+getMilliS()


def getDate():
    """get date time in format YYYY_MM_DD"""
    return datetime.date.today().isoformat()


def getDateTime():
    """get date time in formate YYYY_MM_DD HH:MM:SS.MS"""
    return getDate()+' '+getTime()


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
    if y.shape[1] != 1:
        return sum(0.5*np.linalg.norm(x)**2 for x in (y-a).T)
    return 0.5*(np.linalg.norm(y-a)**2)


def qcost_x_da(y, a):
    """ partial derivative of qcost_x for a """
    return a-y

