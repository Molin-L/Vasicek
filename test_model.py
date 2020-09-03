import pytest
import numpy as np
from matplotlib import pyplot as plt
from vasicek_model import Vasicek

def plot(data):
    if data.ndim == 2:
        plt.plot(np.arange(len(data)), np.mean(data, axis=1), color='orange')
    else:
        plt.plot(np.arange(len(data)), data, color='orange')
    yield plt.show()
    plt.close('all')


def init(func):
    def inner(*args, **kwargs):
        model = Vasicek()

        func(*args, **kwargs)
    return inner


def test_quant(model=Vasicek(n_sim=10000)):
    print(np.mean(model.quant_u_gamma))
    print(np.mean(model.quant_s_gamma))
    #assert np.mean(model.quant_u_gamma)>-0.3 and np.mean(model.quant_u_gamma)<-0.1
    #assert np.mean(model.quant_s_gamma)>0.8 and np.mean(model.quant_s_gamma)<1

def test_Tprice(model=Vasicek(n_sim=10000)):
    print(model.Tprice)

def test_AB():
    due = [0.5, 1, 2]
    model = Vasicek()
    for T in due:
        for t in np.arange(0, model.T, model.dt):
            assert model._B(t, T)!=None
            assert model._A(t, T)!=None

def test_price(model=Vasicek()):
    U_price = model._test_cal_price(model.U)
    plot(U_price)
