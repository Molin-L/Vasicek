import pytest
import numpy as np
from vasicek_model import Vasicek

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
