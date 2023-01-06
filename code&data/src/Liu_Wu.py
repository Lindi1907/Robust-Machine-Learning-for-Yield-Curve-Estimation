import numpy as np
import pandas as pd
#############################
#CANNOT BE IMPLEMENTED!!!
#############################
# all their date is in month

def bandwidth_h(x, ttm, N_0 = 8):
    """Bandwidth function in (3.3)

    Args:
        x (float): date of cash flow, in days
        ttm (np.array): list of time to maturity, indays
        N_0 (int, optional): bandwidth of search. Defaults to 8.

    Returns:
        h: bandwidth of gaussian kernel, in months
    """
    x = x/30
    ttm = ttm/30 
    index = np.arange(0,len(ttm))
    if(len(ttm[ttm<=x])>=N_0/2):
        vv = index[ttm<=x].max
        b = ttm[vv - N_0/2]
        h_l = b/2
    else:
        h_l = x/2
    if(len(ttm[ttm>=x])>=N_0/2):
        vv = index[ttm>=x].min
        b = ttm[vv + N_0/2]
        h_r = b/2
    else:
        h_r = (ttm.max - x)/2
    _list = [h_l, h_r, 3]
    h = min(max(_list), 120)
    return h

def _Gaussian_K(x, n, h):
    """Calculate the result of Liu Wu kernel

    Args:
        x (float): daily data
        n (int): month, from 1 to 360, used in loop
        h (float): bandwidth

    Returns:
        K: kernel value
    """
    con = 1/np.sqrt(2*np.pi*(h**2))
    K = con*np.exp(-((n - x)**2)/(2*(h**2)))
    return K
    
def g_Liu_Wu(x, h, n = 360):
    neu = []
    for ni in range(1, n+1):
        tra = _Gaussian_K(x, ni, h)*np.exp()
        neu.append(tra)
    
    return g



