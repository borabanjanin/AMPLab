import os
import pandas as pd
import numpy as np
import pylab as plt
from scipy import signal
import shaiutil
import util
import copy
from sklearn.linear_model import HuberRegressor
from scipy.interpolate import interp1d
import statsmodels.api as sm
from scipy.interpolate import interp1d

def diff(k,l):
    """
    Takes a computes gradients for a list of columns from a dataframe.
    INPUT:
      k -- dataframe
      l -- list -- column names
    OUTPUT:
      k -- dataframe
    """
    for l_ in l:
        k['d_'+l_] = np.gradient(k[l_])
    return k

def amp_input_final(k):
    """
    Takes a dataframe and computes gradients for the outcomes. Makes the DataFrame
    compatible with the prediction framework
    INPUT:
      k -- dataframe
    OUTPUT:
      Z -- NumPy array -- time series data
      phi -- NumPy vector -- phase estimates
      t -- NumPy vector -- time
      l -- list -- column names
    """
    l = list(k.columns)
    while 'time' in l:
        l.remove('time')
    l.remove('TrialID')
    l.remove('phase')
    k = diff(k,l)
    l = l + ['d_'+l_ for l_ in l]
    Z = np.array(k[l])
    phi = np.array(k['phase'])
    t = np.array(k['time'])
    return Z, phi, t, l

def model0(Y, X):
    '''
        Model:
            Y = AX
        Input:
            X - numpy array (n x d)
            Y - numpy array (n x d)
        Output:
            A - numpy array (d x d+1)
    '''
    n, d = Y.shape

    X_ = util.affenize(X)
    A = np.dot(np.linalg.inv(np.dot(X_.T, X_)),np.dot(X_.T, Y))
    return A

def model0_fit(Z_t, Z_v, phi_t, phi_v, theta):
    '''
        Model:
            Finds Yh_[phi+theta] = A(Y_[phi])
            Works for either phase or time look-ahead model
            Fits on Z_t, phi_t
            Validates on Z_v, phi_v
        Input:
            Z_t - numpy array (m x n)
            Z_v - numpy array (k x n)
            phi_t - numpy array (m x 1)
            phi_v - numpy array (k x 1)
            theta - scalar
        Output:
            Y - numpy array (l x d)
            Yh - numpy array (l x d)
            I - numpy array (d)
    '''
    Y_t, X_t, I_t = util.set_mapper(phi_t, Z_t, theta)
    Y_v, X_v, I_v = util.set_mapper(phi_v, Z_v, theta)
    A = model0(Y_t, X_t)
    Yh = np.dot(util.affenize(X_v), A)
    return Yh, Y_v, I_v

def model1(Y, X, phi, GRID_SAMPLES = 64):
    '''
        Model:
            Y = C(phi, X)
        Input:
            X - numpy array (n x d)
            Y - numpy array (n x l)
            phi - numpy array (n)
        Output:
            A - numpy array (d x d+1)
    '''
    m, n = np.shape(Y)
    m, l = np.shape(X)
    C = np.zeros((GRID_SAMPLES,l+1,l))
    X_ = util.affenize(X)

    phi_c = np.exp(1.j * phi)
    grid = np.arange(GRID_SAMPLES) * (2*np.pi)/GRID_SAMPLES
    grid_c = np.exp(1.j * grid)
    for i, g in enumerate(grid_c):
        w = util.phase_discounting(g, phi_c, STD_DEVIATION = 0.5)
        # w = util.phase_discounting(g, phi_c, STD_DEVIATION = 0.5)
        a = np.multiply(w[:,np.newaxis],X_)
        b = np.multiply(w[:,np.newaxis],Y)
        # C[i,:,:] =  np.linalg.lstsq(a,b)[0]
        C[i,:,:] = np.dot(np.linalg.inv(np.dot(X_.T,a)),np.dot(X_.T,b))
        # W = np.matrix(np.diag(w))
        # C[i,:,:] = np.dot(np.linalg.inv(np.dot(X_.T,np.dot(W,X_))),np.dot(X_.T,np.dot(W,Y)))
        # C[i,:,:] = np.linalg.lstsq(np.dot(W,Y),np.dot(W,X_))[0].T
    return C

def model1_function(C):
    '''
        Model:
            Takes a tensor where the fourier series are fit along
            the kth dimension.
        Input:
            C - numpy array (k x m x n)
        Output:
            f - function returns numpy array (1 -> m x n)
    '''
    ORDER = 3
    k, m, n = np.shape(C)

    phase = np.arange(k)[np.newaxis,:] * (2*np.pi)/k

    fs_list = [[shaiutil.FourierSeries() for i in range(n)] for j in range(m)]
    fs_list = [[fs_list[i][j].fit(ORDER, phase, C[:,i, j][np.newaxis,:]) \
    for j in range(n)] for i in range(m)]
    f = lambda phi : np.array([[fs_list[i][j].val(phi).real[:,0] \
    for j in range(n)] for i in range(m)])

    return f

def model1_fit(Z_t, Z_v, phi_t, phi_v, theta, robust=None):
    '''
        Model:
            Finds Yh_[phi+theta] = A(Y_[phi])
            Works for either phase or time look-ahead model
            Fits on Z_t, phi_t
            Validates on Z_v, phi_v
        Input:
            Z_t - numpy array (m x n)
            Z_v - numpy array (k x n)
            phi_t - numpy array (m x 1)
            phi_v - numpy array (k x 1)
            theta - scalar
        Output:
            Yh - numpy array (l x d)
            Y - numpy array (l x d)
            I - numpy array (d)
    '''
    Y_t, X_t, I_t = util.set_mapper(phi_t, Z_t, theta)
    Y_v, X_v, I_v = util.set_mapper(phi_v, Z_v, theta)
    # return Y_v,Y_v,I_v
    print('Finding Coefficients')
    if robust == None:
        C = model1(Y_t, X_t, phi_t[I_t])
    elif robust == True:
        C = model1_robust(Y_t, X_t, phi_t[I_t])
    print('Creating Fourier Series')
    fs = model1_function(C)
    m, n = np.shape(X_v)
    Yh = np.zeros((m, n))
    print('Predicting')
    A = fs(phi_v[I_v])
    for i in range(m):
        Yh[i,:] = np.dot(util.affenize(X_v[i,:][np.newaxis,:]), A[:,:,i])
    # for i in range(m):
    #     Yh[i,:] = np.dot(util.affenize(X_v[i,:][np.newaxis,:]), fs(phi_v[i]))
    return Yh, Y_v, I_v

def model_phase(Y, phi):
    '''
        Model:
            Y = C(phi)*phi
        Input:
            Y - numpy array (n x d)
            phi - numpy array (n)
        Output:
            A - numpy array (1 x d)
    '''
    GRID_SAMPLES = 1
    X_ = util.affenize(phi[:,np.newaxis])
    n,m  = np.shape(X_)
    n,d = np.shape(Y)
    C = np.zeros((GRID_SAMPLES,m,d))

    phi_c = np.exp(1.j * phi)
    grid = np.arange(GRID_SAMPLES) * (2*np.pi)/GRID_SAMPLES
    grid_c = np.exp(1.j * grid)
    for i,g in enumerate(grid_c):
        w = util.phase_discounting(g, phi_c)
        W = np.matrix(np.diag(w))
        C[i,:,:] = np.dot(np.linalg.inv(np.dot(X_.T,np.dot(W,X_))),np.dot(X_.T,np.dot(W,Y)))
    return C

def model_phase_fit(Y_t, Y_v, phi_t, phi_v):
    '''
        Model:
            Finds Yh = A(phi)
            Works for either phase or time look-ahead model
            Fits on Z_t, phi_t
            Validates on Z_v, phi_v
        Input:
            Z_t - numpy array (m x n)
            Z_v - numpy array (k x n)
            phi_t - numpy array (m x 1)
            phi_v - numpy array (k x 1)
            theta - scalar
        Output:
            Yh - numpy array (l x d)
            Y - numpy array (l x d)
            I - numpy array (d)
    '''

    C = model_phase(Y_t, phi_t)
    fs = model1_function(C)
    m, n = np.shape(Y_v)
    Yh = np.zeros((m, n))
    for i in range(m):
        Yh[i,:] = np.dot(np.array([phi_v[i],1.]), fs(phi_v[i]))
    return Yh, Y_v

def fs_model(Y_t, Y_v, phi_t, phi_v, function=False):
    ORDER = 7
    n,d = np.shape(Y_t)

    fs_list = [shaiutil.FourierSeries() for i in range(d)]
    fs_list = [fs_list[i].fit(ORDER, phi_t[np.newaxis,:], Y_t[:,i][np.newaxis,:])for i in range(d)]

    f = lambda phi : np.array([fs_list[i].val(phi).real[0,0] \
    for i in range(d)])

    if function == True:
        return f
    Yh = np.array([f(phi) for phi in phi_v])
    return Yh, Y_v

def sample_torque(Z,phi,l,pla,samples=10):
    l = copy.deepcopy(l)
    i_afor = l.index('AFO_R')
    i_afol = l.index('AFO_L')
    i_dafor = l.index('d_AFO_R')
    i_dafol = l.index('d_AFO_L')
    for i in sorted([i_afor,i_afol,i_dafor,i_dafol],reverse=True):
        del l[i]
    m,n = np.shape(Z)
    Z_t = np.delete(Z,[i_afor,i_afol,i_dafor,i_dafol],1)
    Z_t = np.hstack((Z_t,np.zeros((m,2*samples))*np.nan))
    for i in range(m):
        j = 0
        while phi[i+j] <= phi[i]+pla:
            j += 1
            if j+i+1 > m:
                j = m-i-1
                break
        if j<=1:
            continue
        f_t = interp1d(phi[i:i+j+1],Z[i:i+j+1,[i_afor,i_afol]].T,bounds_error=False)
        phi_s = np.linspace(phi[i],phi[i+j],num=samples)
        if np.count_nonzero(np.isnan(f_t(phi_s).flatten())) > 0:
            print(i)
        Z_t[i,-2*samples:] = f_t(phi_s).flatten()
    l_rt = ['Torque_R_'+str(i) for i in range(samples)]
    l_lt = ['Torque_L_'+str(i) for i in range(samples)]
    l_t = l+l_rt+l_lt
    return Z_t[~np.isnan(Z_t).any(axis=1)],phi[~np.isnan(Z_t).any(axis=1)],l_t

if __name__ == "__main__":
    print
