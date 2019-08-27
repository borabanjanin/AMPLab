import os
import pandas as pd
import numpy as np
import pylab as plt
from scipy import signal
from collections import deque
from scipy.interpolate import interp1d
import copy
import util
import shaiutil
import models
import util
import models
import phaser
import pickle
import dataloader

def find_phase(k):
    """
    Detrend and compute the phase estimate using Phaser
    INPUT:
      k -- dataframe
    OUTPUT:
      k -- dataframe
    """
    l = ['hip_flexion_l','hip_flexion_r']
    y = np.array([k[l_] for l_ in l])
    y = util.detrend(y.T).T
    phsr = phaser.Phaser(y=y)
    k['phase'] = phsr.phaserEval(y)[0,:]
    return k

def LPV_results(subjectID, pla, fileName, sizeReduction):
    """
    Trains LPV model on K0,K1,K3 and predicts K2 for a given model
    INPUT:
      subjectID -- int
      sizeReduction -- int
    """
    pd.options.mode.chained_assignment = None
    dw = dataloader.load_P00X(subjectID)

    bi = 50

    rrv_all = []
    rrv_split = []

    #creating the trial id
    k_base = copy.deepcopy(dw.kinematics.query('TrialID == 0')).reset_index(drop=True)
    k_tr = copy.deepcopy(dw.kinematics.query('TrialID == [0,1,2]')).reset_index(drop=True)
    k_tr = pd.concat([k_tr[k_tr['TrialID'] == 0][:-sizeReduction],k_tr[k_tr['TrialID'] == 1][:-sizeReduction],k_tr[k_tr['TrialID'] == 2][:-sizeReduction]],ignore_index=True)
    k_val = copy.deepcopy(dw.kinematics.query('TrialID == [3]')).reset_index(drop=True)

    #phaser phase estimate
    k_base = find_phase(k_base)
    k_tr = find_phase(k_tr)
    k_val = find_phase(k_val)
    k_base = k_base.drop(columns=['pelvis_tx','pelvis_ty','pelvis_tz'])
    k_tr = k_tr.drop(columns=['pelvis_tx','pelvis_ty','pelvis_tz'])
    k_val = k_val.drop(columns=['pelvis_tx','pelvis_ty','pelvis_tz'])
    #model input
    Z_base,phi_base,t_base,l_base = models.amp_input_final(k_base)
    Z_tr,phi_tr,t_tr,l_tr = models.amp_input_final(k_tr)
    Z_val,phi_val,t_val,l_val = models.amp_input_final(k_val)
    #Removing the AFO columns in the training set
    I_afo = [l_tr.index('AFO_L'),l_tr.index('AFO_R'),l_tr.index('d_AFO_L'),l_tr.index('d_AFO_R')]
    I = np.array(range(np.shape(Z_tr)[1]))
    Z_base = Z_base[:,np.delete(I,I_afo)]
    ###
    # Z_tr = Z_tr[:,np.delete(I,I_afo)]
    # Z_val = Z_val[:,np.delete(I,I_afo)]
    # l_tr = [l_tr[i] for i in np.delete(I,I_afo)]
    # l_val = [l_val[i] for i in np.delete(I,I_afo)]
    ###
    #sample the torques
    Z_tr,phi_tr,l_tr = models.sample_torque(Z_tr,phi_tr,l_tr,pla)
    Z_val,phi_val,l_val = models.sample_torque(Z_val,phi_val,l_val,pla)
    #fs model
    f_base = models.fs_model(Z_base, Z_val, phi_base, phi_val, function=True)

    #Remove the phase mean
    Z_tr[:,np.delete(I,I_afo)] -= np.array([f_base(phi) for phi in phi_tr])
    Z_val[:,np.delete(I,I_afo)] -= np.array([f_base(phi) for phi in phi_val])
    # Z_tr -= np.array([f_base(phi) for phi in phi_tr])
    # Z_val -= np.array([f_base(phi) for phi in phi_val])
    #studentize data
    mean = np.mean(Z_tr,axis=0)
    std = np.std(Z_tr,axis=0)
    Z_tr = (Z_tr-mean)/std
    Z_val = (Z_val-mean)/std

    Y_t, X_t, I_t = util.set_mapper(phi_tr, Z_tr, pla)
    Y_v, X_v, I_v = util.set_mapper(phi_val, Z_val, pla)

    print('Finding Coefficients')
    C = models.model1(Y_t, X_t, phi_tr[I_t])
    print('Creating Fourier Series')
    fs = models.model1_function(C)
    m, n = np.shape(X_v)
    Yh = np.zeros((m, n))
    print('Predicting')
    A = fs(phi_val[I_v])
    for i in range(m):
        Yh[i,:] = np.dot(util.affenize(X_v[i,:][np.newaxis,:]), A[:,:,i])

    #Save RRV statistic
    file_name = 'P'+'%03d'%(subjectID)+'_LPV_'+fileName+'_'+str(28800-sizeReduction)+'.pckl'
    print('Average error predicting from initial condition with the average model: '+str(np.mean(np.abs(Yh-Y_v))))
    RRV = util.bootstrap_rrv(Yh, Y_v, bi)
    Nstates = int(len(l_val)/2)
    results = pd.DataFrame({"RRV":np.mean(RRV[:,:Nstates], axis=0)},index=l_val[:Nstates])
    print(results)
    results.to_pickle(os.path.join('results',file_name))

    #Save predictions
    # preds = {'Y':Y_v,'Yh':Yh,'phi':phi_val[I_v],'l':l_tr,'X_t':X_t,'X_v':X_v,'A':A}
    # with open(os.path.join('results','predictions',file_name),'wb') as handle:
    #     pickle.dump(preds, handle, protocol=pickle.HIGHEST_PROTOCOL)

# for subjectID in [2,3,4,5,6,7,8,9,10,11,12,13]:
for subjectID in [2]:
    # pl_name_pairs = [(i*np.pi/8,str(i)+'pi-8') for i in range(1,16)]
    pl_pair = (2*np.pi/8,'2pi-8')
    # for pl_pair in pl_name_pairs:
    #     sizeReduction = 0
    for sizeReduction in [0, 2500,5000,7500,10000,12500,15000,17500,20000,22500,25000,27500]:
        LPV_results(subjectID, pl_pair[0], pl_pair[1], sizeReduction)
