import os
import pandas as pd
import numpy as np
import pylab as plt
from scipy import signal
from collections import deque
from scipy.interpolate import interp1d
import dataloader
import copy
import util
import models
import util
import models
import phaser
import pickle
import dataloader

def find_phase(k):
    l = ['hip_flexion_l','hip_flexion_r']
    y = np.array([k[l_] for l_ in l])
    y = util.detrend(y.T).T
    phsr = phaser.Phaser(y=y)
    k['phase'] = phsr.phaserEval(y)[0,:]
    return k

def integrate_torque(k,ic):
    dt = np.array(k['Time'])[1]-np.array(k['Time'])[0]
    k['AFO_R'] = util.integrateTorque(k['AFO_R'][:,np.newaxis],ic[:,np.newaxis],dt,side='Ipsi')
    k['AFO_L'] = util.integrateTorque(k['AFO_L'][:,np.newaxis],ic[:,np.newaxis],dt,side='Ipsi')
    return k

def universal_frames(k,ic):
    n = len(k['TrialID'].unique())
    frames = []
    for i,ic_ in enumerate(ic):
        frames.append(np.array(ic_['Frame']+i*28800-2))
    return np.hstack(frames)

def PV_results(subjectID,sizeReduction):
    pd.options.mode.chained_assignment = None
    dw = dataloader.load_P00X(subjectID)

    bi = 50

    rrv_all = []
    rrv_split = []

    #creating the trial id
    k_base = copy.deepcopy(dw.kinematics.query('TrialID == 0')).reset_index(drop=True)
    k_tr = copy.deepcopy(dw.kinematics.query('TrialID == [0,1,2]')).reset_index(drop=True)
    if sizeReduction > 0:
        k_tr = pd.concat([k_tr[k_tr['TrialID'] == 0][:-sizeReduction],k_tr[k_tr['TrialID'] == 1][:-sizeReduction],k_tr[k_tr['TrialID'] == 3][:-sizeReduction]],ignore_index=True)
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
    #fs model
    f_base = models.fs_model(Z_base, Z_val, phi_base, phi_val, function=True)
    #Remove the phase mean
    Z_tr[:,np.delete(I,I_afo)] -= np.array([f_base(phi) for phi in phi_tr])
    Z_val[:,np.delete(I,I_afo)] -= np.array([f_base(phi) for phi in phi_val])
    #studentize data
    mean = np.mean(Z_tr,axis=0)
    std = np.std(Z_tr,axis=0)
    Z_tr = (Z_tr-mean)/std
    Z_val = (Z_val-mean)/std

    print('Predicting')
    Yh, Y_v = models.fs_model(Z_tr, Z_val, phi_tr, phi_val)

    #file_name = 'P'+'%03d'%(subjectID)+'_PV_.pckl'
    file_name = 'P'+'%03d'%(subjectID)+'_PV_'+str(28800-sizeReduction)+'.pckl'
    print('Average error predicting from initial condition with the average model: '+str(np.mean(np.abs(Yh-Y_v))))
    RRV = util.bootstrap_rrv(Yh, Y_v, bi)
    Nstates = int(len(l_val)/2)
    results = pd.DataFrame({"RRV":np.mean(RRV[:,:Nstates], axis=0)},index=l_val[:Nstates])
    print(results)
    results.to_pickle(os.path.join('results',file_name))

    preds = {'Y':Y_v,'Yh':Yh,'phi':phi_val,'l':l_tr}
    with open(os.path.join('results','predictions',file_name),'wb') as handle:
        pickle.dump(preds, handle, protocol=pickle.HIGHEST_PROTOCOL)

for subjectID in [3]:
# for subjectID in range(2,14):
    # for sizeReduction in [2500,5000,7500,10000,12500,15000,17500,20000,22500,25000,27500]:
    sizeReduction = 0
    PV_results(subjectID,sizeReduction)
