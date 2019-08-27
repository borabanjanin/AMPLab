import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
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
import dataloader
import models
import util
import models
import phaser
import numpy as np
import pickle

#Estimate phase using the Phaser algorithm
def find_phase(k):
    l = ['pelvis_tz','hip_flexion_r','hip_flexion_l','hip_adduction_r','hip_adduction_l' \
    ,'knee_angle_r','knee_angle_l','ankle_angle_r','ankle_angle_r']
    l = ['hip_flexion_l','hip_flexion_r']
    y = np.array([k[l_] for l_ in l])
    y = util.detrend(y.T).T
    phsr = phaser.Phaser(y=y)
    k['phase'] = phsr.phaserEval(y)[0,:]
    return k

def NN_results(subjectID, pla, fileName, sizeReduction):
    """
    Trains NPV model on K0,K1,K3 and predicts K2 for a given model
    INPUT:
      subjectID -- int
      sizeReduction -- int
    """
    np.random.seed(22)
    pd.options.mode.chained_assignment = None
    dw = dataloader.load_P00X(subjectID)

    bi = 50

    rrv_all = []
    rrv_split = []

    #creating the trial id
    k_base = copy.deepcopy(dw.kinematics.query('TrialID == 0')).reset_index(drop=True)
    k_t = copy.deepcopy(dw.kinematics.query('TrialID == [0,1,2]'))
    if sizeReduction > 0:
        k_t = pd.concat([k_t[k_t['TrialID'] == 0][:-sizeReduction],k_t[k_t['TrialID'] == 1][:-sizeReduction],k_t[k_t['TrialID'] == 2][:-sizeReduction]],ignore_index=True)
    k_v = copy.deepcopy(dw.kinematics.query('TrialID == 3'))

    #phaser phase estimate
    k_base = find_phase(k_base)
    k_t = find_phase(k_t)
    k_v = find_phase(k_v)
    k_base = k_base.drop(columns=['pelvis_tx','pelvis_ty','pelvis_tz'])
    k_t = k_t.drop(columns=['pelvis_tx','pelvis_ty','pelvis_tz'])
    k_v = k_v.drop(columns=['pelvis_tx','pelvis_ty','pelvis_tz'])
    #drop global columns
    #convert numpy arrays
    Z_base,phi_base,t_base,l_base = models.amp_input_final(k_base)
    Z_t,phi_t,t_t,l_t = models.amp_input_final(k_t)
    Z_v,phi_v,t_v,l_v = models.amp_input_final(k_v)
    #Removing the AFO columns in the training set
    I_afo = [l_t.index('AFO_L'),l_t.index('AFO_R'),l_t.index('d_AFO_L'),l_t.index('d_AFO_R')]
    # Z_afo_tr = Z_tr[:,I_afo]
    I = np.array(range(np.shape(Z_t)[1]))
    Z_base = Z_base[:,np.delete(I,I_afo)]
    #sample the torques
    Z_t,phi_t,l_t = models.sample_torque(Z_t,phi_t,l_t,pla)
    Z_v,phi_v,l_v = models.sample_torque(Z_v,phi_v,l_v,pla)
    #fs model
    f_base = models.fs_model(Z_base, Z_base, phi_base, phi_base, function=True)
    #Remove the phase mean
    Z_t[:,np.delete(I,I_afo)] -= np.array([f_base(phi) for phi in phi_t])
    Z_v[:,np.delete(I,I_afo)] -= np.array([f_base(phi) for phi in phi_v])
    #studentize data
    mean = np.mean(Z_t,axis=0)
    std = np.std(Z_t,axis=0)
    Z_t = (Z_t-mean)/std
    Z_v = (Z_v-mean)/std
    #map array to input output arrays
    Y_t,X_t,I_t = util.set_mapper(phi_t,Z_t,pla)
    X_t = np.hstack((X_t,phi_t[I_t][:,np.newaxis]))
    Y_v,X_v,I_v = util.set_mapper(phi_v,Z_v,pla)
    X_v = np.hstack((X_v,phi_v[I_v][:,np.newaxis]))
    #Rename according to Keras convetion
    train_data = X_t
    train_labels = Y_t
    test_data = X_v
    test_labels = Y_v

    # Shuffle the training set
    order = np.argsort(np.random.random(train_labels.shape[0]))
    train_data = train_data[order]
    train_labels = train_labels[order]

    print("Training set: {}".format(train_data.shape))
    print("Testing set:  {}".format(test_data.shape))
    #Studentize data
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    #Build 3 layer FFN
    def build_model():
      model = keras.Sequential([
        keras.layers.Dense(128, activation=tf.nn.softsign,
                           input_shape=(train_data.shape[1],)),
        keras.layers.Dense(128, activation=tf.nn.softsign),
        keras.layers.Dense(train_labels.shape[1])
      ])

      optimizer = tf.train.RMSPropOptimizer(0.001)

      model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae'])
      return model

    model = build_model()

    # Display training progress by printing a single dot for each completed epoch.
    class PrintDot(keras.callbacks.Callback):
      def on_epoch_end(self,epoch,logs):
        if epoch % 100 == 0: print('')
      print('.')

    EPOCHS = 1000

    # Store training stats
    history = model.fit(train_data, train_labels, epochs=EPOCHS,
                        validation_split=0.2, verbose=0, batch_size=30000,
                        callbacks=[PrintDot()])


    [loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

    print('Predicting')
    test_predictions = model.predict(test_data)

    #Save RRV statistic
    file_name = 'P'+'%03d'%(subjectID)+'_NN_'+fileName+'_'+str(28800-sizeReduction)+'.pckl'
    RRV = util.bootstrap_rrv(test_predictions, test_labels, bi)
    Nstates = int(len(l_v)/2)
    results = pd.DataFrame({"RRV":np.mean(RRV[:,:Nstates], axis=0)},index=l_v[:Nstates])
    print(results)
    results.to_pickle(os.path.join('results',file_name))
    #Save predictions
    preds = {'Y':test_labels,'Yh':test_predictions,'phi':phi_v[I_v],'l':l_t ,'X_t':X_t,'X_v':X_v}
    with open(os.path.join('results','predictions',file_name),'wb') as handle:
        pickle.dump(preds, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Loop over subjects
# for subjectID in [2,3,4,5,6,7,8,9,10,11,12,13]:
for subjectID in [2]:
    #[,]
    # pl_name_pairs = [(i*np.pi/8,str(i)+'pi-8') for i in range(1,16)]
    #Loop over phase lookaheads
    pl_name_pairs = [(2*np.pi/8,'2pi-8')]
    for pl_pair in pl_name_pairs:
        sizeReduction = 0
    # for sizeReduction in [2500,5000,7500,10000,12500,15000,17500,20000,22500,25000,27500]:
        # NN_results(subjectID, pl_pair[0], pl_pair[1],sizeReduction)
        NN_results(subjectID, pl_pair[0], pl_pair[1],sizeReduction)
