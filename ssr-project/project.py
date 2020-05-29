import numpy as np
from lab3_tools import *
from lab1_proto import mfcc
from prondict import prondict
from lab2_proto import *
from lab3_proto import *

import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils
from models import *
from keras.models import load_model

def generate_data():
    training_data = None
    test_data = None

    try:
        print("--> Trying to load file: traindata.npz")
        training_data = np.load("traindata.npz", allow_pickle = True)
        print("--> Successfully loaded: traindata.npz")
    except:
        print("----- File not found: reading in the training data-----")
        traindata = []
        for root, dirs, files in os.walk('../tidigits/timit/disc_4.1.1/tidigits'):
            for file_ in tqdm(files):
                if file_.endswith('.wav'):
                    filename = os.path.join(root, file_)
                    samples, sampleingrate = loadAudio(filename)
                    lmfcc, mspec, targets = extraction_and_forced_alignment(samples, filename)

                    traindata.append({'filename': filename, 'lmfcc': lmfcc, 'targets': targets})
        np.savez('traindata.npz', traindata = traindata)
        training_data = traindata

        # ABC10-90
        # ABC15-85
        # ABC20-80
    try:
        print("--> Trying to load file: testdata.npz")
        test_data = np.load("testdata.npz", allow_pickle = True)
        print("--> Successfully loaded: testdata.npz")
    except:
        print("----- File not found: reading in the training data-----")
        testdata = []
        for root, dirs, files in os.walk('../../tidigits/disc_4.2.1/tidigits/test'):
            for file_ in tqdm(files):
                if file_.endswith('.wav'):
                    filename = os.path.join(root, file_)
                    samples, sampleingrate = loadAudio(filename)
                    lmfcc, _ , targets = extraction_and_forced_alignment(samples)

                    testdata.append({'filename': filename, 'lmfcc': lmfcc, 'targets': targets})
        np.savez('testdata.npz', testdata = testdata)
        test_data = testdata

    return training_data['traindata'], test_data['testdata']

def generate_data_dyn(traindata_arg, testdata_arg, stack_size=7):
    traindata = traindata_arg.copy()
    testdata = testdata_arg.copy()

    traindata_dyn = None
    testdata_dyn = None

    try:
        print("--> Trying to load file: traindata_dyn.npz")
        traindata_dyn = np.load("traindata_dyn.npz", allow_pickle=True)
        print("--> Successfully loaded: traindata_dyn.npz")
    except IOError:
        print("--> File not found, creating traindata_dyn.npz")
        for sample in tqdm(traindata):
            for key in sample.keys():
                if key == "filename" or key == "targets":
                    continue
                last_timestep = len(sample[key])
                feature_list = np.concatenate((np.array([3, 2, 1]), np.arange(last_timestep), 
                np.array([last_timestep-2, last_timestep-3, last_timestep-4])), axis=None)
                
                dyn_list = np.zeros((last_timestep, stack_size, len(sample[key][0])), dtype="object")
                dyn_feat = np.empty((stack_size, len(sample[key][0])), dtype="object")
                
                for timestep in range(len(sample[key])):
                    for i in range(stack_size):
                        dyn_feat[i] = sample[key][feature_list[timestep+i]]
                    dyn_list[timestep] = dyn_feat
                
                sample[key] = np.array(dyn_list)
        
        traindata_dyn = traindata
        np.savez("traindata_dyn.npz", traindata=traindata_dyn)
    
    
    try:
        print("--> Trying to load file: testdata_dyn.npz")
        testdata_dyn = np.load("testdata_dyn.npz", allow_pickle=True)
        print("--> Successfully loaded: testdata_dyn.npz")
    except IOError:
        print("--> File not found, creating testdata_dyn.npz")
        for sample in tqdm(testdata):
            for key in sample.keys():
                if key == "filename" or key == "targets":
                    continue
                last_timestep = len(sample[key])
                feature_list = np.concatenate((np.array([3, 2, 1]), np.arange(last_timestep), 
                np.array([last_timestep-2, last_timestep-3, last_timestep-4])), axis=None)

                dyn_list = np.zeros((last_timestep, stack_size, len(sample[key][0])), dtype="object")
                dyn_feat = np.empty((stack_size, len(sample[key][0])), dtype="object")
                
                for timestep in range(len(sample[key])):
                    for i in range(stack_size):
                        dyn_feat[i] = sample[key][feature_list[timestep+i]]
                    dyn_list[timestep] = dyn_feat
                
                sample[key] = np.array(dyn_list)
        
        testdata_dyn = testdata
        np.savez("testdata_dyn.npz", testdata=testdata_dyn)
    

    return traindata_dyn['traindata'], testdata_dyn['testdata']

def extract_data(x, key):
    print(x[0][key].shape)
    N = 0

    if key == "targets":
        D = 1
    else:
        D = x[0][key].shape[1] * x[0][key].shape[2]
    
    for i in range(len(x)):
        N += x[i][key].shape[0]
    print(N)
    if key == "targets":
        new_x = np.zeros((N, D), dtype="object")
    else: 
        new_x = np.zeros((N, D))
    prev_idx = 0
    for i in range(len(x)):
        n = x[i][key].shape[0]
        if key == "targets":
            new_x[prev_idx:prev_idx+n, 0] = x[i][key]
        else:
            new_x[prev_idx:prev_idx+n] = x[i][key].reshape((n, D))

        n += prev_idx
    
    return new_x

def normalise_data(traindata, testdata):    
    x_train = extract_data(traindata, "lmfcc")
    x_test = extract_data(testdata, "lmfcc")

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    np.savez("x_train.npz", data=x_train)
    np.savez("x_test.npz", data=x_test)

    return x_train, x_test

def targets_to_int(traindata, testdata, stateList):
    y_train = extract_data(traindata, "targets")
    y_test = extract_data(testdata, "targets")

    y_train_int  = np.zeros(len(y_train))
    y_test_int  = np.zeros(len(y_test))
    print(y_train)

    for i in range(len(y_train)):
        #print(y_train[i], end=", ")
        y_train_int[i] = int(stateList.index(y_train[i]))
        #print(y_train_int[i])
    
    
    for i in range(len(y_test)):
        y_test_int[i] = int(stateList.index(y_test[i]))
    
    #np.savez("y_train.npz", data=y_train_int)
    #np.savez("y_test.npz", data=y_test_int)

    return y_train, y_test

def get_data():

    stateList = np.load("data/stateList.npz", allow_pickle=True)['stateList']

    x_test = np.load("data/lmfcc_test_x.npz", allow_pickle=True)['lmfcc_test_x']
    y_test = np.load("data/test_y_int.npz", allow_pickle=True)['test_y_int']

    x_train = np.load("data/lmfcc_train_x.npz", allow_pickle=True)['lmfcc_train_x']
    y_train = np.load("data/train_y_int.npz", allow_pickle=True)['train_y_int']

    output_dim = len(stateList)

    y_test = np_utils.to_categorical(y_test, output_dim)
    y_train = np_utils.to_categorical(y_train, output_dim)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, stateList, output_dim

def semi_supervised(x_train, y_train, x_test, y_test, stateList, settings, frac=0.5):

    #kanske shuffle???????????
    x_train_a = x_train[:int(frac*len(x_train))]
    y_train_a = y_train[:int(frac*len(y_train))]
    
    print("Training model A...")
    A_model = NN_Model(settings)
    if not os.path.isfile("A_model_" + str(frac) + ".h5"):
        A_model.fit(x_train_a, y_train_a, settings)
        A_model.model.save("A_model_" + str(frac) + ".h5")
    else:
        A_model.model = load_model("A_model_" + str(frac) + ".h5") 

    pred_a_oh = A_model.model.predict(x_train[int(frac*len(x_train)):])

    y_train_b = np.concatenate((y_train_a, pred_a_oh), axis=0)

    print("Training model B...")
    B_model = NN_Model(settings)
    if not os.path.isfile("B_model_" + str(frac) + ".h5"):
        B_model.fit(x_train, y_train_b, settings)
        B_model.model.save("B_model_" + str(frac) + ".h5")
    else:
        B_model.model = load_model("B_model_" + str(frac) + ".h5")
    
    print("Training model C...")
    C_model = NN_Model(settings)
    if not os.path.isfile('C_model.h5'):
        C_model.fit(x_train, y_train, settings)
        C_model.model.save("C_model.h5")
    else:
        C_model.model = load_model("C_model.h5")
        

    print("-> Evaluating model B on test data <-")
    test_loss_B, test_acc_B = B_model.predict_merge(x_test, y_test, stateList)
    print("Model B (Accuracy): " + str(test_acc_B))
    print("Model B (Loss): " + str(test_loss_B)) 

    print("-> Evaluating model C on test data <-")
    test_loss_C, test_acc_C = C_model.predict_merge(x_test, y_test, stateList)
    print("Model C (Accuracy): " + str(test_acc_C))
    print("Model C (Loss): " + str(test_loss_C))

    print("-> Evaluating model A on test data <-")
    test_loss_A, test_acc_A = A_model.predict_merge(x_test, y_test, stateList)
    print("Model A (Accuracy): " + str(test_acc_A))
    print("Model A (Loss): " + str(test_loss_A))
    
    return test_loss_A, test_acc_A, test_loss_B, test_acc_B, test_loss_C, test_acc_C


if __name__ == '__main__':
    x_train, x_test, y_train, y_test, stateList, output_dim = get_data()
    
    settings = {
        "option" : "CNN1",
        "output_dim" : output_dim,
        "learning_rate" : None,
        "batch_size" : None,
        "timesteps" : len(x_train),
        "features" : len(x_train[0]),
        "n_epochs" : 5
    }

    if settings['option'] == "CNN1":
        x_train = np.expand_dims(x_train, axis=2)
        x_test = np.expand_dims(x_test, axis=2)

    #fraction of supervised data
    FRACTIONS =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    hist_loss_A = np.zeros(len(FRACTIONS))
    hist_loss_B = np.zeros(len(FRACTIONS))
    hist_loss_C = np.zeros(len(FRACTIONS))

    hist_acc_A = np.zeros(len(FRACTIONS))
    hist_acc_B = np.zeros(len(FRACTIONS))
    hist_acc_C = np.zeros(len(FRACTIONS))

    for i, frac in enumerate(FRACTIONS):
        print("Iteration " + str(i+1) + "/" + str(len(FRACTIONS)))
        test_loss_A, test_acc_A, test_loss_B, test_acc_B, test_loss_C, test_acc_C = semi_supervised(x_train, y_train, x_test, y_test, stateList, settings, frac)
        hist_acc_A[i] = test_acc_A
        hist_acc_B[i] = test_acc_B
        hist_acc_C[i] = test_acc_C
        
        hist_loss_A[i] = test_loss_A
        hist_loss_B[i] = test_loss_B
        hist_loss_C[i] = test_loss_C

    print("-------------------- PRINTOUTS --------------------------")
    print(hist_loss_A)
    print(hist_loss_B)
    print(hist_loss_C)

    print(hist_acc_A)
    print(hist_acc_B)
    print(hist_acc_C)
    np.savetxt("hist_loss_A.txt", hist_loss_A, delimiter=',')
    np.savetxt("hist_loss_B.txt", hist_loss_B, delimiter=',')
    np.savetxt("hist_loss_C.txt", hist_loss_C, delimiter=',')

    np.savetxt("hist_acc_A.txt", hist_acc_A, delimiter=',')
    np.savetxt("hist_acc_B.txt", hist_acc_B, delimiter=',')
    np.savetxt("hist_acc_C.txt", hist_acc_C, delimiter=',')