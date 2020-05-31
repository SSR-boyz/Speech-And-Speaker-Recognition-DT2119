import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils
from models import NN_Model
from keras.models import load_model

def get_data(feature_type):
    stateList = np.load("data/stateList.npz", allow_pickle=True)['stateList']
    output_dim = len(stateList)
    
    if feature_type == "lmfcc":
        x_test = np.load("data/lmfcc_test_x.npz", allow_pickle=True)['lmfcc_test_x']
        x_train = np.load("data/lmfcc_train_x.npz", allow_pickle=True)['lmfcc_train_x']
    elif feature_type == "mspec":
        x_test = np.load("data/mspec_test_x.npz", allow_pickle=True)['mspec_test_x']
        x_train = np.load("data/mspec_train_x.npz", allow_pickle=True)['mspec_train_x']
    else:
        pass
    
    y_test = np.load("data/test_y_int.npz", allow_pickle=True)['test_y_int']
    y_train = np.load("data/train_y_int.npz", allow_pickle=True)['train_y_int']

    y_test = np_utils.to_categorical(y_test, output_dim)
    y_train = np_utils.to_categorical(y_train, output_dim)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, stateList, output_dim

def semi_supervised(x_train, y_train, x_test, y_test, stateList, settings, frac=0.5):
    x_train_a = x_train[:int(frac*len(x_train))]
    y_train_a = y_train[:int(frac*len(y_train))]
    
    print("Training model A...")
    A_model = NN_Model(settings)
    if not os.path.isfile(settings['dir_name'] + "A_model_" + str(frac) + ".h5"):
        A_model.fit(x_train_a, y_train_a, settings)
        A_model.model.save(settings['dir_name'] + "A_model_" + str(frac) + ".h5")
    else:
        A_model.model = load_model(settings['dir_name'] + "A_model_" + str(frac) + ".h5") 

    pred_a_oh = A_model.model.predict(x_train[int(frac*len(x_train)):])

    y_train_b = np.concatenate((y_train_a, pred_a_oh), axis=0)

    print("Training model B...")
    B_model = NN_Model(settings)
    if not os.path.isfile(settings['dir_name'] + "B_model_" + str(frac) + ".h5"):
        B_model.fit(x_train, y_train_b, settings)
        B_model.model.save(settings['dir_name'] + "B_model_" + str(frac) + ".h5")
    else:
        B_model.model = load_model(settings['dir_name'] + "B_model_" + str(frac) + ".h5")
    
    print("Training model C...")
    C_model = NN_Model(settings)
    if not os.path.isfile(settings['dir_name'] + 'C_model.h5'):
        C_model.fit(x_train, y_train, settings)
        C_model.model.save(settings['dir_name'] + "C_model.h5")
    else:
        C_model.model = load_model(settings['dir_name'] + "C_model.h5")
        

    print("-> Evaluating model B on test data <-")
    test_acc_B = B_model.predict_merge(x_test, y_test, stateList)
    print("Model B (Accuracy): " + str(test_acc_B))

    print("-> Evaluating model C on test data <-")
    test_acc_C = C_model.predict_merge(x_test, y_test, stateList)
    print("Model C (Accuracy): " + str(test_acc_C))

    print("-> Evaluating model A on test data <-")
    test_acc_A = A_model.predict_merge(x_test, y_test, stateList)
    print("Model A (Accuracy): " + str(test_acc_A))
    
    return test_acc_A, test_acc_B, test_acc_C


if __name__ == '__main__':
    option = "FC1"
    feature_type = "lmfcc"

    x_train, x_test, y_train, y_test, stateList, output_dim = get_data(feature_type)

    settings = {
        "option" : option,
        "output_dim" : output_dim,
        "learning_rate" : None,
        "batch_size" : None,
        "timesteps" : len(x_train),
        "features" : len(x_train[0]),
        "n_epochs" : 5,
        "feature_type" : feature_type
    }

    dir_name = settings['option'] + "_" + settings['feature_type'] + "/"
    settings['dir_name'] = dir_name

    if settings['option'] == "CNN1":
        x_train = np.expand_dims(x_train, axis=2)
        x_test = np.expand_dims(x_test, axis=2)
    elif settings['option'] == "CNN2":
        x_train = np.expand_dims(np.expand_dims(x_train, axis=2), axis=3)
        x_test = np.expand_dims(np.expand_dims(x_test, axis=2), axis=3)
    else:
        pass

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
        test_acc_A, test_acc_B, test_acc_C = semi_supervised(x_train, y_train, x_test, y_test, stateList, settings, frac)
        hist_acc_A[i] = test_acc_A
        hist_acc_B[i] = test_acc_B
        hist_acc_C[i] = test_acc_C

    print("-------------------- PRINTOUTS --------------------------")

    print(hist_acc_A)
    print(hist_acc_B)
    print(hist_acc_C)

    np.savetxt(dir_name + "hist_acc_A.txt", hist_acc_A, delimiter=',')
    np.savetxt(dir_name + "hist_acc_B.txt", hist_acc_B, delimiter=',')
    np.savetxt(dir_name + "hist_acc_C.txt", hist_acc_C, delimiter=',')