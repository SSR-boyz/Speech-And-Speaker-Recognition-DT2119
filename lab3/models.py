import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
from string import digits
from Levenshtein import distance as levenshtein_distance

class NN_Model:
    def __init__(self, output_dim, option=None):
        self.option = option
        if option == "lmfcc": 
            self.model = Sequential([
                Dense(256, input_dim=13, activation='relu'),
                Dense(256, activation='relu'),
                Dense(256, activation='relu'),
                Dense(output_dim, activation='sigmoid')
            ])

            self.model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        elif option == "mspec":
            self.model = Sequential([
                Dense(256, input_dim=40, activation='relu'),
                Dense(256, activation='relu'),
                Dense(256, activation='relu'),
                Dense(output_dim, activation='softmax')
            ])

            self.model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        elif option == "dyn_lmfcc":
            self.model = Sequential([
                Dense(256, input_dim=91, activation='relu'),
                Dense(256, activation='relu'),
                Dense(256, activation='relu'),
                Dense(output_dim, activation='softmax')
            ])

            self.model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        elif option == "dyn_mspec":
            self.model = Sequential([
                Dense(256, input_dim=280, activation='relu'),
                Dense(256, activation='relu'),
                Dense(256, activation='relu'),
                Dense(output_dim, activation='softmax')
            ])

            self.model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    def fit(self, data, labels, n_epochs=10):
        self.model.fit(data, labels, epochs=n_epochs)
    
    def predict_state_distance(self, test_data, labels, stateList):
        predictions_oh = self.model.predict(test_data)
        predictions = np.argmax(predictions_oh, axis=1)
        phones = stateList[predictions]
        prev_p = ""
        pred = ""
        for p in phones:
            if p != prev_p:
                pred += p
                prev_p = p
        
        prev_p = ""
        targets = np.argmax(labels, axis=1)
        result_targets = stateList[targets]
        targ = ""
        for p in result_targets:
            if p != prev_p:
                targ += p
                prev_p = p

        
        diff = levenshtein_distance(targ, pred)

        """
        len = std::max(s1.length(), s2.length());
        // normalize by length, high score wins
        fDist = float(len - levenshteinDistance(s1, s2)) / float(len);
        """
        
        return float(diff/len(targ))
        
    def predict_merge(self, test_data, labels, stateList):
        predictions_oh = self.model.predict(test_data)
        predictions = np.argmax(predictions_oh, axis=1)
        phones = stateList[predictions]
        for j,s in enumerate(phones):
            remove_digits = str.maketrans('', '', digits)
            phones[j] = s.translate(remove_digits)

        targets = np.argmax(labels, axis=1)
        result_targets = stateList[targets]
        for j,s in enumerate(result_targets):
            remove_digits = str.maketrans('', '', digits)
            result_targets[j] = s.translate(remove_digits)
        
        accuracy = float(np.sum(phones == result_targets)/len(result_targets))
        
        return accuracy
        
    def evaluate(self, test_data, test_labels):
        test_loss, test_acc = self.model.evaluate(test_data, test_labels, verbose=2)
        return test_loss, test_acc

    
