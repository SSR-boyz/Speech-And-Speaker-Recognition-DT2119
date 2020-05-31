import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers, Input
from keras.models import Sequential
from keras.layers import Dense, Masking, Bidirectional, TimeDistributed, LSTM, Conv1D, Flatten, Conv2D
from string import digits
from tqdm import tqdm

class NN_Model:
    def __init__(self, settings):
        if settings['option'] == "FC1":
            self.model = Sequential([
                Dense(2000, input_dim=settings['features'], activation='relu'),
                Dense(1000, activation='relu'),
                Dense(1000, activation='relu'),
                Dense(settings['output_dim'], activation='softmax')
            ])

            self.model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        
        # CNN1 inspiration: https://www.researchgate.net/publication/317379506_Automatic_Speech_Recognition_using_different_Neural_Network_Architectures_-_A_Survey

        if settings['option'] == "CNN1":
            self.model = Sequential()

            self.model.add(Conv1D(32, 25, activation='relu', input_shape=(settings['features'], 1)))

            self.model.add(Conv1D(64, 25, activation='relu'))

            self.model.add(Conv1D(96, 25, activation='relu'))

            self.model.add(Conv1D(96, 25, activation='relu', padding='same'))

            self.model.add(Flatten())

            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dense(128, activation='relu'))

            self.model.add(Dense(settings['output_dim'], activation='softmax'))

            if settings['learning_rate'] == None:
                optimizer = tf.keras.optimizers.Adam()
            else:
               optimizer = tf.keras.optimizers.Adam(learning_rate=settings['learning_rate'])
            
            self.model.compile(optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        
        # CNN2 inspiration from: http://cs229.stanford.edu/proj2017/final-reports/5244201.pdf

        if settings['option'] == "CNN2":
            self.model = Sequential()

            self.model.add(Conv2D(filters=64, kernel_size=(20, 8), strides=(1, 3), activation='relu', input_shape=(settings['features'], 1, 1), padding='same'))

            self.model.add(Conv2D(filters=64, kernel_size=(10, 4), strides=(1,1), activation='relu', padding='same'))

            self.model.add(Flatten())

            self.model.add(Dense(128, activation='relu'))

            self.model.add(Dense(settings['output_dim'], activation='softmax'))

            if settings['learning_rate'] == None:
                    optimizer = tf.keras.optimizers.Adam()
            else:
               optimizer = tf.keras.optimizers.Adam(learning_rate=settings['learning_rate'])
            
            self.model.compile(optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        
    def fit(self, data, labels, settings):
        self.model.fit(data, labels, epochs=settings['n_epochs'])
    
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

    
