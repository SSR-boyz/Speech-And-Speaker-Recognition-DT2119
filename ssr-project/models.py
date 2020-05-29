import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers, Input
from keras.models import Sequential
from keras.layers import Dense, Masking, Bidirectional, TimeDistributed, LSTM, Conv1D, Flatten
from string import digits
#from Levenshtein import distance
import stringdist
from tqdm import tqdm

#Levenshtein.distance
#levenshtein_distance

class NN_Model:
    def __init__(self, settings):
        if settings['option'] == "FC1":
            self.model = Sequential([
                Dense(256, input_dim=91, activation='relu'),
                Dense(256, activation='relu'),
                Dense(256, activation='relu'),
                Dense(settings['output_dim'], activation='softmax')
            ])

            self.model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        
        if settings['option'] == "CNN1":
            self.model = Sequential()

            self.model.add(Conv1D(16, 25, activation='relu', input_shape=(settings['features'], 1)))

            self.model.add(Conv1D(32, 25, activation='relu'))

            self.model.add(Conv1D(48, 25, activation='relu'))

            self.model.add(Conv1D(48, 25, activation='relu', padding='same'))

            self.model.add(Flatten())

            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dense(64, activation='relu'))

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
    
    def predict_phoneme_distance(self, test_data, labels, stateList):
        predictions_oh = self.model.predict(test_data)

        decodeChars = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E',
        'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
        'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8'])

        for i in range(len(stateList)):
            stateList[i] = stateList[i][:-2]

        stateList_unique = np.unique(stateList)
        stateList_unique = stateList_unique.tolist()

        #PREDICTIONS
        predictions = np.argmax(predictions_oh, axis=1)

        prev_p = -1
        pred = []
        for p in predictions:
            if p != prev_p:
                pred.append(int(p))
                prev_p = p
        #pred = np.array(pred)
        #sil

        pred_code = []
        for i in range(len(pred)):
            pred_code.append(decodeChars[stateList_unique.index(stateList[pred[i]])])
        
        #TARGETS
        targets = np.argmax(labels, axis=1)
        
        prev_p = -1
        targ = []
        for p in targets:
            if p != prev_p:
                targ.append(int(p))
                prev_p = p
        #targ = np.array(targ)
        targ_code = []
        for i in range(len(targ)):
            targ_code.append(decodeChars[stateList_unique.index(stateList[targ[i]])])
        #targ_code = decodeChars[targ]

        targ = "".join(targ_code)
        pred = "".join(pred_code)
        
        #print("Targ Length: " + str(len(targ)))
        #print("Pred Length: " + str(len(pred)))

        #print("Targ Content: " + str(targ[:100]))
        #print("Pred Content: " + str(pred[:100]))
        diff = stringdist.levenshtein(targ, pred)
        print("Distance: " + str(diff))
        return float(diff/len(targ))

    def predict_state_distance(self, test_data, labels, stateList):
        predictions_oh = self.model.predict(test_data)
        
        decodeChars = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E',
        'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
        'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8'])
        codeList = np.zeros((stateList.shape), dtype="object")
        #stateList[codeList.index('a')]
        for i in range(len(codeList)):
            codeList[i] = decodeChars[i]
        
        predictions = np.argmax(predictions_oh, axis=1)
        phones = codeList[predictions]
        prev_p = ""
        pred = ""
        for p in phones:
            if p != prev_p:
                pred += p
                prev_p = p
        
        prev_p = ""
        targets = np.argmax(labels, axis=1)
        result_targets = codeList[targets]
        targ = ""
        for p in result_targets:
            if p != prev_p:
                targ += p
                prev_p = p

        print("Targ Length: " + str(len(targ)))
        print("Pred Length: " + str(len(pred)))

        print("Targ Content: " + str(targ[:100]))
        print("Pred Content: " + str(pred[:100]))
        print(pred)
        diff = distance(targ, pred)
        print("Distance: " + str(diff))
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
        
        loss = 0
        return loss, accuracy
        
    def evaluate(self, test_data, test_labels):
        test_loss, test_acc = self.model.evaluate(test_data, test_labels, verbose=2)
        return test_loss, test_acc

    
