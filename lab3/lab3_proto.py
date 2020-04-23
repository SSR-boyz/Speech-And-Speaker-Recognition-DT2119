import numpy as np
from lab3_tools import *
from lab1_proto import mfcc
from prondict import prondict
from lab2_proto import *
import os

phoneHMMs = np.load('../lab2/lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()

def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
   """ word2phones: converts word level to phone level transcription adding silence

   Args:
      wordList: list of word symbols
      pronDict: pronunciation dictionary. The keys correspond to words in wordList
      addSilence: if True, add initial and final silence
      addShortPause: if True, add short pause model "sp" at end of each word
   Output:
      list of phone symbols
   """
   sequence = []
   for digit in wordList:
      sequence.append(pronDict[digit])
      if addShortPause:
         sequence.append(['sp'])
   
   flat_list = []
   for sublist in sequence:
      for item in sublist: 
         flat_list.append(item)

   if addSilence:
      flat_list = ['sil'] + flat_list + ['sil']

   return flat_list

def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
   """ forcedAlignmen: aligns a phonetic transcription at the state level

   Args:
      lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
            computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

   Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
   """
   utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)

   obsloglik = log_multivariate_normal_density_diag(lmfcc, utteranceHMM['means'], utteranceHMM['covars'])
   path, best_score = viterbi(obsloglik, utteranceHMM['startprob'][:-1], utteranceHMM['transmat'][:-1, :-1])

   phones = sorted(phoneHMMs.keys())
   nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}

   stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans
             for stateid in range(nstates[phone])]
   
   phoneme_path = [stateTrans[int(idx)] for idx in path]

   return phoneme_path


def hmmLoop(hmmmodels, namelist=None):
   """ Combines HMM models in a loop

   Args:
      hmmmodels: list of dictionaries with the following keys:
         name: phonetic or word symbol corresponding to the model
         startprob: M+1 array with priori probability of state
         transmat: (M+1)x(M+1) transition matrix
         means: MxD array of mean vectors
         covars: MxD array of variances
      namelist: list of model names that we want to combine, if None,
            all the models in hmmmodels are used

   D is the dimension of the feature vectors
   M is the number of emitting states in each HMM model (could be
      different in each model)

   Output
      combinedhmm: dictionary with the same keys as the input but
                  combined models
      stateMap: map between states in combinedhmm and states in the
               input models.

   Examples:
      phoneLoop = hmmLoop(phoneHMMs)
      wordLoop = hmmLoop(wordHMMs, ['o', 'z', '1', '2', '3'])
   """

def extraction_and_forced_alignment(samples, filename):
   lmfcc, mspec = mfcc(samples)
   wordTrans = list(path2info(filename)[2])
   phoneTrans = words2phones(wordTrans, prondict)
   targets = forcedAlignment(lmfcc, phoneHMMs, phoneTrans) #Tror detta är targets
   return lmfcc, mspec, targets

def generate_data():
   training_data = None
   test_data = None
   val_data = None

   try: 
      print("----------Trying to load file: traindata.npz----------")
      training_data = np.load("traindata.npz", allow_pickle=True)
      training_data = training_data['traindata']
      print("-----------File found: loaded training data-----------")
   except IOError:
      print("-----File not found: reading in the training data-----")
      traindata = []
      it = 0
      for root, dirs, files in os.walk('../../tidigits/disc_4.1.1/tidigits/train'):
         for file_ in files:
            if file_.endswith('.wav'):
               filename = os.path.join(root, file_)
               samples, samplingrate = loadAudio(filename)
               
               lmfcc, mspec, targets = extraction_and_forced_alignment(samples, filename)

               traindata.append({'filename': filename, 'lmfcc': lmfcc, 'mspec': mspec, 'targets': targets})
            it = it + 1
            print("Train: " + str(it) + " of " + str(len(files) * len(dirs)))
      np.savez('traindata.npz', traindata=traindata)
      training_data = traindata
   
   try:
      print("-----------Trying to load file: testdata.npz-----------")
      test_data = np.load("testdata.npz", allow_pickle=True)
      test_data = test_data['testdata']
      print("------------File found: loaded testing data------------")
   except IOError:
      print("------File not found: reading in the testing data------")
      it = 0
      testdata = []
      for root, dirs, files in os.walk('../../tidigits/disc_4.2.1/tidigits/test'):
         for file_ in files:
            if file_.endswith('.wav'):
               filename = os.path.join(root, file_)
               samples, samplingrate = loadAudio(filename)
               
               lmfcc, mspec, targets = extraction_and_forced_alignment(samples, filename)

               testdata.append({'filename': filename, 'lmfcc': lmfcc, 'mspec': mspec, 'targets': targets})
            it = it + 1
            print("Test: " + str(it) + " of " + str(len(files) * len(dirs)))
      np.savez('testdata.npz', testdata=testdata)
      test_data = testdata
   
   try:
      print("-----------Trying to load file: valdata.npz-----------")
      val_data = np.load("valdata.npz", allow_pickle=True)
      val_data = val_data['valdata']
      print("------------File found: loaded validation data------------")
   except IOError:
      print("------File not found: creating the validation data------")
      speakerIDs = ['ae', 'aj', 'al', 'aw', 'bd', 'ac', 'ag', 'ai', 'an', 'bh', 'bi']
      valdata = []
      traindata = []
      for i in range(len(training_data)):
         gender, speakerID, digits, repetition = path2info(training_data[i]['filename'])
         if speakerID in speakerIDs:
            valdata.append(training_data[i])
         else:
            traindata.append(training_data[i])
      np.savez('traindata.npz', traindata=traindata)
      np.savez('valdata.npz', valdata=valdata)
      val_data = valdata
      training_data = traindata

   return training_data, val_data, test_data

def dynamic_features(train_data, val_data, test_data, stack_size=7):
   last_timestep, _ = train_data[0]['lmfcc'].shape
   feature_list = np.concatenate((np.array([3, 2, 1]), np.arange(last_timestep), np.array([65, 64, 63])), axis=None)

   """ TODO
   for sample in train_data:
      for timestep, feature in enumerate(sample):
            
            sample['lmfcc'][feature_list[timestep]]
   """

train_data, val_data, test_data = generate_data()

print(train_data[0].keys())
   
