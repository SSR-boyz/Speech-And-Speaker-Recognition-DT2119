import numpy as np
from lab3_tools import *
from lab1_proto import mfcc
from prondict import prondict
from lab2_proto import *
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils
from models import *
from keras.models import load_model


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
    pass

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

	stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans for stateid in range(nstates[phone])]
	
	phoneme_path = [stateTrans[int(idx)] for idx in path]

	return phoneme_path

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
		print("--> Trying to load file: traindata.npz")
		training_data = np.load("traindata.npz", allow_pickle=True)
		#training_data = training_data['traindata']
		print("--> Success")
	except IOError:
		print("-----File not found: reading in the training data-----")
		traindata = []
		for root, dirs, files in os.walk('../../tidigits/disc_4.1.1/tidigits/train'):
			for file_ in tqdm(files):
				if file_.endswith('.wav'):
					filename = os.path.join(root, file_)
					samples, samplingrate = loadAudio(filename)
					
					lmfcc, mspec, targets = extraction_and_forced_alignment(samples, filename)

					traindata.append({'filename': filename, 'lmfcc': lmfcc, 'mspec': mspec, 'targets': targets})
		np.savez('traindata.npz', traindata=traindata)
		training_data = traindata
	
	try:
		print("--> Trying to load file: testdata.npz")
		test_data = np.load("testdata.npz", allow_pickle=True)
		#test_data = test_data['testdata']
		print("--> Success")
	except IOError:
		print("------File not found: reading in the testing data------")
		testdata = []
		for root, dirs, files in os.walk('../../tidigits/disc_4.2.1/tidigits/test'):
			for file_ in tqdm(files):
				if file_.endswith('.wav'):
					filename = os.path.join(root, file_)
					samples, samplingrate = loadAudio(filename)
					
					lmfcc, mspec, targets = extraction_and_forced_alignment(samples, filename)

					testdata.append({'filename': filename, 'lmfcc': lmfcc, 'mspec': mspec, 'targets': targets})
		np.savez('testdata.npz', testdata=testdata)
		test_data = testdata
	
	try:
		print("--> Trying to load file: valdata.npz")
		val_data = np.load("valdata.npz", allow_pickle=True)
		#val_data = val_data['valdata']
		print("--> Success")
	except IOError:
		print("------File not found: creating the validation data------")
		speakerIDs = ['ae', 'aj', 'al', 'aw', 'bd', 'ac', 'ag', 'ai', 'an', 'bh', 'bi']
		valdata = []
		traindata = []
		for i in tqdm(range(len(training_data))):
			gender, speakerID, digits, repetition = path2info(training_data[i]['filename'])
			if speakerID in speakerIDs:
				valdata.append(training_data[i])
			else:
				traindata.append(training_data[i])
		np.savez('traindata.npz', traindata=traindata)
		np.savez('valdata.npz', valdata=valdata)
		val_data = valdata
		training_data = traindata

	return training_data['traindata'], val_data['valdata'], test_data['testdata']

def dynamic_features(train_data=None, val_data=None, test_data=None, stack_size=7):

	traindata_dyn = None
	testdata_dyn = None
	valdata_dyn = None

	try:
		print("--> Trying to load file: traindata_dyn.npz")
		traindata_dyn = np.load("traindata_dynn.npz", allow_pickle=True)['traindata_dyn']
		print("--> Success")
	except IOError:
		print("-----------File not found, creating traindata_dyn.npz-----------")
		for sample in tqdm(train_data):
			for key in sample.keys():
				if key == "filename":
					continue
				last_timestep = len(sample[key])
				feature_list = np.concatenate((np.array([3, 2, 1]), np.arange(last_timestep), np.array([last_timestep-2, last_timestep-3, last_timestep-4])), axis=None)
				if key == "targets":
					dyn_list = np.zeros((last_timestep, stack_size), dtype="object")
					dyn_feat = np.empty((stack_size), dtype="object")
				elif key == "lmfcc":
					dyn_list = np.zeros((last_timestep, stack_size, 13), dtype="object")
					dyn_feat = np.empty((stack_size, len(sample[key][0])))
				else: #mspec
					dyn_list = np.zeros((last_timestep, stack_size, 40), dtype="object")
					dyn_feat = np.empty((stack_size, len(sample[key][0])))
				
				for timestep in range(len(sample[key])):
					dyn_feat[0] = sample[key][feature_list[timestep]]
					dyn_feat[1] = sample[key][feature_list[timestep+1]]
					dyn_feat[2] = sample[key][feature_list[timestep+2]]
					dyn_feat[3] = sample[key][feature_list[timestep+3]]
					dyn_feat[4] = sample[key][feature_list[timestep+4]]
					dyn_feat[5] = sample[key][feature_list[timestep+5]]
					dyn_feat[6] = sample[key][feature_list[timestep+6]]
					dyn_list[timestep] = dyn_feat

				sample[key] = np.array(dyn_list)
				
		traindata_dyn = train_data
		np.savez('traindata_dynn.npz', traindata_dyn=traindata_dyn)
	
	try:
		print("--> Trying to load file: testdata_dyn.npz")
		testdata_dyn = np.load("testdata_dynn.npz", allow_pickle=True)['testdata_dyn']
		print("--> Success")
	except IOError:
		print("-----------File not found, creating testdata_dyn.npz-----------")
		for sample in tqdm(test_data):
			for key in sample.keys():
				if key == "filename":
					continue
				last_timestep = len(sample[key])
				feature_list = np.concatenate((np.array([3, 2, 1]), np.arange(last_timestep), np.array([last_timestep-2, last_timestep-3, last_timestep-4])), axis=None)
				
				if key == "targets":
					dyn_list = np.zeros((last_timestep, stack_size), dtype="object")
					dyn_feat = np.empty((stack_size), dtype="object")
				elif key == "lmfcc":
					dyn_list = np.zeros((last_timestep, stack_size, 13), dtype="object")
					dyn_feat = np.empty((stack_size, len(sample[key][0])))
				else: #mspec
					dyn_list = np.zeros((last_timestep, stack_size, 40), dtype="object")
					dyn_feat = np.empty((stack_size, len(sample[key][0])))

				for timestep in range(len(sample[key])):
					dyn_feat[0] = sample[key][feature_list[timestep]]
					dyn_feat[1] = sample[key][feature_list[timestep+1]]
					dyn_feat[2] = sample[key][feature_list[timestep+2]]
					dyn_feat[3] = sample[key][feature_list[timestep+3]]
					dyn_feat[4] = sample[key][feature_list[timestep+4]]
					dyn_feat[5] = sample[key][feature_list[timestep+5]]
					dyn_feat[6] = sample[key][feature_list[timestep+6]]
					dyn_list[timestep] = dyn_feat

				sample[key] = np.array(dyn_list)
				
				
		testdata_dyn = test_data
		np.savez('testdata_dynn.npz', testdata_dyn=testdata_dyn)

	try:
		print("--> Trying to load file: valdata_dyn.npz")
		valdata_dyn = np.load("valdata_dynn.npz", allow_pickle=True)['valdata_dyn']
		print("--> Success")
	except IOError:
		print("-----------File not found, creating valdata_dyn.npz-----------")
		for sample in tqdm(val_data):
			for key in sample.keys():
				if key == "filename":
					continue
				last_timestep = len(sample[key])
				feature_list = np.concatenate((np.array([3, 2, 1]), np.arange(last_timestep), np.array([last_timestep-2, last_timestep-3, last_timestep-4])), axis=None)
				
				if key == "targets":
					dyn_list = np.zeros((last_timestep, stack_size), dtype="object")
					dyn_feat = np.empty((stack_size), dtype="object")
				elif key == "lmfcc":
					dyn_list = np.zeros((last_timestep, stack_size, 13), dtype="object")
					dyn_feat = np.empty((stack_size, len(sample[key][0])))
				else: #mspec
					dyn_list = np.zeros((last_timestep, stack_size, 40), dtype="object")
					dyn_feat = np.empty((stack_size, len(sample[key][0])))

				for timestep in range(len(sample[key])):
					dyn_feat[0] = sample[key][feature_list[timestep]]
					dyn_feat[1] = sample[key][feature_list[timestep+1]]
					dyn_feat[2] = sample[key][feature_list[timestep+2]]
					dyn_feat[3] = sample[key][feature_list[timestep+3]]
					dyn_feat[4] = sample[key][feature_list[timestep+4]]
					dyn_feat[5] = sample[key][feature_list[timestep+5]]
					dyn_feat[6] = sample[key][feature_list[timestep+6]]
					dyn_list[timestep] = dyn_feat
				
				sample[key] = np.array(dyn_list)
					
		valdata_dyn = val_data
		np.savez('valdata_dynn.npz', valdata_dyn=valdata_dyn)
	
	return traindata_dyn, valdata_dyn, testdata_dyn


# Feature Standardisation
def process_data(traindata_dyn, valdata_dyn, testdata_dyn):
	print("Processing data")
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

def normalise_data():
    	
	traindata = np.load("traindata.npz", allow_pickle=True)['traindata']
	valdata = np.load("valdata.npz", allow_pickle=True)['valdata']
	testdata = np.load("testdata.npz", allow_pickle=True)['testdata']
	

	
	lmfcc_train_x = extract_data(traindata, "lmfcc")
	lmfcc_val_x = extract_data(valdata, "lmfcc")
	lmfcc_test_x = extract_data(testdata, "lmfcc")

	mspec_train_x = extract_data(traindata, "mspec")
	mspec_val_x = extract_data(valdata, "mspec")
	mspec_test_x = extract_data(testdata, "mspec")
	"""
	lmfcc_train_x = np.array([np.array(traindata[i]['lmfcc']) for i in range(len(traindata))])
	lmfcc_val_x = np.array([np.array(valdata[i]['lmfcc']) for i in range(len(valdata))])
	lmfcc_test_x = np.asmatrix([np.asmatrix(testdata[i]['lmfcc']) for i in range(len(testdata))])
	
	mspec_train_x = np.array([np.array(traindata[i]['mspec']) for i in range(len(traindata))])
	mspec_val_x = np.array([np.array(valdata[i]['mspec']) for i in range(len(valdata))])
	mspec_test_x = np.array([np.array(testdata[i]['mspec']) for i in range(len(testdata))])
	"""
	print(lmfcc_test_x.shape)
	scaler = StandardScaler()
	scaler.fit(lmfcc_train_x)
	lmfcc_test_x = scaler.transform(lmfcc_test_x)
	lmfcc_train_x = scaler.transform(lmfcc_train_x)
	lmfcc_val_x = scaler.transform(lmfcc_val_x)

	scaler = StandardScaler()
	scaler.fit(mspec_train_x)
	mspec_test_x = scaler.transform(mspec_test_x)
	mspec_train_x = scaler.transform(mspec_train_x)
	mspec_val_x = scaler.transform(mspec_val_x)
	

	return lmfcc_test_x, lmfcc_train_x, lmfcc_val_x, mspec_test_x, mspec_train_x, mspec_val_x

# OK
def normalise_data_dyn():
	lmfcc_test_x = np.load("data/lmfcc_test_x.npz", allow_pickle=True)['lmfcc_test_x']
	lmfcc_train_x = np.load("data/lmfcc_train_x.npz", allow_pickle=True)['lmfcc_train_x']
	lmfcc_val_x = np.load("data/lmfcc_val_x.npz", allow_pickle=True)['lmfcc_val_x']

	mspec_test_x = np.load("data/mspec_test_x.npz", allow_pickle=True)['mspec_test_x']
	mspec_train_x = np.load("data/mspec_train_x.npz", allow_pickle=True)['mspec_train_x']
	mspec_val_x = np.load("data/mspec_val_x.npz", allow_pickle=True)['mspec_val_x']

	scaler = StandardScaler()
	scaler.fit(lmfcc_train_x)
	lmfcc_test_x = scaler.transform(lmfcc_test_x)
	lmfcc_train_x = scaler.transform(lmfcc_train_x)
	lmfcc_val_x = scaler.transform(lmfcc_val_x)

	scaler = StandardScaler()
	scaler.fit(mspec_train_x)
	mspec_test_x = scaler.transform(mspec_test_x)
	mspec_train_x = scaler.transform(mspec_train_x)
	mspec_val_x = scaler.transform(mspec_val_x)

	return lmfcc_test_x, lmfcc_train_x, lmfcc_val_x, mspec_test_x, mspec_train_x, mspec_val_x

# OK
def targets_to_int(train_y, val_y, test_y, stateList):
	train_y_int = np.zeros((len(train_y)))
	val_y_int = np.zeros((len(val_y)))
	test_y_int = np.zeros((len(test_y)))

	for i in tqdm(range(len(train_y))):
		train_y_int[i] = stateList.index(train_y[i])
	
	
	for i in tqdm(range(len(val_y))):
         val_y_int[i] = stateList.index(val_y[i])

	for i in tqdm(range(len(test_y))):
         test_y_int[i] = stateList.index(test_y[i])

	np.savez("data/train_y_int.npz", train_y_int=train_y_int)
	np.savez("data/val_y_int.npz", val_y_int=val_y_int)
	np.savez("data/test_y_int.npz", test_y_int=test_y_int)
	np.savez("data/stateList.npz", stateList=stateList)
	return train_y_int, val_y_int, test_y_int

# OK
def to_categorical_targets():
	try:
		train_y = np.load("data/train_y_int.npz", allow_pickle=True)['train_y_int']
		val_y = np.load("data/val_y_int.npz", allow_pickle=True)['val_y_int']
		test_y = np.load("data/test_y_int.npz", allow_pickle=True)['test_y_int']
		stateList = np.load("data/stateList.npz", allow_pickle=True)['stateList']
		output_dim = len(stateList)
	except IOError:
		train_y = np.load("data/train_y.npz", allow_pickle=True)['train_y']
		val_y = np.load("data/val_y.npz", allow_pickle=True)['val_y']
		test_y = np.load("data/test_y.npz", allow_pickle=True)['test_y']
		
		phones = sorted(phoneHMMs.keys())
		nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
		stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]
		output_dim = len(stateList)
		
		train_y, val_y, test_y = targets_to_int(train_y, val_y, test_y, stateList)

	train_y = np_utils.to_categorical(train_y, output_dim)
	val_y = np_utils.to_categorical(val_y, output_dim)
	test_y = np_utils.to_categorical(test_y, output_dim)

	return train_y, val_y, test_y, output_dim, stateList


#train_data, val_data, test_data = generate_data()
#traindata_dyn, valdata_dyn, testdata_dyn = dynamic_features(); #dynamic_features(train_data, val_data, test_data)

#process_data(traindata_dyn, valdata_dyn, testdata_dyn)

'''print("Normalising data")
lmfcc_test_x, lmfcc_train_x, lmfcc_val_x, mspec_test_x, mspec_train_x, mspec_val_x = normalise_data_dyn()
print("Making targets categorical")
train_y, val_y, test_y, output_dim, stateList = to_categorical_targets()


print("Starting training for dyn_lmfcc")

model = NN_Model(output_dim, "dyn_lmfcc")
model.model = load_model('lmfcc_model.h5')
#model.fit(lmfcc_train_x, train_y, n_epochs=10)
#test_loss, test_acc = model.evaluate(lmfcc_test_x, test_y)
test_acc = model.predict_phoneme_distance(lmfcc_test_x, test_y, stateList)
print("Phone Error Rate (Phoneme Level): " + str(test_acc))
#model.model.save('lmfcc_model.h5')
#print("Test loss: " + str(test_loss) + " , " + "Test accuracy: " + str(test_acc))'''
