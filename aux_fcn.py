import scipy.io
import pandas as pd
import numpy as np
import h5py
import sys
import os
import math
import pickle
from tensorflow import keras
from keras.models import Sequential
from keras import layers, optimizers
from keras.initializers import GlorotUniform, Orthogonal
from xgboost import XGBClassifier

def fcn_save_pickle(name,x):
	'''
	void fcn_save_pickle(name,x) \n
	Generates a pickle file in path n, containing variable x\n
	'''
	with open(name, 'wb') as handle:
			pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
	return 

def fcn_load_pickle(name):
	'''
	[x] = fcn_load_pickle(name) loads the content of the pickle file to x
	'''
	with open(name, 'rb') as handle:
    		return( pickle.load(handle) )


def loadChunk(fid, nChannels, channels, nSamples, precision):
	size = int(nChannels * nSamples * precision)
	nSamples = int(nSamples)

	data = fid.read(size)

	# fromstring to read the data as int16
	# reshape to give it the appropiate shape (nSamples x nChannels)
	data = np.fromstring(data, dtype=np.int16).reshape(nSamples, len(channels))
	data = data[:, channels]

	return data

def bz_LoadBinary(filename, nChannels, channels, sampleSize, verbose=False):

	if (len(channels) > nChannels):
		print("Cannot load specified channels (listed channel IDs inconsistent with total number of channels).")
		return

	with open(filename, "rb") as f:
		dataOffset = 0

		# Determine total number of samples in file
		fileStart = f.tell()
		if verbose:
			print("fileStart ", fileStart)
		status = f.seek(0, 2) # Go to the end of the file
		fileStop = f.tell()
		f.seek(0, 0) # Back to the begining
		if verbose:
			print("fileStop ", fileStop)

		# (floor in case all channels do not have the same number of samples)
		maxNSamplesPerChannel = math.floor(((fileStop-fileStart)/nChannels/sampleSize))
		nSamplesPerChannel = maxNSamplesPerChannel

		# For large amounts of data, read chunk by chunk
		maxSamplesPerChunk = 10000
		nSamples = int(nSamplesPerChannel*nChannels)

		if verbose:
			print("nSamples ", nSamples)

		if nSamples <= maxNSamplesPerChannel:
			data = loadChunk(f, nChannels, channels, nSamples, sampleSize)
		else:
			# Determine chunk duration and number of chunks
			nSamplesPerChunk = math.floor(maxSamplesPerChunk/nChannels)*nChannels
			nChunks = math.floor(nSamples/nSamplesPerChunk)

			if verbose:
				print("nSamplesPerChannel ", nSamplesPerChannel)
				print("nSamplesPerChunk ", nSamplesPerChunk)

			# Preallocate memory
			data = np.zeros((nSamplesPerChannel,len(channels)), dtype=np.int16)

			if verbose:
				print("size data ", np.size(data, 0))

			# Read all chuncks
			i = 0
			for j in range(nChunks):
				d = loadChunk(f, nChannels, channels, nSamplesPerChunk/nChannels, sampleSize)
				m = np.size(d, 0)

				if m == 0:
					break

				data[i:i+m, :] = d
				i = i+m

			# If the data size is not a multiple of the chunk size, read the remainder
			remainder = nSamples - nChunks*nSamplesPerChunk
			if remainder != 0:
				d = loadChunk(f, nChannels, channels, remainder/nChannels, sampleSize)
				m = np.size(d, 0)

				if m != 0:
					data[i:i+m, :] = d

	return data


# Functions used to load the raw LFP, select channels, load ripples, downsample and normalize
def load_info (path):
	try:
		mat = scipy.io.loadmat(os.path.join(path, "info.mat"))
	except:
		print("info.mat file does not exist.")
		sys.exit()

	sf = mat["fs"][0][0]
	expName = mat["expName"][0]

	ref_channels = {}
	ref_channels["so"] = mat["so"][0]
	ref_channels["pyr"] = mat["pyr"][0]
	ref_channels["rad"] = mat["rad"][0]
	ref_channels["slm"] = mat["slm"][0]


	if len(mat["chDead"]) <= 0:
		dead_channels = []
	else:
		dead_channels = [x-1 for x in (mat["chDead"][0]).astype(int)]

	return sf, expName, ref_channels, dead_channels

def load_ripples (path, verbose=False):
	try:
		dataset = pd.read_csv(os.path.join(path,"ripples.csv"), delimiter=' ', header=0, usecols = ["ripIni","ripEnd"])# "ripMiddle", "ripEnd", "type", "shank"])
	except:
		print(path+"ripples.csv file does not exist.")
		sys.exit()

	ripples = dataset.values
	ripples = ripples[np.argsort(ripples, axis=0)[:, 0], :]
	if verbose:
		print("Loaded ripples: ", len(ripples))

	return ripples

def load_channels_map (path):
	try:
		dataset = pd.read_csv(path+"/mapsCh.csv", delimiter=' ', header=0)
	except:
		print("ripples.csv file does not exist.")
		sys.exit()

	channels_map = dataset.values

	return channels_map

def reformat_channels (channels_map, ref_channels):
	channels = np.where(np.isnan(channels_map[:, 0]) == False, channels_map[:, 0], 0)
	channels = [x-1 for x in (channels).astype(int)]

	shanks = np.where(np.isnan(channels_map[:, 1]) == False, channels_map[:, 1], 0)
	shanks = [x-1 for x in (shanks).astype(int)]

	ref_channels["so"] = np.where(np.isnan(ref_channels["so"]) == False, ref_channels["so"], 0)
	ref_channels["so"] = [x-1 for x in ref_channels["so"].astype(int)]
	ref_channels["pyr"] = np.where(np.isnan(ref_channels["pyr"]) == False, ref_channels["pyr"], 0)
	ref_channels["pyr"] = [x-1 for x in ref_channels["pyr"].astype(int)]
	ref_channels["rad"] = np.where(np.isnan(ref_channels["rad"]) == False, ref_channels["rad"], 0)
	ref_channels["rad"] = [x-1 for x in ref_channels["rad"].astype(int)]
	ref_channels["slm"] = np.where(np.isnan(ref_channels["slm"]) == False, ref_channels["slm"], 0)
	ref_channels["slm"] = [x-1 for x in ref_channels["slm"].astype(int)]

	return channels, shanks, ref_channels

def load_raw_data (path, expName, channels, verbose=False):
	
	# There is .dat file
	is_dat = any([file.endswith(".dat") for file in os.listdir(path)])

	# There is .eeg file
	is_eeg = any([file.endswith(".eeg") for file in os.listdir(path)])
	
	# There is .mat file with the name of the last folder
	is_mat = any([os.path.basename(os.path.normpath(path))+".mat" in file for file in os.listdir(path)])

	if is_dat:
		name_dat = os.listdir(path)[np.where([file.endswith(".dat") for file in os.listdir(path)])[0][0]]
		if verbose:
			print(path+"/"+name_dat)
		data = bz_LoadBinary(path+"/"+name_dat, len(channels), channels, 2, verbose)

	elif is_eeg:
		name_eeg = os.listdir(path)[np.where([file.endswith(".eeg") for file in os.listdir(path)])[0][0]]
		if verbose:
			print(path+"/"+name_eeg)
		data = bz_LoadBinary(path+"/"+name_eeg, len(channels), channels, 2, verbose)

	elif is_mat:
		folder = path + "/" + os.path.basename(os.path.normpath(path))+".mat"
		if verbose:
			print(folder)
		try:
			mat = scipy.io.loadmat(folder)
			data = mat["fil"]
		except:
			mat = h5py.File(folder, 'r')
			data = np.array(mat["fil"]).T
	else:
		print('Not data found')

	return data

def downsample_data (data, sf, downsampled_fs):

    # Dowsampling
    if sf > downsampled_fs:
        downsampled_pts = np.linspace(0, data.shape[0]-1, int(np.round(data.shape[0]/sf*downsampled_fs))).astype(int)
        downsampled_data = data[downsampled_pts, :]

    # Upsampling
    elif sf < downsampled_fs:
        print("Original sampling rate below 1250 Hz!")
        return None


    # Change from int16 to float16 if necessary
    # int16 ranges from -32,768 to 32,767
    # float16 has ±65,504, with precision up to 0.0000000596046
    if downsampled_data.dtype != 'float16':
        downsampled_data = np.array(downsampled_data, dtype="float16")

    return downsampled_data

def z_score_normalization(data):
	channels = range(np.shape(data)[1])

	for channel in channels:
		# Since data is in float16 type, we make it smaller to avoid overflows
		# and then we restore it.
		# Mean and std use float64 to have enough space
		# Then we convert the data back to float16
		dmax = np.amax(data[:, channel])
		dmin = abs(np.amin(data[:, channel]))
		dabs = dmax if dmax>dmin else dmin
		m = np.mean(data[:, channel] / dmax, dtype='float64') * dmax
		s = np.std(data[:, channel] / dmax, dtype='float64') * dmax
		s = 1 if s == 0 else s # If std == 0, change it to 1, so data-mean = 0
		data[:, channel] = ((data[:, channel] - m) / s).astype('float16')
	
	return data

def load_data_fs(path, shank, verbose=False):
	# Read info.mat
	sf, expName, ref_channels, dead_channels = load_info(path)

	#Read mapsCh.csv
	channels_map = load_channels_map(path)

	# Reformat channels into correct values
	channels, shanks, ref_channels = reformat_channels(channels_map, ref_channels)
	# Read .dat
	data = load_raw_data(path, expName, channels, verbose=verbose)


	return data, sf

def generate_overlapping_windows(data, window_size, stride, sf):
	window_pts = int(window_size * sf)
	stride_pts = int(stride * sf)
	r = range(0, data.shape[0], stride_pts)

	new_data = np.empty((len(list(r)), window_pts, data.shape[1]))

	cont = 0
	for idx in r:
		win = data[idx:idx+window_pts, :]

		if (win.shape[0] < window_pts):
			continue

		new_data[cont,:,:]  = win

		cont = cont+1

	return new_data


# Detection functions
def process_LFP(LFP,sf,channels):
    
    ''' 
	This function processes the LFP before calling the detection algorithm.
	1. It extracts the desired channels from the original LFP, and interpolates where there is a value of -1.
	2. Downsamples the LFP to 1250 Hz.
	3. Normalizes each channel separately by z-scoring them.

	Mandatory inputs:
		LFP: 		LFP recorded data (np.array: n_samples x n_channels).
		sf: 		sampling frequency (in Hz).
		channels: 	channel to which compute the undersampling and z-score normalization. Counting starts in 0. 
					If channels contains any -1, interpolation will be also applied. 
					See channels of rippl_AI.predict(), or aux_fcn.interpolate_channels() for more information.
	Output:
		LFP_norm: normalized LFP (np.array: n_samples x len(channels)). It is undersampled to 1250Hz, z-scored, 
					and transformed to used the channels specified in channels.
    A Rubio, LCN 2023
    '''
    data=interpolate_channels(LFP,channels)
    if sf!=1250:
        print('Downsampling data at 1250 Hz...')
        data = downsample_data(data, sf, downsampled_fs=1250)
        print("Shape of downsampled data:",data.shape)
    else:
        print("Data is already sampled at 1250 Hz!")
	
    print('Normalizing data...')
    normalized_data=z_score_normalization(data)
    return normalized_data

def prediction_parser(LFP,arch='CNN1D',model_number=1):
    '''
    [y] = prediction_parser(LFP,model_sel) 
    Computes the output of the model passed in params \n
    Inputs:		
        LFP:			 [n x 8] lfp data,subsampled and z-scored 
        arch:   		 string containing the name of the architecture 
        model_number:    int, if 1 the best model will be used to predict

    Output: 
        y: (n) shape array with the output of the chosen model
	A Rubio, LCN 2023
    '''
   
   
    # Looks for the name of the selected model
    for filename in os.listdir('optimized_models'):
        if f'{arch}_{model_number}' in filename:
            break
    print(filename)
    sp=filename.split('_')
    n_channels=int(sp[2][2])
    timesteps=int(sp[4][2:])

	
	#print(f'Validating arquitecture {arch} using {n_channels} channels and {timesteps} timesteps')
	# Input shape: number of channels

    input_len=LFP.shape[0]
    # Make sure the input data and the model number of 
    assert n_channels==LFP.shape[1],f'The model expects {n_channels} channels and the data has {LFP.shape[1]}'
    print(n_channels,LFP.shape[1])
	# Input shape: timesteps
    if arch=='XGBOOST':
        LFP=LFP[:len(LFP)-len(LFP)%timesteps,:].reshape(-1,timesteps*n_channels)
        y_predict= np.zeros(shape=(input_len,1,1))
		# model load
        xgb=XGBClassifier()
        xgb.load_model(os.path.join('optimized_models',filename))
        windowed_signal=xgb.predict_proba(LFP)[:,1]
        for i,window in enumerate(windowed_signal):
            y_predict[i*timesteps:(i+1)*timesteps]=window
    elif arch=='SVM':
        LFP=LFP[:len(LFP)-len(LFP)%timesteps,:].reshape(-1,timesteps*n_channels)
        y_predict= np.zeros(shape=(input_len,1,1))
		# model load
        clf=fcn_load_pickle(os.path.join('optimized_models',filename)).calibrated_classifiers_[0]

        windowed_signal= clf.predict_proba(LFP)[:,1]
        for i,window in enumerate(windowed_signal):
            y_predict[i*timesteps:(i+1)*timesteps]=window
        # y_predict: after expanding the windows, to be compatible with perf array
    elif arch=='LSTM':
        LFP=LFP[:len(LFP)-len(LFP)%timesteps,:].reshape(-1,timesteps,n_channels)
		# Model load
        model = keras.models.load_model(os.path.join('optimized_models',filename))
        y_predict = model.predict(LFP,verbose=1)
        y_predict=y_predict.reshape(-1,1,1)
        y_predict=np.append(y_predict,np.zeros(shape=(input_len%timesteps,1,1))) if (input_len%timesteps!=0) else y_predict
    elif arch=='CNN1D':
        LFP=LFP[:len(LFP)-len(LFP)%timesteps,:].reshape(-1,timesteps,n_channels)
        optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        model = keras.models.load_model(os.path.join('optimized_models',filename), compile=False)
        model.compile(loss="binary_crossentropy", optimizer=optimizer)

        windowed_signal = model.predict(LFP, verbose=True)
        windowed_signal=windowed_signal.reshape(-1)
        y_predict=np.zeros(shape=(input_len,1,1))
        for i,window in enumerate(windowed_signal):
            y_predict[i*timesteps:(i+1)*timesteps]=window
    elif arch=='CNN2D':
        model = keras.models.load_model(os.path.join('optimized_models',filename))
        LFP=LFP[:len(LFP)-len(LFP)%timesteps,:].reshape(-1,timesteps,n_channels,1)
        y_predict= np.zeros(shape=(input_len,1,1))
        windowed_signal= model.predict(LFP,verbose=1)
        for i,window in enumerate(windowed_signal):
            y_predict[i*timesteps:(i+1)*timesteps]=window
        
    return(y_predict.reshape(-1))


# Selecting the index functions
def get_predictions_index(predictions,threshold=0.5):
	'''
		[pred_indexes] = get_predictions_index(predictions, thershold)\n
		Returns the begining and ending samples of the events above a given threshold\n
		Inputs:
			predictions:	X, array with the continuous output of a model (even the Gt)
			threshold:		float, signal intervals above this value will be considered events
		
		Output:		
			pred_indexes:	Nx2, array containing the begining and ending index sample of the events
		'''
	aux=np.copy(predictions)
	aux[aux>=threshold]=1
	aux[aux<threshold]=0
	dif=np.diff(aux,axis=0)
	begin_indexes=np.where(dif==1)[0]
	end_indexes=np.where(dif==-1)[0]

	if len(begin_indexes)>len(end_indexes):
		begin_indexes=begin_indexes[:-1]
	elif len(begin_indexes)<len(end_indexes):
		end_indexes=end_indexes[1:]

	pred_indexes=np.empty(shape=(len(begin_indexes),2))
	pred_indexes[:,0]=begin_indexes
	pred_indexes[:,1]=end_indexes

	return pred_indexes

def middle_stamps(pred_ind):
    ''' [mids]=middle_stamps(pred_ind) returns the middle stamps of the events passed in pred_ind
    '''
    mids=[]
    for pred in pred_ind:
        mids.append(int((pred[1]+pred[0])//2))
    return(np.array(mids))

def get_click_th(event):
    if event.xdata<0:
        x=0
    elif event.xdata>1:
        x=1
    else:
        x=event.xdata
    return(x)

def format_predictions(path,preds,sf):
    ''' 
    format_predictions(path,preds,sf) writes a .txt with the initial and end times of events in seconds 
    inputs:
        path       str, absolute path of the file that will be created
        preds      (2,n_events) np.array with the initial and end timestamps of the events
        sf         int, sampling frequency of the data
	A Rubio, LCN 2023
    '''
    f=open(path,'w')

    preds=preds/sf
    for pred in preds:
		
        f.write(str(pred[0])+' ')
        f.write(str(pred[1]))
        f.write('\n')
    f.close()
    return  


# Performance (precission, recall, F1) metrics
def get_performance(pred_events, true_events, threshold=0, exclude_matched_trues=False, verbose=True):

	'''
	[precision, recall, F1, TP, FN, IOU] = get_performance(pred_events, true_events)
	Computes all these measures given a cell array with boundaries of pred_events

	Inputs:
		pred_events		 Nx2 matrix with start and end of pred events (seconds)
		true_events		 Mx2 matrix with start and end of true events (seconds)
		threshold		   Threshold to IoU. By default is 0
		exclude_matched_trues False by defaut (one true can match many predictions)

	Output:
		precision		   Metric indicating the percentage of correct 
							predictions out of total predictions
		recall			  Metric indicating the percentage of true events 
							predicted correctly
		F1				  Metric with a measure that combines precision and recall.
		TP				  Nx1 matrix indicating which pred event detected a true event, and
							a true positive (True) or was false negative (False)
		FN				  Mx1 matrix indicating which true event has been detected (False)
							or not (True)
		IOU				 NxM array with intersections over unions

	A Navas-Olive, LCN 2020
	'''

	# Check similarity of pred and true events by computing if intersection over union > 0
	# Compute IOU
	[IOU, IOU_pred, IOU_true] = intersection_over_union(pred_events, true_events)
	# IOU_pred-> IOU de cada evento predecido
	# IOU_true-> IOU de la GT


 # Excluye los true coincidentes
	if exclude_matched_trues:
		# Take maximal IOUs, and make the rest be zero
		pred_with_maxIOU = np.argmax(IOU, axis=0)
		IOU_pred_one_true_match = np.zeros_like(IOU)
		for itrue, ipred in enumerate(pred_with_maxIOU):
			IOU_pred_one_true_match[ipred, itrue] = IOU[ipred, itrue]
		# True positive: Predicted event that has a IoU with any true > 0
		TP = (IOU_pred_one_true_match.sum(axis=1) > threshold)
		# False negative: Predicted event that has not a IoU with any true
		FN = (IOU_true <= threshold)

	else:
		# True positive: Predicted event that has a IoU with any true > 0
		# 
		TP = (IOU_pred>threshold) 
		# False negative: Predicted event that has not a IoU with any true
		FN = (IOU_true<=threshold)  
	
	# Precision and recall
	precision = np.mean(TP)      # Media de verdaderos positivos: 1 si todas las predicciones son aciertos
	recall = 1. - np.mean(FN)    # 1-media de falsos negativos: 1 si toda la GT está presente en los aciertos
	F1 = 2. * (precision * recall) / (precision + recall)  
	if (precision + recall) == 0:
		F1 = 0.
	else:
		F1 = 2. * (precision * recall) / (precision + recall)

	if verbose:
		print('precision =', precision)
		print('recall =', recall)
		print('F1 =', F1)
	
	# Variable outputs
	return precision, recall, F1, TP, FN, IOU

def intersection_over_union(x, y):
	'''
	IOU = intersection_over_union(x, y) computes the percentage of 
	intersection over their union between every two pair of intervals 
	x and y.
	
	Inputs:
	  x	 Nx2 array with beginnings and ends of 1D events
	  y	 Mx2 array with beginnings and ends of 1D events
	
	Output:
	  IOU   NxM array with intersections over unions
	  IOUx  (optional) Nx1 array with indexes of y events with maximal IOU 
			It's zero if IOU=0 for all y events
	  IOUy  (optional) Mx1 array with indexes of x events with maximal IOU 
			It's zero if IOU=0 for all x events
	
	A Navas-Olive, LCN 2020
	'''

	if (len(x)>0) and (len(y)>0):

		# Initialize
		Intersection = np.zeros((x.shape[0],  y.shape[0]),dtype=np.float32)
		Union = np.ones((x.shape[0],  y.shape[0]),dtype=np.float32)
		# Go through every y (beginning-end) pair
		for iy in range(y.shape[0]):
			# Intersection duration: difference between minimum end and maximum ini
			Intersection[:, iy] = np.maximum( np.minimum(x[:, 1], y[iy, 1]) - np.maximum(x[:, 0], y[iy, 0]), 0)
			# Union duration: sum of durations of both events minus its intersection
			Union[:, iy, None] = np.diff(x, axis=1) + np.diff(y[iy, :]) - Intersection[:, iy, None]

		# Compute intersection over union
		IOU = Intersection / Union

		# Compute which events from y have maximal IOU with x
		IOUx = np.max(IOU, axis=1, keepdims=True)

		# Compute which events from x have maximal IOU with y
		IOUy = np.max(IOU, axis=0, keepdims=True)

		# Optional outputs
		return IOU, IOUx, IOUy
		
	elif len(x)==0:
		
		print('x is empty. Cant perform IoU')
		return np.array([]), np.array([]), np.zeros((y.shape[0], 1))
		
	elif len(y)==0:
		
		print('y is empty. Cant perform IoU')
		return np.array([]), np.zeros((1, x.shape[0])), np.array([])


# Interpolation 
def interpolate_channels(data, ch_map):
    
    interp_data = np.zeros((data.shape[0], len(ch_map)))
    for idx, ch in enumerate(ch_map):
        if ch>-1:
            interp_data[:,idx] = data[:,ch]
        else:
            pre_ch_idx = np.where(np.array(ch_map[:idx])>-1)[0][-1]
            pre_ch = ch_map[pre_ch_idx]
            post_ch_idx = np.where(np.array(ch_map[idx+1:])>-1)[0][0]+idx+1
            post_ch = ch_map[post_ch_idx]
            ch_dist = post_ch_idx - pre_ch_idx
            interp_data[:,idx] = data[:, pre_ch] + ((idx-pre_ch_idx)/ch_dist) * \
                (data[:, post_ch] - data[:, pre_ch])
    return interp_data