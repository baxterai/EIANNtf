# -*- coding: utf-8 -*-
"""SANItf2_loadDataset.py

# Requirements:
Python 3 and Tensorflow 2.1+ 

# License:
MIT License

# Usage:
see SANItf2.py

# Description:

Provide dataset load functions

- Author: Richard Bruce Baxter - Copyright (c) 2020 Baxter AI (baxterai.com)

# Datasets:

	Terminology: 
	*   Note "batch" refers to the actual batch (not the minibatch) being trained
	*   Note "train" refers to the neural network train/validation/test data (it is automatically divided up into train/test sets)
	
## Dataset type 1 (recommended):

	XtrainBatchXXXX.bat: 
	*   x input features (cols) per experience (row) [space delimited]

	YtrainBatchXXXX.bat:
	*   x output classes (cols) per experience (row) [space delimited]
	*   One-hot encoded with a 1 corresponding to the class, and 0 otherwise
	
## Dataset type 2:

	trainBatchXXXX.dat: 
	*   y class (first col) then x input features (cols) per experience (row) [comma delimited]
	*   y class is an integer between 1 and numberOfClasses (not hot encoded)	
	
## Dataset type 1 example - POStagSequence (XtrainBatchXXXX.dat and YtrainBatchXXXX.dat)

	This example dataset is generated from wikidump. 

	All possible POS tags are identified for each word in n-word sequence (e.g. 11 words) using ~wordnet , where the centre word has a non-ambigious POS tag.
	~53 POS tags identifiable per word.

	XtrainBatchXXXX.bat: 
	*   530 features (cols) per word sequence (row).
	*       10 words per sequence (missing centre word). 
	*       53 features per word in sequence: ie 53*10 - x53 features [word #1] x53 features [word #2] etc 
	*   includes POS_INDEX_OUT_OF_SENTENCE_BOUNDS if !GIA_PREPROCESSOR_POS_TAGGER_DATABASE_DO_NOT_TRAIN_POS_INDEX_OUT_OF_SENTENCE_BOUNDS
        *   supports ambiguous contextual POS data (ie more than valid POS index per word in the sequence)
	*   a 1 is assigned to each possible POS tag index, 0 otherwise
	
	YtrainBatchXXXX.bat:
	*   52 possible classes. 
	*   One-hot encoded with a 1 corresponding to its generated POS tag index.

	To generate the POS dataset with GIA (XtrainBatchXXXX.bat/YtrainBatchXXXX.bat):
	```
	download wikipedia archive (e.g. enwiki-20171201-pages-articles.xml.bz2)
	download wikiextractor tool (https://github.com/attardi/wikiextractor)
	execute WikiExtractor.py on wikipedia archive
	edit SHAREDglobalDefs.hpp; activate:
		#define COMPILE_GIA_WITH_ANN_GENERATE_POS_TAGGER_DATABASE
	edit GIAglobalDefs.hpp;
		disable GIA_PREPROCESSOR_POS_TAGGER_DATABASE_NEURAL_NETWORK_EXTERNAL_TRAIN_SINGLE_BATCH_ONLY and set GIA_PREPROCESSOR_POS_TAGGER_GENERATE_DATABASE_DOC_XML_OUTPUT_START_FILE accordingly (or use batch script with GIA_PREPROCESSOR_POS_TAGGER_DATABASE_NEURAL_NETWORK_EXTERNAL_TRAIN_SINGLE_BATCH_ONLY)
		enable GIA_PREPROCESSOR_POS_TAGGER_DATABASE_TRAIN_AMBIGUOUS_PERMUTATIONS
	make --makefile=makefile.GIAwithANNgeneratePOStaggerDatabase.UB16
	./GIAgeneratePOStaggerDatabase.exe -dbpostaggerfolder "/home/rich/source/GIAPOStaggerDatabase" -lrp -lrpfolder "/home/rich/source/source/LRPdata" -wikiDumpFolder "/home/rich/soft/wiki/output" (-wikiDumpFileBatchIndex X)
	this creates XtrainBatchXXXX.dat and YtrainBatchXXXX.dat
	```
## Dataset type 2 example - NewThyroid (trainBatchXXXX.dat - aka NewThyroid.data): 

	This example dataset is medical data relating to the diagnosis of thyroid disease.


"""

import tensorflow as tf
import numpy as np
from numpy import genfromtxt


percentageDatasetTrain = 80.0	#ie 80% train, 20% test

datasetType1alreadyNormalised = True	#if True, assume that the dataset includes values between 0 and 1 only
datasetType2alreadyNormalised = False	#if True, assume that the dataset includes values between 0 and 1 only


#"""**Please select both data files from local harddrive: XtrainBatchSmall.dat and YtrainBatchSmall.dat:**"""
#
#from google.colab import files
#uploaded = files.upload()
#for fn in uploaded.keys():
#  print('User uploaded file "{name}" with length {length} bytes'.format(
#		name=fn, length=len(uploaded[fn])))



def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=float):
	def iter_func():
		with open(filename, 'r') as infile:
			for _ in range(skiprows):
				next(infile)
			for line in infile:
				line = line.rstrip().split(delimiter)
				for item in line:
					yield dtype(item)
		iter_loadtxt.rowlength = len(line)

	data = np.fromiter(iter_func(), dtype=dtype)
	data = data.reshape((-1, iter_loadtxt.rowlength))
	return data
	
def hotEncode(y, maxY):
	yHotEncoded = np.zeros(maxY)
	yHotEncoded[y-1] = 1
	return yHotEncoded
			
def loadDatasetType1(datasetFileNameX, datasetFileNameY):
	
	#all_X = genfromtxt(datasetFileNameX, delimiter=' ')
	#all_Y = genfromtxt(datasetFileNameY, delimiter=' ')
	all_X = iter_loadtxt(datasetFileNameX, delimiter=' ')
	all_Y = iter_loadtxt(datasetFileNameY, delimiter=' ')
	
	all_Y = np.array(all_Y, np.uint8)

	#randomise data
	#dataRawRandomised = dataRaw
	#np.random.shuffle(dataRawRandomised)
	#all_X = dataRawRandomised[:,1:]
	#all_Y = dataRawRandomised[:,0]
		
	datasetNumExamples = all_Y.shape[0]
	datasetNumFeatures = all_X.shape[1]
	datasetNumClasses = all_Y.shape[1]

	datasetNumExamplesTrain = int(float(datasetNumExamples)*percentageDatasetTrain/100.0)
	datasetNumExamplesTest = int(float(datasetNumExamples)*(100.0-percentageDatasetTrain)/100.0)
	
	#reduce 1-hot encoding to class index:
	all_Ycollapsed = np.empty([datasetNumExamples])
	for r in range(datasetNumExamples):
		for i in range(datasetNumClasses):
			if all_Y[r,i] == 1:
				all_Ycollapsed[r] = i
	all_Ynormalised = all_Ycollapsed
	
	if(datasetType1alreadyNormalised):
		all_Xnormalised = all_X
	else:
		#normalise x data between 0.0 and 1.0:
		all_Xnormalised = all_X #np.zeros(all_X.shape)
		for i in range(datasetNumFeatures):
			maxXi = np.amax(all_X[:,i])
			minXi = np.amin(all_X[:,i])
			all_Xnormalised[:,i] = (all_X[:,i] - minXi) / (maxXi - minXi) #all_X[:,i]/maxXi		

	train_x = all_Xnormalised[0:datasetNumExamplesTrain, :]
	test_x = all_Xnormalised[-datasetNumExamplesTest:, :]
	train_y = all_Ynormalised[0:datasetNumExamplesTrain]
	test_y = all_Ynormalised[-datasetNumExamplesTest:]
			
	# Convert x/y data to float32/uint8.
	train_x, test_x = np.array(train_x, np.float32), np.array(test_x, np.float32)
	train_y, test_y = np.array(train_y, np.uint8), np.array(test_y, np.uint8) 
	#https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data?version=stable
	#https://medium.com/@HojjatA/could-not-find-valid-device-for-node-while-eagerly-executing-8f2ff588d1e

	return datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y


def loadDatasetType2(datasetFileName):

	#dataRaw = genfromtxt(datasetFileName, delimiter=',')
	dataRaw = iter_loadtxt(datasetFileName, delimiter=',')
	
	datasetNumExamples = dataRaw.shape[0]
	#print (dataRaw)
	#print ("datasetNumExamples: " + str(datasetNumExamples))
	#randomise data
	dataRawRandomised = dataRaw
	np.random.shuffle(dataRawRandomised)
	#print (dataRawRandomised)
	all_X = dataRawRandomised[:,1:]
	all_Y = dataRawRandomised[:,0]

	datasetNumExamplesTrain = int(float(datasetNumExamples)*percentageDatasetTrain/100.0)
	datasetNumExamplesTest = int(float(datasetNumExamples)*(100.0-percentageDatasetTrain)/100.0)
		
	datasetNumFeatures = all_X.shape[1]
	maxY = int(np.amax(all_Y))
	datasetNumClasses = maxY
		 
	if(datasetType2alreadyNormalised):
		all_Ynormalised = all_Y
		all_Xnormalised = all_X
	else:
		#normalise y data between 0 and datasetNumClasses
		all_Ynormalised = all_Y - 1 #->0
		
		#normalise x data between 0.0 and 1.0:
		all_Xnormalised = all_X #np.zeros(all_X.shape)
		for i in range(datasetNumFeatures):
			maxXi = np.amax(all_X[:,i])
			minXi = np.amin(all_X[:,i])
			all_Xnormalised[:,i] = (all_X[:,i] - minXi) / (maxXi - minXi) #all_X[:,i]/maxXi
	
	train_x = all_Xnormalised[0:datasetNumExamplesTrain, :]
	test_x = all_Xnormalised[-datasetNumExamplesTest:, :]
	train_y = all_Ynormalised[0:datasetNumExamplesTrain]
	test_y = all_Ynormalised[-datasetNumExamplesTest:]
	
	#hot encode y data (not required):
	#all_Yhotencoded = np.zeros((datasetNumExamples, maxY))
	#for i in range(datasetNumExamples):	#for every row (ie example) in all_Y
	#	y = int(all_Y[i])
	#	yHotEncoded = hotEncode(y, maxY)
	#	all_Yhotencoded[i] = yHotEncoded
	#train_y = all_Yhotencoded[0:datasetNumExamplesTrain, :]
	#test_y = all_Yhotencoded[-datasetNumExamplesTest:, :]

	#print(datasetNumExamplesTrain)
	#print(datasetNumExamplesTest)
	#print(train_x.shape)
	#print(train_y.shape)
	#print(test_x.shape)
	#print(test_y.shape)

	# Convert x/y data to float32/uint8.
	train_x, test_x = np.array(train_x, np.float32), np.array(test_x, np.float32)
	train_y, test_y = np.array(train_y, np.uint8), np.array(test_y, np.uint8) 
	#https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data?version=stable
	#https://medium.com/@HojjatA/could-not-find-valid-device-for-node-while-eagerly-executing-8f2ff588d1e

	return datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y
	

