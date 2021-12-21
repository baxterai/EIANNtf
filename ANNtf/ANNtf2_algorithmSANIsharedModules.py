"""ANNtf2_algorithmSANIsharedModules.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2021 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNtf2.py

# Usage:
see ANNtf2.py

# Description:

ANNtf algorithm SANI shared modules - define Sequentially Activated Neuronal Input neural network with shared modules

Neural modules can be shared between different areas of input sequence, eg sentence (cf RNN).
This code mirrors that of GIA Sequence Grammar ANN.
Can parse (by default expects to parse) full sentences; ie features for each word in sentence.

"""

#start common ANNtf2_algorithmSANI.py code:

import tensorflow as tf
import numpy as np
from ANNtf2_operations import * #generateParameterNameSeq, generateParameterName
import ANNtf2_algorithmSANIoperations
from ANNtf2_algorithmSANIglobalDefs import *
import ANNtf2_globalDefs


#parameters
#static parameters (convert from tf.variable to tf.constant?):
Cseq = {}
CseqLayer = {}	
n_h_cumulative = {}
#variable parameters:
WRseq = {}	#weights matrix
WR = {}	#weights matrix
BR = {}	#biases vector
Wseq = {}	#weights matrix
Bseq = {}	#biases vector
W = {}	#weights matrix
B = {}	#biases vector

#parameters
#static parameters (convert from tf.variable to tf.constant?):
#if(not supportFullConnectivity):
#	if(useSparseTensors):
#		Cseq = {}	#connectivity vector
#		if(supportSkipLayers):	
#			CseqLayer = {}	
#			n_h_cumulative = {}
##variable parameters:	
#if((algorithmSANI == "sharedModulesHebbian") or (algorithmSANI == "sharedModulesBinary") or (algorithmSANI == "sharedModules")):
#	if(recordNetworkWeights):
#		if(recordSubInputsWeighted):
#			AseqInputVerified = {}
#			WRseq = {}	#weights matrix
#		if(recordSequentialInputsWeighted):
#			WR = {}	#weights matrix
#		if(recordNeuronsWeighted):
#			BR = {}	#biases vector
#if((algorithmSANI == "sharedModulesHebbian") or (algorithmSANI == "sharedModules") or (algorithmSANI == "repeatedModules")):
#	#variable parameters (tf.variable): 
#	if(allowMultipleSubinputsPerSequentialInput):
#		if(performSummationOfSubInputsWeighted):
#			Wseq = {}	#weights matrix
#			Bseq = {}	#biases vector
#	if(performSummationOfSequentialInputsWeighted):
#		W = {}	#weights matrix
#		B = {}	#biases vector
	
			
#Network parameters
n_h = []
numberOfLayers = 0

#if((algorithmSANI == "sharedModulesHebbian") or (algorithmSANI == "sharedModulesBinary") or (algorithmSANI == "sharedModules")):	#only code to currently use these variables
numberOfFeaturesPerWord = -1
paddingTagIndex = -1
def defineTrainingParametersSANIsharedModules(numberOfFeaturesPerWordNew, paddingTagIndexNew):
	#if((algorithmSANI == "sharedModulesHebbian") or (algorithmSANI == "sharedModulesBinary") or (algorithmSANI == "sharedModules")):	#only code to currently use these variables
	global numberOfFeaturesPerWord
	global paddingTagIndex
	numberOfFeaturesPerWord = numberOfFeaturesPerWordNew
	paddingTagIndex = paddingTagIndexNew

def defineNetworkParametersSANIwrapper(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, useSmallSentenceLengths, numberOfFeaturesPerWord):
	global n_h
	global numberOfLayers
	n_h, numberOfLayers = ANNtf2_algorithmSANIoperations.defineNetworkParametersSANI(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, useSmallSentenceLengths, numberOfFeaturesPerWord)
	
def defineTrainingParametersSANIwrapper(dataset, trainMultipleFiles):
	return ANNtf2_algorithmSANIoperations.defineTrainingParametersSANI(dataset, trainMultipleFiles)
	

def defineNeuralNetworkParameters():
	global n_h_cumulative
	ANNtf2_algorithmSANIoperations.defineNeuralNetworkParametersSANI(n_h, numberOfLayers, Cseq, CseqLayer, n_h_cumulative, WRseq, WR, BR, Wseq, Bseq, W, B)
			

#temporary variables for neuralNetworkPropagationSANI:
if(algorithmSANI == "sharedModules"):
	Vseq = {}
	Zseq = {}
	Aseq = {}
	TMaxSeq = {}
	TMinSeq = {}
	ZseqTadjusted = {}
	Z = {}
	A = {}
	TMax = {}
	TMin = {}
	sequentialActivationFound = {}
	AseqInputVerified = {}

#end common ANNtf2_algorithmSANI.py code



def neuralNetworkPropagation(x, networkIndex=None):
	return neuralNetworkPropagationSANI(x)
	
def neuralNetworkPropagationSANI(x):
		
	#note connectivity indexes are used rather than sparse weight matrices due to limitations in current tf2 sparse tensor implementation
	
	#definitions for reference:
	
	#neuron sequential input vars;
	#x/AprevLayer	#output vector (dim: batchSize*n_h[l])
	#if(allowMultipleSubinputsPerSequentialInput):
		#if(useSparseTensors):
			#Cseq	#static connectivity matrix (int) - indexes of neurons on prior layer stored; mapped to W  (dim: maxNumberSubinputsPerSequentialInput*n_h[l])
			#if(supportSkipLayers):
				#CseqLayer	#static connectivity matrix (int) - indexes of pior layer stored; mapped to W (dim: maxNumberSubinputsPerSequentialInput*n_h[l])
			#Wseq #weights of connections; see Cseq (dim: maxNumberSubinputsPerSequentialInput*n_h[l])
			#AseqSum	#combination variable
		#else:
			#if(supportSkipLayers):
				#Wseq #weights of connections (dim: n_h_cumulativeNP[l]*n_h[l])
			#else:
				#Wseq #weights of connections (dim: n_h[l-1]*n_h[l])
	#else:
		#Cseq	#static connectivity vector (int) - indexes of neurons on prior layer stored; mapped to W - defines which prior layer neuron a sequential input is connected to (dim: n_h[l])
		#if(supportSkipLayers):
			#CseqLayer	#static connectivity matrix (int) - indexes of pior layer stored; mapped to W - defines which prior layer a sequential input is connected to  (dim: n_h[l])
		#Wseq #weights of connections; see Cseq (dim: n_h[l])	
	
	#Vseq	#mutable verification vector (dim: batchSize*n_h[l] - regenerated for each sequential input index)			#records whether particular neuron sequential inputs are currently active
	#TMaxSeq	#mutable time vector (dim: batchSize*n_h[l] - regenerated for each sequential input index)				#records the time at which a particular sequential input last fired
	#TMinSeq	#mutable time vector (dim: batchSize*n_h[l] - regenerated for each sequential input index)				#records the time at which a first encapsulated subinput fired
	#Zseq	#neuron activation function output vector (dim: batchSize*n_h[l]  - regenerated for each sequential input index)	#records the current output value of sequential inputs
	#Aseq	#neuron activation function output vector (dim: batchSize*n_h[l]  - regenerated for each sequential input index)	#records the current output value of sequential inputs
	#if(allowMultipleContributingSubinputsPerSequentialInput):
		#if((resetSequentialInputsIfOnlyFirstInputValid) and (s == 0)):
			#ZseqTadjusted	#neuron activation function output vector T adjusted (dim: batchSize*n_h[l]  - regenerated for each sequential input index)	#records the current output value of sequential inputs*time
	#if(recordSubInputsWeighted):
		#AseqInputVerified	#neuron input (sequentially verified) (dim: batchSize*numberSubinputsPerSequentialInput*n_h[l]  - records the subinputs that were used to activate the sequential input)
		#WRseq #weights of connections; see Cseq (dim: maxNumberSubinputsPerSequentialInput*n_h[l])

	#Q  
	#Z	#neuron activation function input (dim: batchSize*n_h[l])
	#A	#neuron activation function output (dim: batchSize*n_h[l])
	#TMax	#mutable time vector (dim: batchSize*n_h[l]) - same as TMaxSeq[numberOfSequentialInputs-1]
	#TMin	#mutable time vector (dim: batchSize*n_h[l]) - same as TMinSeq[numberOfSequentialInputs-1]
	#if(performSummationOfSequentialInputsWeighted):	
		#W	(dim: numberOfSequentialInputs*n_h[l])	
	
	#combination vars (per layer)???;
	#if(performSummationOfSequentialInputs):
		#these are all used for different methods of sequential input summation:
		#if(sequentialInputCombinationModeSummation == 1):
			#ZseqSum	#(dim: batchSize*n_h[l])
		#if(sequentialInputCombinationModeSummation == 2):
			#AseqSum	#(dim: batchSize*n_h[l])
		#if(sequentialInputCombinationModeSummation == 3):
			#ZseqWeightedSum	#(dim: batchSize*n_h[l])
		#if(sequentialInputCombinationModeSummation == 4):
			#AseqWeightedSum	#(dim: batchSize*n_h[l])
	
	batchSize = x.shape[0]
	
	#optimise feed length based on max sentence length in batch:
	#unoptimised: numberOfFeatures = x.shape[1]
	xIsNotPadding = tf.math.less(x, paddingTagIndex) #tf.math.less(tf.dtypes.cast(x, tf.int32), paddingTagIndex)
	coordinatesOfNotPadding = tf.where(xIsNotPadding)
	numberOfFeaturesCropped = tf.reduce_max(coordinatesOfNotPadding[:, 1]).numpy()+1
	
	inputLength = numberOfFeaturesPerWord*numberOfWordsInConvolutionalWindowSeen
	
	maxNumberOfWordsInSentenceBatch = int(numberOfFeaturesCropped/numberOfFeaturesPerWord)
	
	for l in range(1, numberOfLayers+1):
		Z[generateParameterName(l, "Z")] = tf.Variable(tf.zeros([batchSize, n_h[l]]), dtype=tf.float32)
		A[generateParameterName(l, "A")] = tf.Variable(tf.zeros([batchSize, n_h[l]]), dtype=tf.float32)
		TMax[generateParameterName(l, "TMax")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.int32))
		TMin[generateParameterName(l, "TMin")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.int32))
		for s in range(numberOfSequentialInputs):
			Vseq[generateParameterNameSeq(l, s, "Vseq")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), tf.bool), dtype=tf.bool)
			TMaxSeq[generateParameterNameSeq(l, s, "TMaxSeq")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.int32))
			TMinSeq[generateParameterNameSeq(l, s, "TMinSeq")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.int32))
			Zseq[generateParameterNameSeq(l, s, "Zseq")] = tf.Variable(tf.zeros([batchSize, n_h[l]]), dtype=tf.float32)
			Aseq[generateParameterNameSeq(l, s, "Aseq")] = tf.Variable(tf.zeros([batchSize, n_h[l]]), dtype=tf.float32)
			if(allowMultipleContributingSubinputsPerSequentialInput):
				if((resetSequentialInputsIfOnlyFirstInputValid) and (s == 0)):
					ZseqTadjusted[generateParameterNameSeq(l, s, "ZseqTadjusted")] = tf.Variable(tf.zeros([batchSize, n_h[l]]), dtype=tf.float32)
			if(recordNetworkWeights):
				if(recordSubInputsWeighted):
					numberSubinputsPerSequentialInput = ANNtf2_algorithmSANIoperations.calculateNumberSubinputsPerSequentialInput(s)
					AseqInputVerified[generateParameterNameSeq(l, s, "AseqInputVerified")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, numberSubinputsPerSequentialInput, n_h[l]]), dtype=tf.bool), dtype=tf.bool)
				
	#for w in range(maxNumberOfWordsInSentenceBatch):
	#for w in range(0, 1):
	for w in range(maxNumberOfWordsInSentenceBatch-numberOfWordsInConvolutionalWindowSeen+1):
	
		#print("w = " + str(w))
		
		for l in range(1, numberOfLayers+1):
			sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), tf.bool), dtype=tf.bool)

		if(w == 0):
			AfirstLayerShifted = x[:, 0:inputLength]
		else:
			paddings = tf.constant([[0, 0], [w*numberOfFeaturesPerWord, 0]])	#shift input to the right by x words (such that a different input window will be presented to the network)
			#AfirstLayerShifted = x[:, w*numberOfFeaturesPerWord:min(w*numberOfFeaturesPerWord+inputLength, numberOfFeaturesCropped)]
			AfirstLayerShifted = x[:, w*numberOfFeaturesPerWord:w*numberOfFeaturesPerWord+inputLength]
			tf.pad(AfirstLayerShifted, paddings, "CONSTANT")

		
		AfirstLayerShifted = tf.dtypes.cast(AfirstLayerShifted, tf.float32)	#added 01 Sept 2021 #convert input from int to float
		#print("AfirstLayerShifted = ", AfirstLayerShifted)
			
		#printShape(AfirstLayerShifted, "AfirstLayerShifted")
			
		AprevLayer = AfirstLayerShifted
		
		TMaxPrevLayer = tf.ones(([batchSize, n_h[0]]), tf.int32)
		TMaxPrevLayer = TMaxPrevLayer*w
		TMinPrevLayer = tf.ones(([batchSize, n_h[0]]), tf.int32)
		TMinPrevLayer = TMinPrevLayer*w
		
		for l in range(1, numberOfLayers+1):	#start algorithm at n_h[1]; ie first hidden layer

			#print("\tl = " + str(l))

			#declare variables used across all sequential input of neuron
			#primary vars;
			if(l == 1):
				if(supportSkipLayers):
					AprevLayerAll = AprevLayer	#x
					TMaxPrevLayerAll = TMaxPrevLayer
					TMinPrevLayerAll = TMinPrevLayer
			else:
				if(supportSkipLayers):
					AprevLayerAll = tf.concat([AprevLayerAll, AprevLayer], 1)
					TMaxPrevLayerAll = tf.concat([TMaxPrevLayerAll, TMaxPrevLayer], 1)
					TMinPrevLayerAll = tf.concat([TMinPrevLayerAll, TMinPrevLayer], 1)

			#combination vars;
			if(performSummationOfSequentialInputs):
				#these are all used for different methods of sequential input summation:
				if(sequentialInputCombinationModeSummation == 1):
					ZseqSum = tf.zeros([batchSize, n_h[l]], tf.float32)
				if(sequentialInputCombinationModeSummation == 2):
					AseqSum = tf.zeros([batchSize, n_h[l]], tf.float32)
				if(sequentialInputCombinationModeSummation == 3):
					ZseqWeightedSum = tf.zeros([batchSize, n_h[l]], tf.float32)
				if(sequentialInputCombinationModeSummation == 4):
					AseqWeightedSum = tf.zeros([batchSize, n_h[l]], tf.float32)

			
			for sForward in range(numberOfSequentialInputs):
				sReverse = numberOfSequentialInputs-sForward-1	
				s = sReverse
				
				#for each sequential input of each neuron (stating from last), see if its requirements are satisfied
					#if first sequential input, and hypothetical valid activation of first input, allow reset of neuron sequential input 
						#ie if((s == 0) && (resetSequentialInputsIfOnlyFirstInputValid)):
							#skip sequential validation requirements as neuron sequential input can be reset
				
				#print("\t\ts = " + str(s))

				#identify (hypothetical) activation of neuron sequential input
				if(supportFullConnectivity):
					print("neuralNetworkPropagationSANI error: supportFullConnectivity incomplete")
				else:
					if(useSparseTensors):
						if(supportSkipLayers):

							CseqCrossLayerBase = tf.gather(n_h_cumulative['n_h_cumulative'], CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")])
							CseqCrossLayer = tf.add(Cseq[generateParameterNameSeq(l, s, "Cseq")], CseqCrossLayerBase)
							#printShape(AprevLayerAll, "AprevLayerAll")
							#printShape(CseqCrossLayer, "CseqCrossLayer")						
							AseqInput = tf.gather(AprevLayerAll, CseqCrossLayer, axis=1)
							if(allowMultipleContributingSubinputsPerSequentialInput):
								if((resetSequentialInputsIfOnlyFirstInputValid) and (s == 0)):
									AseqInputTadjusted = tf.gather(tf.multiply(AprevLayerAll, tf.dtypes.cast(TprevLayerAll, tf.float32)), CseqCrossLayer, axis=1) 
							if(not allowMultipleContributingSubinputsPerSequentialInput):
								#printShape(TprevLayerAll, "TprevLayerAll")
								#printShape(CseqCrossLayer, "CseqCrossLayer")
								TMaxSeqInput = tf.gather(TMaxPrevLayerAll, CseqCrossLayer, axis=1)
								TMinSeqInput = tf.gather(TMinPrevLayerAll, CseqCrossLayer, axis=1)
						else:
							AseqInput = tf.gather(AprevLayer, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
							if(allowMultipleContributingSubinputsPerSequentialInput):
								if((resetSequentialInputsIfOnlyFirstInputValid) and (s == 0)):
									AseqInputTadjusted = tf.gather(tf.multiply(AprevLayer, tf.dtypes.cast(TprevLayer, tf.float32)), Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
							if(not allowMultipleContributingSubinputsPerSequentialInput):
								TMaxSeqInput = tf.gather(TMaxPrevLayer, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
								TMinSeqInput = tf.gather(TMinPrevLayer, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
					else:
						if(supportSkipLayers):
							AseqInput = AprevLayerAll
							if(allowMultipleContributingSubinputsPerSequentialInput):
								if((resetSequentialInputsIfOnlyFirstInputValid) and (s == 0)):
									AseqInputTadjusted = tf.multiply(AprevLayerAll, tf.dtypes.cast(TprevLayerAll, tf.float32))
							if(not allowMultipleContributingSubinputsPerSequentialInput):
								TMaxSeqInput = TMaxPrevLayerAll
								TMinSeqInput = TMinPrevLayerAll
						else:
							AseqInput = AprevLayer
							if(allowMultipleContributingSubinputsPerSequentialInput):
								if((resetSequentialInputsIfOnlyFirstInputValid) and (s == 0)):
									AseqInputTadjusted = tf.multiply(AprevLayer, tf.dtypes.cast(TprevLayer, tf.float32))
							if(not allowMultipleContributingSubinputsPerSequentialInput):
								TMaxSeqInput = TMaxPrevLayer
								TMinSeqInput = TMinPrevLayer
									
				#calculate validation matrix based upon sequentiality requirements
				#if Vseq[s-1] is True and Vseq[s] is False;
					#OLD: and TMaxSeq[s-1] < w [this condition is guaranteed by processing s in reverse]
				if(s > 0):
					VseqTemp = Vseq[generateParameterNameSeq(l, s-1, "Vseq")]
					#VseqTemp = tf.math.logical_and(VseqTemp, tf.math.logical_less(TMaxSeq[generateParameterNameSeq(l, s-1, "TMaxSeq")], w))
					VseqTemp = tf.math.logical_and(VseqTemp, tf.math.logical_not(Vseq[generateParameterNameSeq(l, s, "Vseq")]))
				else:
					if(resetSequentialInputsIfOnlyFirstInputValid):
						#VseqTemp = tf.math.logical_or(tf.math.logical_not(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")]), tf.math.logical_not(Vseq[generateParameterNameSeq(l, s, "Vseq")]))
						VseqTemp = tf.math.logical_not(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")])
					else:
						VseqTemp = tf.math.logical_and(VseqTemp, tf.math.logical_not(Vseq[generateParameterNameSeq(l, s, "Vseq")]))

				#VseqInt = tf.dtypes.cast(VseqTemp, tf.int32)
				VseqFloat = tf.dtypes.cast(VseqTemp, tf.float32)
				
				#AseqInput = tf.multiply(VseqFloat, AseqInput)

				#calculate output for layer sequential input s
				if(allowMultipleSubinputsPerSequentialInput):
					if(not useSparseTensors):
						if(performSummationOfSubInputsWeighted):
							ZseqHypothetical = tf.add(tf.matmul(AseqInput, Wseq[generateParameterNameSeq(l, s, "Wseq")]), Bseq[generateParameterNameSeq(l, s, "Bseq")])
							#ZseqHypothetical = tf.matmul(AseqInput, Wseq[generateParameterNameSeq(l, s, "Wseq")])
						else:
							AseqInputAverage = tf.math.reduce_mean(AseqInput, axis=1)	#take average
							multiples = tf.constant([1,n_h[l]], tf.int32)
							ZseqHypothetical = tf.tile(tf.reshape(AseqInputAverage, [batchSize, 1]), multiples)
					else:
						if(not allowMultipleContributingSubinputsPerSequentialInput):
							#ensure that T continguous constraint is met;
							#NO: take the max subinput pathway only (ie matrix mult but with max() rather than sum() for each dot product)
							if(s > 0):

								numberSubinputsPerSequentialInput = ANNtf2_algorithmSANIoperations.calculateNumberSubinputsPerSequentialInput(s)
								multiples = tf.constant([1,numberSubinputsPerSequentialInput,1], tf.int32)

								if(enforceTcontiguityBetweenSequentialInputs):
									#1. first select the AseqInput inputs that have TMaxSeq[l, s] contiguous with TMaxSeq[l, s-1]:
									TMaxSeqPrevPlus1 = TMaxSeq[generateParameterNameSeq(l, s-1, "TMaxSeq")]+1
									
									TMaxSeqPrevPlus1Tiled = tf.tile(tf.reshape(TMaxSeqPrevPlus1, [batchSize, 1, n_h[l]]), multiples)
									TMinSeqInputThreshold = tf.math.equal(TMinSeqInput, TMaxSeqPrevPlus1Tiled)
									AseqInputTthresholded = tf.multiply(AseqInput, tf.dtypes.cast(TMinSeqInputThreshold, tf.float32))
									
									TMinSeqInputThresholded = TMinSeqPrev	#note this is recording the min of the sequence, not the sequential input	#OLD: TMaxSeqPrevPlus1
									
									AseqInput = AseqInputTthresholded
									
								if(enforceTcontiguityTakeEncapsulatedSubinputWithMaxTvalueEqualsW):
									TMaxSeqInputThreshold = tf.math.equal(TMaxSeqInput, w)	#CHECKTHIS
									AseqInputTthresholded = tf.multiply(AseqInput, tf.dtypes.cast(TMaxSeqInputThreshold, tf.float32))
								
									TMaxSeqInputThresholded = tf.math.reduce_max(TMaxSeqInput, axis=1)	#CHECKTHIS - ensure equal to w

									AseqInput = AseqInputTthresholded

									#VseqFloatTiled = tf.tile(tf.reshape(VseqFloat, [batchSize, 1, n_h[l]]), multiples)
									#AseqInputThresholded = tf.multiply(VseqFloatTiled, AseqInput)	#done later
							else:
								if(resetSequentialInputsIfOnlyFirstInputValid):
									#NO: ZseqHypotheticalResetThreshold = 1 	#always reset if valid first input (CHECKTHIS)	#dummy variable
									#only reset first sequential input if TMaxSeqInput > TMax[l]
									
									numberSubinputsPerSequentialInput = ANNtf2_algorithmSANIoperations.calculateNumberSubinputsPerSequentialInput(s)
									multiples = tf.constant([1,numberSubinputsPerSequentialInput,1], tf.int32)
									
									TMaxtiled = tf.tile(tf.reshape(TMax[generateParameterName(l, "TMax")], [batchSize, 1, n_h[l]]), multiples)
									VseqTiled = tf.tile(tf.reshape(Vseq[generateParameterNameSeq(l, s, "Vseq")], [batchSize, 1, n_h[l]]), multiples)

									ZseqHypotheticalResetThreshold = tf.math.logical_or(tf.math.logical_not(VseqTiled), tf.math.greater(TMaxSeqInput, tf.dtypes.cast(TMaxtiled, tf.int32)))
									AseqInputTthresholded = tf.multiply(AseqInput, tf.dtypes.cast(ZseqHypotheticalResetThreshold, tf.float32))

									AseqInput = AseqInputTthresholded
									
									if(enforceTcontiguityTakeEncapsulatedSubinputWithMinTvalue):
										TMinSeqInputThresholdIndices = tf.dtypes.cast(tf.math.argmin(TMinSeqInput, axis=1), tf.int32)
										
										#AseqInput shape: batchSize*numSubinputs*numNeuronsOnLayer
										AseqInputReordedAxes = tf.transpose(AseqInput, [0, 2, 1])
										AseqInputReordedAxesFlattened = tf.reshape(AseqInputReordedAxes, [AseqInputReordedAxes.shape[0]*AseqInputReordedAxes.shape[1], AseqInputReordedAxes.shape[2]])
										idx_0 = tf.reshape(tf.range(AseqInputReordedAxesFlattened.shape[0]), TMinSeqInputThresholdIndices.shape)
										indices=tf.stack([idx_0,TMinSeqInputThresholdIndices],axis=-1)
										AseqInputTthresholded = tf.gather_nd(AseqInputReordedAxesFlattened, indices)
										AseqInputTthresholded = tf.expand_dims(AseqInputTthresholded, axis=1)	#note this dimension will be reduced/eliminated below so its size does not matter

										TMinSeqInputThresholded = tf.math.reduce_min(TMinSeqInput, axis=1)
										
										AseqInput = AseqInputTthresholded
	
									if(enforceTcontiguityTakeEncapsulatedSubinputWithMaxTvalueEqualsW):
										TMaxSeqInputThreshold = tf.math.equal(TMaxSeqInput, w)	#CHECKTHIS
										AseqInputTthresholded = tf.multiply(AseqInput, tf.dtypes.cast(TMaxSeqInputThreshold, tf.float32))
									
										TMaxSeqInputThresholded = tf.math.reduce_max(TMaxSeqInput, axis=1)	#CHECKTHIS - ensure equal to w

										AseqInput = AseqInputTthresholded


						#2. take the AseqInput with the highest weighting
						if(performSummationOfSubInputsWeighted):
							multiplesSeq = tf.constant([batchSize,1,1], tf.int32)
							numberSubinputsPerSequentialInput = ANNtf2_algorithmSANIoperations.calculateNumberSubinputsPerSequentialInput(s)
							WseqTiled = tf.tile(tf.reshape(Wseq[generateParameterNameSeq(l, s, "Wseq")], [1, numberSubinputsPerSequentialInput, n_h[l]]), multiplesSeq)
							AseqInputWeighted = tf.multiply(AseqInput, WseqTiled)
						else:
							AseqInputWeighted = AseqInput

						if(performSummationOfSubInputs):
							ZseqHypothetical = tf.math.reduce_sum(AseqInputWeighted, axis=1)
						else:
							#take sub input with max input signal*weight
							ZseqHypothetical = tf.math.reduce_max(AseqInputWeighted, axis=1)
							ZseqHypotheticalIndex = tf.math.argmax(AseqInputWeighted, axis=1)
				else:
					#ensure that T continguous constraint is met;
					if(s > 0):
						if(enforceTcontiguityBetweenSequentialInputs):
							#1. first select the AseqInput inputs that have TMaxSeq[l, s] contiguous with TMaxSeq[l, s-1]:

							TMaxSeqPrevPlus1 = TMaxSeq[generateParameterNameSeq(l, s-1, "TMaxSeq")]+1
							TseqInputThreshold = tf.math.equal(TMinSeqInput, TMaxSeqPrevPlus1)
							
							#print("AseqInput = ", AseqInput)
							AseqInputTthresholded = tf.multiply(AseqInput, tf.dtypes.cast(TseqInputThreshold, tf.float32))
							
							TMinSeqInputThresholded = TMaxSeqPrevPlus1
							
							AseqInput = AseqInputTthresholded
							
						if(enforceTcontiguityTakeEncapsulatedSubinputWithMaxTvalueEqualsW):
							TMaxSeqInputThreshold = tf.math.equal(TMaxSeqInput, w)	#CHECKTHIS
							AseqInputTthresholded = tf.multiply(AseqInput, tf.dtypes.cast(TMaxSeqInputThreshold, tf.float32))

							TMaxSeqInputThresholded = TMaxSeqInput	#CHECKTHIS - ensure equal to w

							AseqInput = AseqInputTthresholded
					else:
						if(resetSequentialInputsIfOnlyFirstInputValid):
							#only reset first sequential input if TMaxSeqInput > T[l]

							ZseqHypotheticalResetThreshold = tf.math.logical_or(tf.math.logical_not(Vseq[generateParameterNameSeq(l, s, "Vseq")]), tf.math.greater(TMaxSeqInput, tf.dtypes.cast(TMax[generateParameterName(l, "TMax")], tf.int32)))

							AseqInputTthresholded = tf.multiply(AseqInput, tf.dtypes.cast(ZseqHypotheticalResetThreshold, tf.float32))
							AseqInput = AseqInputTthresholded
							
							if(enforceTcontiguityTakeEncapsulatedSubinputWithMinTvalue):
								TMinSeqInputThresholded = TMinSeqInput

							if(enforceTcontiguityTakeEncapsulatedSubinputWithMaxTvalueEqualsW):
								TMaxSeqInputThreshold = tf.math.equal(TMaxSeqInput, w)	#CHECKTHIS
								AseqInputTthresholded = tf.multiply(AseqInput, tf.dtypes.cast(TMaxSeqInputThreshold, tf.float32))

								TMaxSeqInputThresholded = TMaxSeqInput	#CHECKTHIS - ensure equal to w

								AseqInput = AseqInputTthresholded

					ZseqHypothetical = AseqInput	#CHECKTHIS
					
					

				#apply sequential validation matrix
				ZseqHypothetical = tf.multiply(VseqFloat, ZseqHypothetical)
				
				#check output threshold
				ZseqPassThresold = tf.math.greater(ZseqHypothetical, sequentialInputActivationThreshold)
				
				#update ZseqPassThresold based on reset
				if(allowMultipleSubinputsPerSequentialInput):
					if(allowMultipleContributingSubinputsPerSequentialInput):
						if((resetSequentialInputsIfOnlyFirstInputValid) and (s == 0)):
							#ensure that T continguous constraint is met;
							
							#calculate output for layer sequential input s
							#old slow: only factor in inputs that have changed since the component was originally activated; if Torigin > TMaxSeq[s]
							#new: only factor in inputs if ZseqHypotheticalTadjusted > ZseqTadjusted+1
							ZseqHypotheticalTadjusted = tf.add(tf.matmul(AseqInputTadjusted, Wseq[generateParameterNameSeq(l, s, "Wseq")]), Bseq[generateParameterNameSeq(l, s, "Bseq")])
							ZseqHypotheticalTadjusted = tf.divide(ZseqHypotheticalTadjusted, AseqInputTadjusted.shape[1])	#normalise T adjustment
							ZseqHypotheticalTadjustedResetThreshold = tf.math.logical_or(tf.math.logical_not(Vseq[generateParameterNameSeq(l, s, "Vseq")]), tf.math.greater(ZseqHypotheticalTadjusted, ZseqTadjusted[generateParameterNameSeq(l, s, "ZseqTadjusted")]+averageTimeChangeOfNewInputRequiredForReset))

							ZseqPassThresold = tf.math.logical_and(ZseqPassThresold, ZseqHypotheticalTadjustedResetThreshold)	#ensures that if reset is required, Tadjusted threshold is met
				
				ZseqPassThresoldInt = tf.dtypes.cast(ZseqPassThresold, tf.int32)
				ZseqPassThresoldNot = tf.math.logical_not(ZseqPassThresold)
				ZseqPassThresoldNotInt = tf.dtypes.cast(ZseqPassThresoldNot, tf.int32)
				
				if(enforceTcontiguityConstraints):
					TMaxSeqPassThresoldUpdatedValues = tf.multiply(ZseqPassThresoldInt, TMaxSeqInputThresholded)
					TMaxSeqUpdated = tf.multiply(TMaxSeq[generateParameterNameSeq(l, s, "TMaxSeq")], ZseqPassThresoldNotInt)
					TMaxSeqUpdated = tf.add(TMaxSeqUpdated, TMaxSeqPassThresoldUpdatedValues)

					TMinSeqPassThresoldUpdatedValues = tf.multiply(ZseqPassThresoldInt, TMinSeqInputThresholded)
					TMinSeqUpdated = tf.multiply(TMinSeq[generateParameterNameSeq(l, s, "TMinSeq")], ZseqPassThresoldNotInt)
					TMinSeqUpdated = tf.add(TMinSeqUpdated, TMinSeqPassThresoldUpdatedValues)
			
				#calculate updated validation matrix
				VseqUpdated = tf.math.logical_or(Vseq[generateParameterNameSeq(l, s, "Vseq")], ZseqPassThresold)
				
				#reset appropriate neurons
				if((resetSequentialInputsIfOnlyFirstInputValid) and (s == 0)):
					resetRequiredMatrix = tf.math.logical_and(tf.math.logical_and(ZseqPassThresold, Vseq[generateParameterNameSeq(l, s, "Vseq")]), tf.math.logical_not(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")]))
					for s2 in range(numberOfSequentialInputs):
						Vseq[generateParameterNameSeq(l, s2, "Vseq")] = tf.dtypes.cast(tf.multiply(tf.dtypes.cast(Vseq[generateParameterNameSeq(l, s2, "Vseq")], tf.int32), tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32)), tf.bool)
						if(not doNotResetNeuronOutputUntilAllSequentialInputsActivated):
							TMaxSeq[generateParameterNameSeq(l, s2, "TMaxSeq")] = tf.multiply(TMaxSeq[generateParameterNameSeq(l, s2, "TMaxSeq")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32))
							TMinSeq[generateParameterNameSeq(l, s2, "TMinSeq")] = tf.multiply(TMinSeq[generateParameterNameSeq(l, s2, "TMinSeq")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32))
							Zseq[generateParameterNameSeq(l, s2, "Zseq")] = tf.multiply(Zseq[generateParameterNameSeq(l, s2, "Zseq")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.float32))
							Aseq[generateParameterNameSeq(l, s2, "Aseq")] = tf.multiply(Aseq[generateParameterNameSeq(l, s2, "Aseq")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.float32))
					sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")] = tf.math.logical_and(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")], tf.math.logical_not(resetRequiredMatrix))
					if(not doNotResetNeuronOutputUntilAllSequentialInputsActivated):
						Z[generateParameterName(l, "Z")] = tf.multiply(Z[generateParameterName(l, "Z")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.float32))
						A[generateParameterName(l, "A")] = tf.multiply(A[generateParameterName(l, "A")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.float32))
						TMax[generateParameterName(l, "TMax")] = tf.multiply(TMax[generateParameterName(l, "TMax")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32))
						TMin[generateParameterName(l, "TMin")] = tf.multiply(TMin[generateParameterName(l, "TMin")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32))

				#store updated validation matrix
				Vseq[generateParameterNameSeq(l, s, "Vseq")] = VseqUpdated
				if(enforceTcontiguityConstraints):
					TMaxSeq[generateParameterNameSeq(l, s, "TMaxSeq")] = TMaxSeqUpdated
					TMinSeq[generateParameterNameSeq(l, s, "TMinSeq")] = TMinSeqUpdated
				sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")] = tf.math.logical_or(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")], ZseqPassThresold)
				
				#calculate thresholded output
				ZseqThresholded = tf.multiply(ZseqHypothetical, tf.dtypes.cast(ZseqPassThresoldInt, tf.float32))
				
				if(allowMultipleContributingSubinputsPerSequentialInput):
					if((resetSequentialInputsIfOnlyFirstInputValid) and (s == 0)):
						ZseqTadjustedThresholded = tf.multiply(ZseqHypotheticalTadjusted, tf.dtypes.cast(ZseqPassThresoldInt, tf.float32))

				#update Zseq
				ZseqUpdated = tf.multiply(Zseq[generateParameterNameSeq(l, s, "Zseq")], tf.dtypes.cast(ZseqPassThresoldNotInt, tf.float32))
				ZseqUpdated = tf.add(ZseqUpdated, ZseqThresholded)
				Zseq[generateParameterNameSeq(l, s, "Zseq")] = ZseqUpdated
				
				if(allowMultipleContributingSubinputsPerSequentialInput):
					if((resetSequentialInputsIfOnlyFirstInputValid) and (s == 0)):
						ZseqTadjustedUpdated = tf.multiply(ZseqTadjusted[generateParameterNameSeq(l, s, "ZseqTadjusted")], tf.dtypes.cast(ZseqPassThresoldNotInt, tf.float32))
						ZseqTadjustedUpdated = tf.add(ZseqTadjustedUpdated, ZseqTadjustedThresholded)
						ZseqTadjusted[generateParameterNameSeq(l, s, "ZseqTadjusted")] = ZseqTadjustedUpdated
				
				
				#regenerate Aseq after Zseq update
				if(allowMultipleSubinputsPerSequentialInput):
					if(performSummationOfSubInputsNonlinear):
						Aseq[generateParameterNameSeq(l, s, "Aseq")] = tf.nn.sigmoid(Zseq[generateParameterNameSeq(l, s, "Zseq")])
					else:
						Aseq[generateParameterNameSeq(l, s, "Aseq")] = Zseq[generateParameterNameSeq(l, s, "Zseq")]
					
				
				#apply weights to input of neuron sequential input
				if(performSummationOfSequentialInputs):
					if(performSummationOfSequentialInputsWeighted):
						multiples = tf.constant([batchSize,1], tf.int32)
						Wtiled = tf.tile(tf.reshape(W[generateParameterName(l, "W")][s], [1, n_h[l]]), multiples)
					
					#these are all used for different methods of sequential input summation
					if(sequentialInputCombinationModeSummation == 1):
						ZseqSum = tf.add(ZseqSum, Zseq[generateParameterNameSeq(l, s, "Zseq")])
					if(sequentialInputCombinationModeSummation == 2):
						AseqSum = tf.math.add(AseqSum, Aseq[generateParameterNameSeq(l, s, "Aseq")])
					if(sequentialInputCombinationModeSummation == 3):
						ZseqWeighted = tf.multiply(Zseq[generateParameterNameSeq(l, s, "Zseq")], Wtiled)
						ZseqWeightedSum = tf.math.add(ZseqWeightedSum, ZseqWeighted)
					if(sequentialInputCombinationModeSummation == 4):	
						AseqWeighted = tf.multiply(Aseq[generateParameterNameSeq(l, s, "Aseq")], Wtiled)
						AseqWeightedSum = tf.math.add(AseqWeightedSum, AseqWeighted)
								
				if(s == numberOfSequentialInputs-1):
					ZseqLast = Zseq[generateParameterNameSeq(l, s, "Zseq")]
					AseqLast = Aseq[generateParameterNameSeq(l, s, "Aseq")]
					VseqLast = Vseq[generateParameterNameSeq(l, s, "Vseq")]
					TMaxSeqLast = TMaxSeq[generateParameterNameSeq(l, s, "TMaxSeq")]
					TMinSeqLast = TMinSeq[generateParameterNameSeq(l, s, "TMinSeq")]
				
					if(recordNetworkWeights):
						if(recordSubInputsWeighted):
							for s2 in range(numberOfSequentialInputs):
								numberSubinputsPerSequentialInput = ANNtf2_algorithmSANIoperations.calculateNumberSubinputsPerSequentialInput(s2)
								multiples = tf.constant([1,numberSubinputsPerSequentialInput,1], tf.int32)
								neuronNewlyActivatedTiled = tf.tile(tf.reshape(ZseqPassThresold, [batchSize, 1, n_h[l]]), multiples)	#or; VseqBool (since will be logically anded with AseqInputVerified)

								AseqInputVerifiedAndNeuronActivated = tf.math.logical_and(AseqInputVerified[generateParameterNameSeq(l, s2, "AseqInputVerified")], neuronNewlyActivatedTiled)
								AseqInputVerifiedAndNeuronActivatedBatchSummed = tf.math.reduce_sum(tf.dtypes.cast(AseqInputVerifiedAndNeuronActivated, tf.float32), axis=0)

								if(recordSequentialInputsWeighted):
									WRseq[generateParameterNameSeq(l, s2, "WRseq")] = tf.add(WRseq[generateParameterNameSeq(l, s2, "WRseq")], AseqInputVerifiedAndNeuronActivatedBatchSummed)

							if(recordNeuronsWeighted):
								neuronNewlyActivated = ZseqPassThresold
								neuronNewlyActivatedBatchSummed = tf.math.reduce_sum(tf.dtypes.cast(neuronNewlyActivated, tf.float32), axis=0)

								if(recordNeuronsWeighted):
									BR[generateParameterName(l, "BR")] = tf.add(BR[generateParameterName(l, "BR")], neuronNewlyActivatedBatchSummed)
					
				if(recordNetworkWeights):	
					if(recordSequentialInputsWeighted):	
						sequentialInputNewlyActivated = ZseqPassThresold
						sequentialInputNewlyActivatedBatchSummed = tf.math.reduce_sum(tf.dtypes.cast(sequentialInputNewlyActivated, tf.float32), axis=0)

						if(recordSequentialInputsWeighted):	
							WR[generateParameterName(l, "WR")] = tf.add(WR[generateParameterName(l, "WR")][s], sequentialInputNewlyActivatedBatchSummed)

				
				
			if(performSummationOfSequentialInputs):
				if(sequentialInputCombinationModeSummation == 1):
					Z1 = ZseqSum
					if(sequentialInputCombinationModeSummationAveraged):
						Z1 = Z1/numberOfSequentialInputs
					if(performSummationOfSequentialInputsNonlinear):
						A1 = tf.nn.sigmoid(Z1)	#no weights are applied
					else:
						A1 = Z1
				elif(sequentialInputCombinationModeSummation == 2):
					Z1 = AseqSum
					if(sequentialInputCombinationModeSummationAveraged):
						Z1 = Z1/numberOfSequentialInputs
					if(performSummationOfSequentialInputsNonlinear):
						A1 = tf.nn.sigmoid(Z1)	#no weights are applied
					else:
						A1 = Z1	
				elif(sequentialInputCombinationModeSummation == 3):
					Z1 = ZseqWeightedSum
					if(performSummationOfSequentialInputsNonlinear):
						A1 = tf.nn.sigmoid(Z1)
					else:
						A1 = Z1
				elif(sequentialInputCombinationModeSummation == 4):
					Z1 = AseqWeightedSum
					if(performSummationOfSequentialInputsNonlinear):
						A1 = tf.nn.sigmoid(Z1)
					else:
						A1 = Z1
				
				if(performSummationOfSequentialInputsVerify):
					Z1 = tf.multiply(Z1, tf.dtypes.cast(VseqLast, tf.float32))
					A1 = tf.multiply(A1, tf.dtypes.cast(VseqLast, tf.float32))
			else:
				#VseqLastFloat = VseqFloat
				Z1 = ZseqLast
				A1 = AseqLast
						
			A[generateParameterName(l, "A")] = A1
			Z[generateParameterName(l, "Z")] = Z1
			
			TMax[generateParameterName(l, "TMax")] = TMaxSeqLast
			TMin[generateParameterName(l, "TMin")] = TMinSeqLast
			
			AprevLayer = A[generateParameterName(l, "A")]
			TMaxPrevLayer = TMax[generateParameterName(l, "TMax")]
			TMinPrevLayer = TMin[generateParameterName(l, "TMin")]
			
	
	ZlastLayer = Z[generateParameterName(numberOfLayers, "Z")]
	if(enforceTcontiguityStartAndEndOfSequence):
		TMaxLastLayer = TMax[generateParameterName(numberOfLayers, "TMax")] 
		TMinLastLayer = TMin[generateParameterName(numberOfLayers, "TMin")]
		TMaxLastLayerThreshold = tf.math.equal(TMaxLastLayer, w-1)
		TMinLastLayerThreshold = tf.math.equal(TMinLastLayer, 0)
		ZlastLayer = tf.multiply(ZlastLayer, tf.dtypes.cast(TMaxLastLayerThreshold, tf.float32))
		ZlastLayer = tf.multiply(ZlastLayer, tf.dtypes.cast(TMinLastLayerThreshold, tf.float32))
	
	#return tf.nn.softmax(ZlastLayer)
	return tf.nn.sigmoid(ZlastLayer)	#binary classification


