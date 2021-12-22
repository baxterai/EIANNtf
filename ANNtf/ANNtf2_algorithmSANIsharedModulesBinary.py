"""ANNtf2_algorithmSANIsharedModulesBinary.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2021 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNtf2.py

# Usage:
see ANNtf2.py

# Description:
ANNtf algorithm SANI shared modules binary - define Sequentially Activated Neuronal Input neural network with shared modules and binary weights and activation signals

See shared modules

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

def defineNetworkParametersSANIwrapper(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, useSmallSentenceLengths, numberOfFeaturesPerWord):
	global n_h
	global numberOfLayers
	n_h, numberOfLayers = ANNtf2_algorithmSANIoperations.defineNetworkParametersSANI(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, useSmallSentenceLengths, numberOfFeaturesPerWord)
	
def defineTrainingParametersSANIwrapper(dataset, trainMultipleFiles):
	return ANNtf2_algorithmSANIoperations.defineTrainingParametersSANI(dataset, trainMultipleFiles)
	

def defineNeuralNetworkParameters():
	global n_h_cumulative
	ANNtf2_algorithmSANIoperations.defineNeuralNetworkParametersSANI(n_h, numberOfLayers, Cseq, CseqLayer, n_h_cumulative, WRseq, WR, BR, Wseq, Bseq, W, B)
			

#temporary variables for neuralNetworkPropagationSANI:
if(algorithmSANI == "sharedModulesBinary"):
	Vseq = {}
	Zseq = {}
	Aseq = {}
	TMaxSeq = {}
	TMinSeq = {}
	ZseqTadjusted = {}
	Z = {}
	A = {}
	T = {}
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

	#Cseq	#static connectivity matrix (int) - indexes of neurons on prior layer stored; mapped to W  (dim: maxNumberSubinputsPerSequentialInput*n_h[l])
	#if(supportSkipLayers):
		#CseqLayer	#static connectivity matrix (int) - indexes of pior layer stored; mapped to W (dim: maxNumberSubinputsPerSequentialInput*n_h[l])
	#AseqSum	#combination variable

	#Vseq	#mutable verification vector (dim: batchSize*n_h[l] - regenerated for each sequential input index)			#records whether particular neuron sequential inputs are currently active
	#TMaxSeq	#mutable time vector (dim: batchSize*n_h[l] - regenerated for each sequential input index)				#records the time at which a particular sequential input last fired
	#TMinSeq	#mutable time vector (dim: batchSize*n_h[l] - regenerated for each sequential input index)				#records the time at which a first encapsulated subinput fired
	#Zseq	#neuron activation function output vector (dim: batchSize*n_h[l]  - regenerated for each sequential input index)	#records the current output value of sequential inputs
	#Aseq	#neuron activation function output vector (dim: batchSize*n_h[l]  - regenerated for each sequential input index)	#records the current output value of sequential inputs
	#if(recordSubInputsWeighted):
		#AseqInputVerified	#neuron input (sequentially verified) (dim: batchSize*numberSubinputsPerSequentialInput*n_h[l]  - records the subinputs that were used to activate the sequential input)
		#WRseq #weights of connections; see Cseq (dim: maxNumberSubinputsPerSequentialInput*n_h[l])
		
	#Q  
	#Z	#neuron activation function input (dim: batchSize*n_h[l])
	#A	#neuron activation function output (dim: batchSize*n_h[l])
	#TMax	#mutable time vector (dim: batchSize*n_h[l]) - same as TMaxSeq[numberOfSequentialInputs-1]
	#TMin	#mutable time vector (dim: batchSize*n_h[l]) - same as TMinSeq[numberOfSequentialInputs-1]
	#if(recordSequentialInputsWeighted):	
		#W	(dim: numberOfSequentialInputs*n_h[l])	
	#if(recordNeuronsWeighted):	
		#B	(dim: n_h[l])	
				
	batchSize = x.shape[0]
	
	#optimise feed length based on max sentence length in batch:
	#unoptimised: numberOfFeatures = x.shape[1]
	
	xIsNotPadding = tf.math.less(x, paddingTagIndex)
	coordinatesOfNotPadding = tf.where(xIsNotPadding)
	numberOfFeaturesCropped = tf.reduce_max(coordinatesOfNotPadding[:, 1]).numpy()+1

	#method 2 trial;
	#match_indices2 = tf.where(tf.equal(paddingTagIndex, x), x=tf.range(tf.shape(x)[1])*tf.ones_like(x), y=0*tf.ones_like(x))
	#print(match_indices2)
	#numberOfFeaturesCropped = tf.reduce_max(match_indices2)
	#print("numberOfFeaturesCropped = ", numberOfFeaturesCropped)

	inputLength = numberOfFeaturesPerWord*numberOfWordsInConvolutionalWindowSeen
	
	maxNumberOfWordsInSentenceBatch = int(numberOfFeaturesCropped/numberOfFeaturesPerWord)
	
	match_indices = tf.where(tf.equal(paddingTagIndex, x), x=tf.range(tf.shape(x)[1])*tf.ones_like(x), y=(tf.shape(x)[1])*tf.ones_like(x))
	numberOfFeaturesActiveBatch = tf.math.argmin(match_indices, axis=1).numpy()
	numberOfWordsInSentenceBatch = (numberOfFeaturesActiveBatch/numberOfFeaturesPerWord).astype(np.int32) 
	print(numberOfWordsInSentenceBatch)
	
	for l in range(1, numberOfLayers+1):
		Z[generateParameterName(l, "Z")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), dtype=tf.bool), dtype=tf.bool)	#A=Z for binary activations
		A[generateParameterName(l, "A")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), dtype=tf.bool), dtype=tf.bool)	#A=Z for binary activations
		TMax[generateParameterName(l, "TMax")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.int32))
		TMin[generateParameterName(l, "TMin")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.int32))
		for s in range(numberOfSequentialInputs):
			Vseq[generateParameterNameSeq(l, s, "Vseq")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), tf.bool), dtype=tf.bool)
			TMaxSeq[generateParameterNameSeq(l, s, "TMaxSeq")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.int32))
			TMinSeq[generateParameterNameSeq(l, s, "TMinSeq")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.int32))
			Zseq[generateParameterNameSeq(l, s, "Zseq")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), dtype=tf.bool), dtype=tf.bool)	#A=Z for binary activations
			Aseq[generateParameterNameSeq(l, s, "Aseq")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), dtype=tf.bool), dtype=tf.bool)	#A=Z for binary activations
			if(recordSubInputsWeighted):
				numberSubinputsPerSequentialInput = ANNtf2_algorithmSANIoperations.calculateNumberSubinputsPerSequentialInput(s)
				AseqInputVerified[generateParameterNameSeq(l, s, "AseqInputVerified")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, numberSubinputsPerSequentialInput, n_h[l]]), dtype=tf.bool), dtype=tf.bool)
			
			
	for w in range(maxNumberOfWordsInSentenceBatch-numberOfWordsInConvolutionalWindowSeen+1):
	
		#print("w = " + str(w))
		
		for l in range(1, numberOfLayers+1):
			sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), tf.bool), dtype=tf.bool)

		if(w == 0):
			AfirstLayerShifted =  tf.dtypes.cast(x[:, 0:inputLength], tf.bool)
		else:
			paddings = tf.constant([[0, 0], [w*numberOfFeaturesPerWord, 0]])	#shift input to the right by x words (such that a different input window will be presented to the network)
			#AfirstLayerShifted = x[:, w*numberOfFeaturesPerWord:min(w*numberOfFeaturesPerWord+inputLength, numberOfFeaturesCropped)]
			AfirstLayerShifted = tf.dtypes.cast(x[:, w*numberOfFeaturesPerWord:w*numberOfFeaturesPerWord+inputLength], tf.bool)
			#print("AfirstLayerShifted = ", AfirstLayerShifted)
			#print("paddings = ", paddings)
			tf.pad(AfirstLayerShifted, paddings, "CONSTANT")
		
		#printShape(AfirstLayerShifted, "AfirstLayerShifted")	
		#printAverage(AfirstLayerShifted, "AfirstLayerShifted", 1)
		
		AprevLayer = AfirstLayerShifted
		
		TMaxPrevLayer = tf.ones(([batchSize, n_h[0]]), tf.int32)
		TMaxPrevLayer = TMaxPrevLayer*w
		TMinPrevLayer = tf.ones(([batchSize, n_h[0]]), tf.int32)
		TMinPrevLayer = TMinPrevLayer*w
		
		printAverage(AprevLayer, "AprevLayer", 1)
		printAverage(TMaxPrevLayer, "TMaxPrevLayer", 1)
		printAverage(TMinPrevLayer, "TMinPrevLayer", 1)
		
		for l in range(1, numberOfLayers+1):	#start algorithm at n_h[1]; ie first hidden layer

			#print("\tl = " + str(l))

			#declare variables used across all sequential input of neuron
			#primary vars;
			if(l == 1):
				if(supportSkipLayers):
					AprevLayerAll = AprevLayer	#x
					TMaxPrevLayerAll = TMaxPrevLayer
					TMinPrevLayerAll = TMinPrevLayer
					
					#print(TMinPrevLayer)
					#print(TMaxPrevLayer)
					#print(TMinPrevLayerAll)
					#print(TMaxPrevLayerAll)	
			else:
				if(supportSkipLayers):
					AprevLayerAll = tf.concat([AprevLayerAll, AprevLayer], 1)
					TMaxPrevLayerAll = tf.concat([TMaxPrevLayerAll, TMaxPrevLayer], 1)
					TMinPrevLayerAll = tf.concat([TMinPrevLayerAll, TMinPrevLayer], 1)

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
					if(supportSkipLayers):
						CseqCrossLayerBase = tf.gather(n_h_cumulative['n_h_cumulative'], CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")])
						CseqCrossLayer = tf.add(Cseq[generateParameterNameSeq(l, s, "Cseq")], CseqCrossLayerBase)

						AseqInput = tf.gather(AprevLayerAll, CseqCrossLayer, axis=1)

						TMaxSeqInput = tf.gather(TMaxPrevLayerAll, CseqCrossLayer, axis=1)
						TMinSeqInput = tf.gather(TMinPrevLayerAll, CseqCrossLayer, axis=1)
					else:
						#printAverage(AprevLayer, "AprevLayer", 3)
						#printAverage(TprevLayer, "TprevLayer", 3)
						AseqInput = tf.gather(AprevLayer, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
						TMaxSeqInput = tf.gather(TMaxPrevLayer, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
						TMinSeqInput = tf.gather(TMinPrevLayer, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)

				
				#printAverage(AseqInput, "AseqInput", 3)
				#printAverage(TMaxSeqInput, "TMaxSeqInput", 3)
				#printAverage(TMinSeqInput, "TMinSeqInput", 3)
				
				#calculate validation matrix based upon sequentiality requirements
				#if Vseq[s-1] is True and Vseq[s] is False;
					#OLD: and Tseq[s-1] < w [this condition is guaranteed by processing s in reverse]
				if(s > 0):
					VseqTemp = Vseq[generateParameterNameSeq(l, s-1, "Vseq")]
					VseqTemp = tf.math.logical_and(VseqTemp, tf.math.logical_not(Vseq[generateParameterNameSeq(l, s, "Vseq")]))
				else:
					if(resetSequentialInputsIfOnlyFirstInputValid):
						VseqTemp = tf.math.logical_not(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")])
					else:
						VseqTemp = tf.math.logical_and(VseqTemp, tf.math.logical_not(Vseq[generateParameterNameSeq(l, s, "Vseq")]))

				VseqBool = VseqTemp
			
				#printAverage(VseqBool, "VseqBool", 3)
			
				#AseqInput = tf.multiply(VseqBool, AseqInput)
			
				#calculate output for layer sequential input s

				#ensure that T continguous constraint is met;
				#NO: take the max subinput pathway only (ie matrix mult but with max() rather than sum() for each dot product)
				if(s > 0):
					numberSubinputsPerSequentialInput = ANNtf2_algorithmSANIoperations.calculateNumberSubinputsPerSequentialInput(s)
					multiples = tf.constant([1,numberSubinputsPerSequentialInput,1], tf.int32)

					if(enforceTcontiguityBetweenSequentialInputs):
						#1. first select the AseqInput inputs that have TMinSeq[l, s] contiguous with TMaxSeq[l, s-1]:
						TMaxSeqPrevPlus1 = TMaxSeq[generateParameterNameSeq(l, s-1, "TMaxSeq")]+1
						TMinSeqPrev = TMinSeq[generateParameterNameSeq(l, s-1, "TMinSeq")]

						#printAverage(TMaxSeqPrevPlus1, "TMaxSeqPrevPlus1PRE", 3)	
						TMaxSeqPrevPlus1Tiled = tf.tile(tf.reshape(TMaxSeqPrevPlus1, [batchSize, 1, n_h[l]]), multiples)
						#printAverage(TMaxSeqPrevPlus1Tiled, "TMaxSeqPrevPlus1TiledPRE", 3)
						TMinSeqInputThreshold = tf.math.equal(TMinSeqInput, TMaxSeqPrevPlus1Tiled)
						#printAverage(TMinSeqInputThreshold, "TMinSeqInputThresholdPRE", 3)	
						AseqInputTthresholded = tf.math.logical_and(AseqInput, TMinSeqInputThreshold)
						
						TMinSeqInputThresholded = TMinSeqPrev	#note this is recording the min of the sequence, not the sequential input	#OLD: TMaxSeqPrevPlus1
						
						#printAverage(AseqInputTthresholded, "AseqInputTthresholdedPRE", 3)			

						AseqInput = AseqInputTthresholded
						
					if(enforceTcontiguityTakeEncapsulatedSubinputWithMaxTvalueEqualsW):
						TMaxSeqInputThreshold = tf.math.equal(TMaxSeqInput, w)	#CHECKTHIS
						AseqInputTthresholded = tf.math.logical_and(AseqInput, TMaxSeqInputThreshold)

						TMaxSeqInputThresholded = tf.math.reduce_max(TMaxSeqInput, axis=1)	#CHECKTHIS - ensure equal to w
						
						AseqInput = AseqInputTthresholded
						
				else:
					if(resetSequentialInputsIfOnlyFirstInputValid):
						#only reset first sequential input if TMaxSeqInput > TMax[l]

						numberSubinputsPerSequentialInput = ANNtf2_algorithmSANIoperations.calculateNumberSubinputsPerSequentialInput(s)
						multiples = tf.constant([1,numberSubinputsPerSequentialInput,1], tf.int32)
						
						TMaxtiled = tf.tile(tf.reshape(TMax[generateParameterName(l, "TMax")], [batchSize, 1, n_h[l]]), multiples)
						VseqTiled = tf.tile(tf.reshape(Vseq[generateParameterNameSeq(l, s, "Vseq")], [batchSize, 1, n_h[l]]), multiples)

						ZseqHypotheticalResetThreshold = tf.math.logical_or(tf.math.logical_not(VseqTiled), tf.math.greater(TMaxSeqInput, tf.dtypes.cast(TMaxtiled, tf.int32)))
						AseqInputTthresholded = tf.math.logical_and(AseqInput, ZseqHypotheticalResetThreshold)
						
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
							AseqInputTthresholded = tf.math.logical_and(AseqInput, TMaxSeqInputThreshold)

							TMaxSeqInputThresholded = tf.math.reduce_max(TMaxSeqInput, axis=1)	#CHECKTHIS - ensure equal to w
													
							AseqInput = AseqInputTthresholded
				
				
				
				#printAverage(AseqInputTthresholded, "AseqInputTthresholded", 3)			
				#printAverage(TMinSeqInputThresholded, "TMinSeqInputThresholded", 3)
				#printAverage(TMaxSeqInputThresholded, "TMaxSeqInputThresholded", 3)

				
				#take any active and Tthreshold valid sub input:
				ZseqHypothetical = tf.math.reduce_any(AseqInput, axis=1)
			
				#printAverage(ZseqHypothetical, "ZseqHypothetical", 3)
								
				#printAverage(VseqBool, "VseqBool", 3)			
				#printAverage(ZseqHypothetical, "ZseqHypothetical", 3)	
				
				#apply sequential validation matrix
				ZseqHypothetical = tf.math.logical_and(VseqBool, ZseqHypothetical)
				
		

				#printAverage(ZseqHypothetical, "ZseqHypothetical (2)", 3)
				
				#check output threshold
				ZseqPassThresold = ZseqHypothetical	#redundant
				
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
						Vseq[generateParameterNameSeq(l, s2, "Vseq")] = tf.math.logical_and(Vseq[generateParameterNameSeq(l, s2, "Vseq")], tf.math.logical_not(resetRequiredMatrix))
						if(recordSubInputsWeighted):
							multiples = tf.constant([1,numberSubinputsPerSequentialInput,1], tf.int32)
							resetRequiredMatrixTiled = tf.tile(tf.reshape(resetRequiredMatrix, [batchSize, 1, n_h[l]]), multiples)
							AseqInputVerified[generateParameterNameSeq(l, s2, "AseqInputVerified")] = tf.math.logical_and(AseqInputVerified[generateParameterNameSeq(l, s2, "AseqInputVerified")], tf.math.logical_not(resetRequiredMatrixTiled))
						if(not doNotResetNeuronOutputUntilAllSequentialInputsActivated):
							TMaxSeq[generateParameterNameSeq(l, s2, "TMaxSeq")] = tf.multiply(TMaxSeq[generateParameterNameSeq(l, s2, "TMaxSeq")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32))
							TMinSeq[generateParameterNameSeq(l, s2, "TMinSeq")] = tf.multiply(TMinSeq[generateParameterNameSeq(l, s2, "TMinSeq")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32))
							Zseq[generateParameterNameSeq(l, s2, "Zseq")] = tf.math.logical_and(Zseq[generateParameterNameSeq(l, s2, "Zseq")], tf.math.logical_not(resetRequiredMatrix))
							Aseq[generateParameterNameSeq(l, s2, "Aseq")] = tf.math.logical_and(Aseq[generateParameterNameSeq(l, s2, "Aseq")], tf.math.logical_not(resetRequiredMatrix))
					sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")] = tf.math.logical_and(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")], tf.math.logical_not(resetRequiredMatrix))
					if(not doNotResetNeuronOutputUntilAllSequentialInputsActivated):
						Z[generateParameterName(l, "Z")] = tf.math.logical_and(Z[generateParameterName(l, "Z")], tf.math.logical_not(resetRequiredMatrix))
						A[generateParameterName(l, "A")] = tf.math.logical_and(A[generateParameterName(l, "A")], tf.math.logical_not(resetRequiredMatrix))
						TMax[generateParameterName(l, "TMax")] = tf.multiply(TMax[generateParameterName(l, "TMax")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32))
						TMin[generateParameterName(l, "TMin")] = tf.multiply(TMin[generateParameterName(l, "TMin")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32))

				#store updated validation matrix
				Vseq[generateParameterNameSeq(l, s, "Vseq")] = VseqUpdated
				if(enforceTcontiguityConstraints):
					TMaxSeq[generateParameterNameSeq(l, s, "TMaxSeq")] = TMaxSeqUpdated
					TMinSeq[generateParameterNameSeq(l, s, "TMinSeq")] = TMinSeqUpdated
				sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")] = tf.math.logical_or(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")], ZseqPassThresold)
				
				
				#calculate thresholded output
				ZseqThresholded = ZseqPassThresold	#redundant

				#update Zseq
				ZseqUpdated = tf.math.logical_or(Zseq[generateParameterNameSeq(l, s, "Zseq")], ZseqPassThresold) 
				Zseq[generateParameterNameSeq(l, s, "Zseq")] = ZseqUpdated
			
				#printAverage(ZseqPassThresold, "ZseqPassThresold", 3)
				#printAverage(VseqUpdated, "VseqUpdated", 3)
				#printAverage(ZseqUpdated, "ZseqUpdated", 3)	
				#printAverage(TMaxSeqUpdated, "TMaxSeqUpdated", 3)
				#printAverage(TMinSeqUpdated, "TMinSeqUpdated", 3)
				
				#regenerate Aseq after Zseq update
				Aseq[generateParameterNameSeq(l, s, "Aseq")] = Zseq[generateParameterNameSeq(l, s, "Zseq")]
				
				#printAverage(Zseq[generateParameterNameSeq(l, s, "Zseq")], "Zseq", 3)
				
				if(recordSubInputsWeighted):
					#apply sequential verification matrix to AseqInput (rather than ZseqHypothetical) to record the precise subinputs that resulted in a sequential input being activated:
					
					numberSubinputsPerSequentialInput = ANNtf2_algorithmSANIoperations.calculateNumberSubinputsPerSequentialInput(s)
					multiples = tf.constant([1,numberSubinputsPerSequentialInput,1], tf.int32)
					
					VseqBoolTiled =  tf.tile(tf.reshape(VseqBool, [batchSize, 1, n_h[l]]), multiples)
					AseqInputVerifiedTemp = tf.math.logical_and(AseqInput, VseqBoolTiled)
											
					AseqInputVerified[generateParameterNameSeq(l, s, "AseqInputVerified")] = tf.math.logical_or(AseqInputVerified[generateParameterNameSeq(l, s, "AseqInputVerified")], AseqInputVerifiedTemp)			
				
				if(s == numberOfSequentialInputs-1):
					ZseqLast = Zseq[generateParameterNameSeq(l, s, "Zseq")]
					AseqLast = Aseq[generateParameterNameSeq(l, s, "Aseq")]
					VseqLast = Vseq[generateParameterNameSeq(l, s, "Vseq")]
					TMaxSeqLast = TMaxSeq[generateParameterNameSeq(l, s, "TMaxSeq")]
					TMinSeqLast = TMinSeq[generateParameterNameSeq(l, s, "TMinSeq")]
				
					#printAverage(TMaxSeqLast, "TMaxSeqLast", 3)
					#printAverage(TMinSeqLast, "TMinSeqLast", 3)
				
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
				
			Z1 = ZseqLast
			A1 = AseqLast
											
			A[generateParameterName(l, "A")] = A1
			Z[generateParameterName(l, "Z")] = Z1
			
			TMax[generateParameterName(l, "TMax")] = TMaxSeqLast
			TMin[generateParameterName(l, "TMin")] = TMinSeqLast
			
			AprevLayer = A[generateParameterName(l, "A")]
			TMaxPrevLayer = TMax[generateParameterName(l, "TMax")]
			TMinPrevLayer = TMin[generateParameterName(l, "TMin")]
	
	#return tf.nn.softmax(Z1)
	#return tf.nn.sigmoid(Z1)	#binary classification
	
	ZlastLayer = Z[generateParameterName(numberOfLayers, "Z")]
	if(enforceTcontiguityStartAndEndOfSequence):
		TMaxLastLayer = TMax[generateParameterName(numberOfLayers, "TMax")] 
		TMinLastLayer = TMin[generateParameterName(numberOfLayers, "TMin")]
		
		printAverage(ZlastLayer, "ZlastLayer", 1)
		printAverage(TMaxLastLayer, "TMaxLastLayer", 1)
		printAverage(TMinLastLayer, "TMinLastLayer", 1)
		
		#multiples = tf.constant([1,numberOfLayers], tf.int32)
		#numberOfWordsInSentenceBatchTiled = tf.tile(tf.reshape(numberOfWordsInSentenceBatch, [batchSize, n_h[numberOfLayers]]), multiples)
		multiples = tf.constant([1,n_h[numberOfLayers]], tf.int32)
		numberOfWordsInSentenceBatchTiled = tf.tile(tf.reshape(numberOfWordsInSentenceBatch, [batchSize, 1]), multiples)
		
		TMaxLastLayerThreshold = tf.math.equal(TMaxLastLayer, numberOfWordsInSentenceBatchTiled-1)
		printAverage(TMaxLastLayerThreshold, "TMaxLastLayerThreshold", 1)
		TMinLastLayerThreshold = tf.math.equal(TMinLastLayer, 0)
		printAverage(TMaxLastLayerThreshold, "TMaxLastLayerThreshold", 1)
		
		ZlastLayer = tf.math.logical_and(ZlastLayer, TMaxLastLayerThreshold)
		printAverage(ZlastLayer, "ZlastLayer", 1)
		ZlastLayer = tf.math.logical_and(ZlastLayer, TMinLastLayerThreshold)
		printAverage(ZlastLayer, "ZlastLayer", 1)
	
	return tf.math.reduce_any(ZlastLayer, axis=1)

	

