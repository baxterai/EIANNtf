# -*- coding: utf-8 -*-
"""SANItf2_algorithmSANIrepeatedModules.py

# Requirements:
Python 3 and Tensorflow 2.1+ 

# License:
MIT License

# Usage:
see SANItf2.py

# Description:

Define Sequentially Activated Neuronal Input (SANI) neural net - repeated modules 

Neural modules cannot be shared between different areas of input sequence.

- Author: Richard Bruce Baxter - Copyright (c) 2020 Baxter AI (baxterai.com)

"""

import tensorflow as tf
import numpy as np
from SANItf2_operations import * #generateParameterNameSeq, generateParameterName
import SANItf2_globalDefs


veryLargeInt = 99999999

supportSkipLayers = True	#will add on average an equal number of connections to each previous layer (can be modified in future to bias number of local or distant connections)
allowMultipleSubinputsPerSequentialInput = False

if(allowMultipleSubinputsPerSequentialInput):
	#[multiple subinputs per sequential input] #each sequential input can detect a pattern of activation from the previous layer

	performIndependentSubInputValidation = True
	
	performSummationOfSubInputs = True	#else take sub input with max input signal*weight
	if(performSummationOfSubInputs):
		performSummationOfSubInputsWeighted = True	#determines if backprop is required to update weight matrix associated with inputs to a sequential input?
		performSummationOfSubInputsNonlinear = True
	else:
		performSummationOfSubInputsWeighted = False
		performSummationOfSubInputsNonlinear = False	
	
	performSummationOfSequentialInputs = True	#else useLastSequentialInputOnly
	if(performSummationOfSequentialInputs):
		performSummationOfSequentialInputsWeighted = False #True	#determines if backprop is required to update weight matrix associated with sequential inputs
		performSummationOfSequentialInputsNonlinear = True
		performSummationOfSequentialInputsVerify = True	#verify that all (last) sequential inputs are activated	#CHECKTHIS
	else:
		performSummationOfSequentialInputsWeighted = False
		performSummationOfSequentialInputsNonlinear = False
		performSummationOfSequentialInputsVerify = False
			
	numberSubinputsPerSequentialInput = 3 #sparsity
	
	sequentialityMode = "default"
	#sequentialityMode = "temporalCrossoverAllowed"
	#sequentialityMode = "contiguousInputEnforced"
	
	if(performSummationOfSequentialInputs):
		if(performSummationOfSequentialInputsWeighted):
			#sequentialInputCombinationModeSummation = 3
			sequentialInputCombinationModeSummation = 4
		else:
			#sequentialInputCombinationModeSummation = 1
			sequentialInputCombinationModeSummation = 2	
	else:
		useLastSequentialInputOnly = True	#implied variable (not used)
else:
	#[single subinput per sequential input] #each sequential input is directly connected to a single neuron on the previous layer

	performIndependentSubInputValidation = False	#always False (ie false by definition because there is only 1 subinput per sequential input)
	
	performSummationOfSequentialInputs = True
	if(performSummationOfSequentialInputs):
		performSummationOfSequentialInputsWeighted = True	#does backprop require to update weight matrix associated with sequential inputs?
		performSummationOfSequentialInputsNonlinear = True
	else:
		performSummationOfSequentialInputsWeighted = False
		performSummationOfSequentialInputsNonlinear = False
			
	sequentialityMode = "default"
	#sequentialityMode = "temporalCrossoverAllowed"
	#sequentialityMode = "contiguousInputEnforced"
	

#variable parameters (tf.variable): 
if(allowMultipleSubinputsPerSequentialInput):
	if(performSummationOfSubInputsWeighted):
		Wseq = {}	#weights matrix
		Bseq = {}	#biases vector
if(performSummationOfSequentialInputsWeighted):
	W = {}	#weights matrix
	B = {}	#biases vector

#static parameters (convert from tf.variable to tf.constant?):
Cseq = {}	#connectivity vector
if(supportSkipLayers):	
	CseqLayer = {}	
	n_h_cumulative = {}
#Network parameters
n_h = []
numberOfLayers = 0
numberOfSequentialInputs = 0

def defineTrainingParametersSANI(dataset, trainMultipleFiles):
	#Training parameters
	if(trainMultipleFiles):
		learningRate = 0.001
		if(dataset == "POStagSequence"):
			trainingSteps = 10000
		elif(dataset == "NewThyroid"):
			trainingSteps = 1000
	else:
		learningRate = 0.001
		if(dataset == "POStagSequence"):
			trainingSteps = 10000
		elif(dataset == "NewThyroid"):
			trainingSteps = 1000

	if(allowMultipleSubinputsPerSequentialInput):
		#if(performIndependentSubInputValidation):
		batchSize = 100
		displayStep = 100
		#else:
		#	batchSize = 50
		#	displayStep = 10
	else:
		batchSize = 50
		displayStep = 100
	
	return learningRate, trainingSteps, batchSize, displayStep
	
def defineNetworkParametersSANI(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, useSmallSentenceLengths):

	#useSmallSentenceLengths not implemented
	
	global n_h
	global numberOfLayers
	global numberOfSequentialInputs

	n_x = num_input_neurons #datasetNumFeatures
	n_y = num_output_neurons  #datasetNumClasses
	n_h_0 = n_x
	if(dataset == "POStagSequence"):
		if(allowMultipleSubinputsPerSequentialInput):
			n_h_1 = int(datasetNumFeatures*3)
			n_h_2 = int(datasetNumFeatures/2)
			n_h_3 = n_y
			n_h = [n_h_0, n_h_1, n_h_2, n_h_3]
		else:
			#n_h_1 = int(datasetNumFeatures*3)
			#n_h_2 = int(datasetNumFeatures/2)
			#n_h_3 = n_y
			#n_h = [n_h_0, n_h_1, n_h_2, n_h_3]
			n_h_1 = int(datasetNumFeatures*100)
			n_h_2 = int(datasetNumFeatures*100)
			n_h_3 = int(datasetNumFeatures*100)
			n_h_4 = int(datasetNumFeatures*100)
			n_h_5 = n_y
			n_h = [n_h_0, n_h_1, n_h_2, n_h_3, n_h_4, n_h_5]
	elif(dataset == "NewThyroid"):
		n_h_1 = 4
		n_h_2 = 4
		n_h_3 = n_y
		n_h = [n_h_0, n_h_1, n_h_2, n_h_3]
	else:
		print("dataset unsupported")
		exit()
	numberOfLayers = len(n_h)-1
	numberOfSequentialInputs = 2	#3
	
	
def neuralNetworkPropagationSANI(x):
		
	batchSize = x.shape[0]

	#note connectivity indexes are used rather than sparse weight matrices due to limitations in current tf2 sparse tensor implementation
	
	#definitions for reference:
	
	#neuron sequential input vars;
	#x/AprevLayer	#output vector (dim: batchSize*n_h[l])
	#if(allowMultipleSubinputsPerSequentialInput):
		#Cseq	#static connectivity matrix (int) - indexes of neurons on prior layer stored; mapped to W  (dim: numberSubinputsPerSequentialInput*n_h[l])
		#if(supportSkipLayers):
			#CseqLayer	#static connectivity matrix (int) - indexes of pior layer stored; mapped to W (dim: numberSubinputsPerSequentialInput*n_h[l])
		#Wseq #weights of connections; see Cseq (dim: numberSubinputsPerSequentialInput*n_h[l])
		#AseqSum	#combination variable
	#else:
		#Cseq	#static connectivity vector (int) - indexes of neurons on prior layer stored; mapped to W - defines which prior layer neuron a sequential input is connected to (dim: n_h[l])
		#if(supportSkipLayers):
			#CseqLayer	#static connectivity matrix (int) - indexes of pior layer stored; mapped to W - defines which prior layer a sequential input is connected to  (dim: n_h[l])
		#Wseq #weights of connections; see Cseq (dim: n_h[l])
	#if(performIndependentSubInputValidation):
		#Vseq	#mutable verification vector (dim: batchSize*numberSubinputsPerSequentialInput*n_h[l] - regenerated for each sequential input index)
		#tMinSeq	#mutable time vector (dim: batchSize*numberSubinputsPerSequentialInput*n_h[l])
		#tMidSeq	#mutable time vector (dim: batchSize*numberSubinputsPerSequentialInput*n_h[l])
		#tMaxSeq	#mutable time vector (dim: batchSize*numberSubinputsPerSequentialInput*n_h[l])
	#else
		#Vseq	#mutable verification vector (dim: batchSize*n_h[l] - regenerated for each sequential input index)
		#tMinSeq	#mutable time vector (dim: batchSize*n_h[l])
		#tMidSeq	#mutable time vector (dim: batchSize*n_h[l])
		#tMaxSeq	#mutable time vector (dim: batchSize*n_h[l])
	#Zseq	#neuron activation function input vector (dim: batchSize*n_h[l]  - regenerated for each sequential input index)
	#Aseq	#neuron activation function output vector (dim: batchSize*n_h[l]  - regenerated for each sequential input index)
	
	#neuron vars;
	#tMin	#mutable time vector (dim: batchSize*n_h[l-1])
	#tMid	#mutable time vector (dim: batchSize*n_h[l-1])
	#tMax	#mutable time vector (dim: batchSize*n_h[l-1])
	#Q
	#Z	#neuron activation function input (dim: batchSize*n_h[l])
	#A	#neuron activation function output (dim: batchSize*n_h[l])
	#if(performSummationOfSequentialInputsWeighted):	
		#W	(dim: numberOfSequentialInputs*n_h[l])
		
	#combination vars (multilayer);
	#if(supportSkipLayers):
		#tMinLayerAll	#mutable time vector (dim: batchSize*(n_h[0]+n_h[1]+...n_h[L]))
		#tMidLayerAll	#mutable time vector (dim: batchSize*(n_h[0]+n_h[1]+...n_h[L]))
		#tMaxLayerAll	#mutable time vector (dim: batchSize*(n_h[0]+n_h[1]+...n_h[L]))
		#AprevLayerAll	#output vector (dim: batchSize*(n_h[0]+n_h[1]+...n_h[L]))
	
	#combination vars (per layer);
	#tMidSeqSum	#(dim: batchSize*n_h[l])
	#if(allowMultipleSubinputsPerSequentialInput):
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
	#else:
		#if(performSummationOfSequentialInputs):
			#AseqInputWeightedSum	#(dim: batchSize*n_h[l])	#aka ZseqWeightedSum
			
			
	AprevLayer = x
	
	for l in range(1, numberOfLayers+1):
		
		#print("\tl = " + str(l))
		
		#declare variables used across all sequential input of neuron
		#primary vars;
		if(l == 1):
			#declare variables:
			tMinL0Row = tf.range(0, n_h[l-1], delta=1, dtype=tf.int32)	#n_h[l-1] = datasetNumFeatures
			tMidL0Row = tf.range(0, n_h[l-1], delta=1, dtype=tf.int32)
			tMaxL0Row = tf.range(0, n_h[l-1], delta=1, dtype=tf.int32)
			multiples = tf.constant([batchSize, 1], tf.int32) 
			tMin = tf.tile(tf.reshape(tMinL0Row, [1, n_h[l-1]]), multiples)
			tMid = tf.tile(tf.reshape(tMidL0Row, [1, n_h[l-1]]), multiples)
			tMax = tf.tile(tf.reshape(tMaxL0Row, [1, n_h[l-1]]), multiples)
			if(supportSkipLayers):
				tMinLayerAll = tMin
				tMidLayerAll = tMid
				tMaxLayerAll = tMax
				AprevLayerAll = AprevLayer	#x
		else:
			tMin = tMinNext
			tMid = tMidNext
			tMax = tMaxNext
			if(supportSkipLayers):
				tMinLayerAll = tf.concat([tMinLayerAll, tMin], 1)
				tMidLayerAll = tf.concat([tMidLayerAll, tMid], 1)
				tMaxLayerAll = tf.concat([tMaxLayerAll, tMax], 1)
				AprevLayerAll = tf.concat([AprevLayerAll, AprevLayer], 1)
			
		#combination vars;
		tMidSeqSum = tf.zeros([batchSize, n_h[l]], tf.int32)
		if(allowMultipleSubinputsPerSequentialInput):
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
		else:
			if(performSummationOfSequentialInputs):
				AseqInputWeightedSum = tf.zeros([batchSize, n_h[l]], tf.float32)
			
		for s in range(numberOfSequentialInputs):
			
			#print("\t\ts = " + str(s))
			
			
			#calculate tMin/Mid/Max for sequential input
			#print("tsLxSx1" + str(tf.timestamp(name="tsLxSx1")))
			if(supportSkipLayers):
				if(l == 1):
					CseqCrossLayerBase = 0
				else:
					CseqCrossLayerBase = tf.gather(n_h_cumulative['n_h_cumulative'], CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")])
				CseqCrossLayer = tf.add(Cseq[generateParameterNameSeq(l, s, "Cseq")], CseqCrossLayerBase)
				tMinSeq = tf.gather(tMinLayerAll, CseqCrossLayer, axis=1)
				tMidSeq = tf.gather(tMidLayerAll, CseqCrossLayer, axis=1)
				tMaxSeq = tf.gather(tMaxLayerAll, CseqCrossLayer, axis=1)	
			else:
				tMinSeq = tf.gather(tMin, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
				tMidSeq = tf.gather(tMid, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
				tMaxSeq = tf.gather(tMax, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
			
			tMinSeqTest = tMinSeq
			tMidSeqTest = tMidSeq
			tMaxSeqTest = tMaxSeq
			if(allowMultipleSubinputsPerSequentialInput):
				if not(performIndependentSubInputValidation):
					tMinSeqReduced = tf.math.reduce_min(tMinSeq, axis=1)
					tMidSeqReduced = tf.math.reduce_mean(tMidSeq, axis=1)
					tMaxSeqReduced = tf.math.reduce_max(tMaxSeq, axis=1)
					tMinSeqTest = tMinSeqReduced
					tMidSeqTest = tMidSeqReduced
					tMaxSeqTest = tMaxSeqReduced

			#print("tsLxSx2" + str(tf.timestamp(name="tsLxSx2")))


			#calculate validation matrix based upon sequentiality requirements
			if(s == 0):
				if(performIndependentSubInputValidation):
					Vseq = tf.fill([batchSize, numberSubinputsPerSequentialInput, n_h[l]], True)
				else:
					Vseq = tf.fill([batchSize, n_h[l]], True)	#all values of Vseq0_l are always set to 1 as they have no sequential dependencies		
			else:
				if(performIndependentSubInputValidation):
					multiples = tf.constant([1, numberSubinputsPerSequentialInput, 1], tf.int32) 
					#printShape(tMinSeqPrev, "tMinSeqPrev")
					tMinSeqPrevTest = tf.tile(tf.reshape(tMinSeqPrev, [batchSize, 1, n_h[l]]), multiples)
					tMidSeqPrevTest = tf.tile(tf.reshape(tMidSeqPrev, [batchSize, 1, n_h[l]]), multiples)
					tMaxSeqPrevTest = tf.tile(tf.reshape(tMaxSeqPrev, [batchSize, 1, n_h[l]]), multiples)
					VseqPrevTest = tf.tile(tf.reshape(VseqPrev, [batchSize, 1, n_h[l]]), multiples)
				else:
					tMinSeqPrevTest = tMinSeqPrev
					tMidSeqPrevTest = tMidSeqPrev
					tMaxSeqPrevTest = tMaxSeqPrev
					VseqPrevTest = VseqPrev

				if(sequentialityMode == "default"):
					#the first sub input of sequential input #2 must fire after the last subinput of sequential input #1
					Vseq = tf.math.greater(tMinSeqTest, tMaxSeqPrevTest)
				elif(sequentialityMode == "temporalCrossoverAllowed"):
					#the last sub input of sequential input #1 can fire after the first subinput of sequential input #2
					Vseq = tf.math.greater(tMaxSeqTest, tMaxSeqPrevTest)
				elif(sequentialityMode == "contiguousInputEnforced"):
					#the last sub input of sequential input #1 must fire immediately before the first subinput of sequentialInput #2
					Vseq = tf.math.equal(tMinSeqTest, tMaxSeqPrevTest+1)	#TODO: verify that the +1 here gets properly broadcasted
				Vseq = tf.math.logical_and(Vseq, VseqPrevTest)	#if previous sequentiality check fails, then all future sequentiality checks must fail
			
			VseqInt = tf.dtypes.cast(Vseq, tf.int32)
			VseqFloat = tf.dtypes.cast(VseqInt, tf.float32)
			
			#identify input of neuron sequential input
			if(supportSkipLayers):
				if(l == 1):
					CseqCrossLayerBase = 0
				else:
					CseqCrossLayerBase = tf.gather(n_h_cumulative['n_h_cumulative'], CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")])
				CseqCrossLayer = tf.add(Cseq[generateParameterNameSeq(l, s, "Cseq")], CseqCrossLayerBase)
				AseqInput = tf.gather(AprevLayerAll, CseqCrossLayer, axis=1)
			else:
				AseqInput = tf.gather(AprevLayer, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
				
			
			#apply validation matrix
			if(allowMultipleSubinputsPerSequentialInput):
				if(performIndependentSubInputValidation):
					AseqInput = tf.multiply(VseqFloat, AseqInput)
				else:
					#checkthis:
					multiples = tf.constant([1,numberSubinputsPerSequentialInput,1], tf.int32)
					VseqFloatTiled = tf.tile(tf.reshape(VseqFloat, [batchSize, 1, n_h[l]]), multiples)
					AseqInput = tf.multiply(VseqFloatTiled, AseqInput)
			else:
				AseqInput = tf.multiply(VseqFloat, AseqInput)


			#apply weights to input of neuron sequential input
			if(performSummationOfSequentialInputsWeighted):
				multiples = tf.constant([batchSize,1], tf.int32)
				Wtiled = tf.tile(tf.reshape(W[generateParameterName(l, "W")][s], [1, n_h[l]]), multiples)
					
			if(allowMultipleSubinputsPerSequentialInput):
				if(performSummationOfSubInputsWeighted):
					multiplesSeq = tf.constant([batchSize,1,1], tf.int32)
					WseqTiled = tf.tile(tf.reshape(Wseq[generateParameterNameSeq(l, s, "Wseq")], [1, numberSubinputsPerSequentialInput, n_h[l]]), multiplesSeq)
					AseqInputWeighted = tf.multiply(AseqInput, WseqTiled)
				else:
					AseqInputWeighted = AseqInput
				
				if(performSummationOfSubInputs):
					Zseq = tf.math.reduce_sum(AseqInputWeighted, axis=1)
				else:
					#take sub input with max input signal*weight
					Zseq = tf.math.reduce_max(AseqInputWeighted, axis=1)
					ZseqIndex = tf.math.argmax(AseqInputWeighted, axis=1)
					
				if(performSummationOfSubInputsNonlinear):	#CHECKTHIS: should be made redundant by choice of sequentialInputCombinationModeSummation
					Aseq = tf.nn.sigmoid(Zseq)	#or relu
				else:
					Aseq = Zseq
				
				if(performSummationOfSequentialInputs):
					#these are all used for different methods of sequential input summation
					if(sequentialInputCombinationModeSummation == 1):
						ZseqSum = tf.add(ZseqSum, Zseq)
					if(sequentialInputCombinationModeSummation == 2):
						AseqSum = tf.math.add(AseqSum, Aseq)
					if(sequentialInputCombinationModeSummation == 3):
						ZseqWeighted = tf.multiply(Zseq, Wtiled)
						ZseqWeightedSum = tf.math.add(ZseqWeightedSum, ZseqWeighted)
					if(sequentialInputCombinationModeSummation == 4):	
						AseqWeighted = tf.multiply(Aseq, Wtiled)
						AseqWeightedSum = tf.math.add(AseqWeightedSum, AseqWeighted)
			else:
				if(performSummationOfSequentialInputs):
					AseqInputWeighted = tf.multiply(AseqInput, Wtiled)
					AseqInputWeightedSum = tf.add(AseqInputWeightedSum, AseqInputWeighted)
			
			#generate reduced versions of tMin/Mid/MaxSeq with only sequentially valid elements
			if(performIndependentSubInputValidation):
				if(performSummationOfSubInputs):
					#mask tMin/Mid/MaxSeq based on sequentiality validation matrix 
					VseqNot = tf.logical_not(Vseq)  
					VseqNotInt = tf.dtypes.cast(VseqNot, tf.int32)
					VseqSumAxis1 = tf.math.reduce_sum(VseqInt, axis=1)
					VseqIntMin = tf.add(tf.multiply(VseqNotInt, veryLargeInt), 1)  	#if VseqInt[x] = 0/False, then VseqIntMin = veryLargeInt. If VseqInt[x] = 1/True, then VseqIntMin = 1
					VseqIntMid = tf.multiply(VseqInt, 1)				#if VseqInt[x] = 0/False, then VseqIntMid = 0. If VseqInt[x] = 1/True, then VseqIntMid = 1
					VseqIntMax = tf.multiply(VseqInt, 1)				#if VseqInt[x] = 0/False, then VseqIntMax = 0. If VseqInt[x] = 1/True, then VseqIntMax = 1
					tMinSeqOnlyValid = tf.multiply(tMinSeq, VseqIntMin)
					tMidSeqOnlyValid = tf.multiply(tMidSeq, VseqIntMid)
					tMaxSeqOnlyValid = tf.multiply(tMaxSeq, VseqIntMax)
					tMinSeqValidatedReduced = tf.math.reduce_min(tMinSeqOnlyValid, axis=1)
					tMidSeqValidatedReduced = tf.divide(tf.math.reduce_sum(tMidSeqOnlyValid, axis=1), VseqSumAxis1)
					tMaxSeqValidatedReduced = tf.math.reduce_max(tMaxSeqOnlyValid, axis=1)
					tMinSeqValidatedReduced = tf.dtypes.cast(tMinSeqValidatedReduced, tf.int32)
					tMidSeqValidatedReduced = tf.dtypes.cast(tMidSeqValidatedReduced, tf.int32)
					tMaxSeqValidatedReduced = tf.dtypes.cast(tMaxSeqValidatedReduced, tf.int32)
					VseqReduced = tf.reduce_any(Vseq, axis=1)
				else:
					#take subinput with max input signal (AseqInput)
					tMinSeqValidatedReduced = tMinSeq[:, ZseqIndex, :]
					tMidSeqValidatedReduced = tMidSeq[:, ZseqIndex, :]
					tMaxSeqValidatedReduced = tMaxSeq[:, ZseqIndex, :]
					VseqReduced = Vseq[:, ZseqIndex, :]
					
			#calculate tMin/Mid/MaxNext (ie the tMin/Mid/Max values to be assigned to the current layer after it has been processed):
			if(s == 0):
				if(performIndependentSubInputValidation):
					tMinSeqFirst = tMinSeqValidatedReduced
				else:
					tMinSeqFirst = tMinSeqTest
			if(s == numberOfSequentialInputs-1):
				if(performIndependentSubInputValidation):
					tMaxSeqLast = tMaxSeqValidatedReduced
				else:
					tMaxSeqLast = tMaxSeqTest
			if(performIndependentSubInputValidation):
				tMidSeqSum = tf.math.add(tMidSeqSum, tMidSeqValidatedReduced)
			else:
				tMidSeqSum = tf.math.add(tMidSeqSum, tMidSeqTest)
			
			if(s == numberOfSequentialInputs-1):
				tMinNext = tMinSeqFirst
				tMidNext = tf.dtypes.cast(tf.math.divide(tMidSeqSum, s+1), tf.int32)
				tMaxNext = tMaxSeqLast			
				
			if(performIndependentSubInputValidation):
				tMinSeqPrev = tMinSeqValidatedReduced
				tMidSeqPrev = tMidSeqValidatedReduced
				tMaxSeqPrev = tMaxSeqValidatedReduced
				VseqPrev = VseqReduced
			else:
				tMinSeqPrev = tMinSeqTest
				tMidSeqPrev = tMidSeqTest
				tMaxSeqPrev = tMaxSeqTest
				VseqPrev = Vseq
			
		
		#calculate A (output) matrix of current layer		
		if(allowMultipleSubinputsPerSequentialInput):
			if(performSummationOfSequentialInputs):	
				if(sequentialInputCombinationModeSummation == 1):
					Z = ZseqSum
					if(performSummationOfSequentialInputsNonlinear):
						A = tf.nn.sigmoid(Z)	#no weights are applied
					else:
						A = Z
				elif(sequentialInputCombinationModeSummation == 2):
					Z = AseqSum
					if(performSummationOfSequentialInputsNonlinear):
						A = tf.nn.sigmoid(Z)	#no weights are applied
					else:
						A = Z
				elif(sequentialInputCombinationModeSummation == 3):
					Z = ZseqWeightedSum
					if(performSummationOfSequentialInputsNonlinear):
						A = tf.nn.sigmoid(Z)
					else:
						A = Z
				elif(sequentialInputCombinationModeSummation == 4):
					Z = AseqWeightedSum
					if(performSummationOfSequentialInputsNonlinear):
						A = tf.nn.sigmoid(Z)
					else:
						A = Z
				if(performSummationOfSequentialInputsVerify):
					Z = tf.multiply(Z, tf.dtypes.cast(VseqLast, tf.float32))
					A = tf.multiply(A, tf.dtypes.cast(VseqLast, tf.float32))
			else:
				ZseqLast = Zseq
				AseqLast = Aseq
				#VseqLastFloat = VseqFloat
				Z = ZseqLast
				A = AseqLast
		else:
			if(performSummationOfSequentialInputsWeighted):
				#Z = tf.add(tf.matmul(AseqAll, W[generateParameterName(l, "W")]), B[generateParameterName(l, "B")])
				Z = AseqInputWeightedSum
				if(performSummationOfSequentialInputsNonlinear):
					A = tf.nn.sigmoid(Z)
				else:
					A = Z
			else:
				#CHECKTHIS;
				ZseqLast = Zseq
				AseqLast = Aseq
				#VseqLastFloat = VseqFloat
				Z = ZseqLast
				A = AseqLast
				

		
		AprevLayer = A
		
	return tf.nn.softmax(Z)

	

def defineNeuralNetworkParametersSANI():

	randomNormal = tf.initializers.RandomNormal()
		
	global n_h_cumulative
	
	if(supportSkipLayers):
		n_h_cumulativeNP = np.zeros((numberOfLayers+1), dtype=int)
		n_h_cumulativeNP[0] = n_h[0]
		
	for l in range(1, numberOfLayers+1):
		for s in range(numberOfSequentialInputs):
			if(allowMultipleSubinputsPerSequentialInput):
				if(supportSkipLayers):
					#neuronIndex = np.random.randint(0, n_h_cumulativeNP[l-1]+1, n_h[l])
					CseqNP = np.zeros((numberSubinputsPerSequentialInput, n_h[l]))
					CseqLayerNP = np.random.randint(0, l, (numberSubinputsPerSequentialInput, n_h[l]))	#this can be modified to make local/distant connections more probable
					for i in range(numberSubinputsPerSequentialInput):
						for j in range(n_h[l]):
							l2 = CseqLayerNP[i, j]
							CseqNP[i,j] = np.random.randint(0, n_h[l2], 1)
					Cseq[generateParameterNameSeq(l, s, "Cseq")] = tf.Variable(CseqNP, dtype=tf.int32)
					CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")] = tf.Variable(CseqLayerNP, dtype=tf.int32)
				else:
					CseqNP = np.random.randint(0, n_h[l-1]+1, (numberSubinputsPerSequentialInput, n_h[l]))	#note +1 is required because np.random.randint generates int between min and max-1
					Cseq[generateParameterNameSeq(l, s, "Cseq")] = tf.Variable(CseqNP, dtype=tf.int32)

				if(performSummationOfSubInputsWeighted):
					WseqNP = np.random.rand(numberSubinputsPerSequentialInput, n_h[l])
					Wseq[generateParameterNameSeq(l, s, "Wseq")] = tf.Variable(WseqNP, dtype=tf.float32)
					Bseq[generateParameterNameSeq(l, s, "Bseq")] = tf.Variable(tf.zeros(n_h[l]), dtype=tf.float32)
			else:
				if(supportSkipLayers):
					#neuronIndex = np.random.randint(0, n_h_cumulativeNP[l-1]+1, n_h[l])
					CseqNP = np.zeros((n_h[l]))
					CseqLayerNP = np.random.randint(0, l, n_h[l])	#this can be modified to make local/distant connections more probable
					for index, l2 in enumerate(CseqLayerNP):
						CseqNP[index] = np.random.randint(0, n_h[l2], 1)
					Cseq[generateParameterNameSeq(l, s, "Cseq")] = tf.Variable(CseqNP, dtype=tf.int32)
					CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")] = tf.Variable(CseqLayerNP, dtype=tf.int32)
				else:
					CseqNP = np.random.randint(0, n_h[l-1]+1, n_h[l])	#note +1 is required because np.random.randint generates int between min and max-1
					Cseq[generateParameterNameSeq(l, s, "Cseq")] = tf.Variable(CseqNP, dtype=tf.int32)

		if(performSummationOfSequentialInputsWeighted):	
			W[generateParameterName(l, "W")] = tf.Variable(randomNormal([numberOfSequentialInputs, n_h[l]], dtype=tf.float32))	#randomNormal	#note the architecture of this weight matrix is different than a normal weight matrix. Every set of numberOfSequentialInputs (e.g. 3) represents a unique set of sequential neuron inputs (artificial neurons), and are mapped to an independent neuron (real neuron)
			if(allowMultipleSubinputsPerSequentialInput):
				B[generateParameterName(l, "B")] = tf.Variable(tf.zeros(n_h[l]), tf.float32)
				
		if(supportSkipLayers):
			n_h_cumulativeNP[l] = n_h_cumulativeNP[l-1] + n_h[l]
			  
	if(supportSkipLayers):
		n_h_cumulative['n_h_cumulative'] = tf.Variable(n_h_cumulativeNP, dtype=tf.int32)

def printShape(tensor, tensorName):
	print(tensorName + ".shape = ")
	print(tensor.shape)
	
