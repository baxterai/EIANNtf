# -*- coding: utf-8 -*-
"""SANItf2_algorithmSANI.py

# Requirements:
Python 3 and Tensorflow 2.1+ 

# License:
MIT License

# Execution:
see SANItf2.py

# Description:

Define Sequentially Activated Neuronal Input (SANI) ANN

- Author: Richard Bruce Baxter - Copyright (c) 2020 Baxter AI (baxterai.com)

"""

import tensorflow as tf
import numpy as np
from SANItf2_operations import * #generateParameterNameSeq, generateParameterName


tMinMidMaxUpdateMode = "fastApproximation"
#tMinMidMaxUpdateMode = "slowExact"

useSparseTensors = False
sparsityRatioSeq = 10	#10x: 0s to 1s ratio - sparsity ratio for connections to previous layer for each sequential input

sequentialityMode = "default"
#sequentialityMode = "temporalCrossoverAllowed"
#sequentialityMode = "contiguousInputEnforced"
sequentialInputCombinationMode2 = "summation"			
#sequentialInputCombinationMode2 = "useLastSequentialInputOnly"
sequentialInputCombinationMode1 = 1
#sequentialInputCombinationMode1 = 2
#sequentialInputCombinationMode1 = 3
				

Wseq = {}	#weights matrix
Bseq = {}	#biases vector
Cseq = {}	#biases vector

#W = {}	#weights matrix
#B = {}	#biases vector


#Network parameters
n_h = []
numberOfLayers = 0
numberOfSequentialInputs = 0

def defineNetworkParametersSANI(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset):

	global n_h
	global numberOfLayers
	global numberOfSequentialInputs

	n_x = num_input_neurons #datasetNumFeatures
	n_y = num_output_neurons  #datasetNumClasses
	n_h_0 = n_x
	if(dataset == "POStagSequence"):
		n_h_1 = int(datasetNumFeatures*3) # 1st layer number of neurons.
		n_h_2 = int(datasetNumFeatures/2) # 2nd layer number of neurons.
	elif(dataset == "NewThyroid"):
		n_h_1 = 4
		n_h_2 = 4
	n_h_3 = n_y
	n_h = [n_h_0, n_h_1, n_h_2, n_h_3]
	numberOfLayers = 3
	numberOfSequentialInputs = 3
	
	
def neuralNetworkPropagationSANI(x):
	
	global useSparseTensors
	global sparsityRatioSeq
	global sequentialityMode
	global sequentialInputCombinationMode2
	global sequentialInputCombinationMode1
	
	global n_h
	global numberOfLayers
	global numberOfSequentialInputs
	
	batchSize = x.shape[0]

	#definitions for reference:
	#tMinSeq	#mutable time vector
	#tMidSeq	#mutable time vector
	#tMaxSeq	#mutable time vector
	#Cseq	#static connectivity matrix (bool) - defines sparsity of connection matrix W
	#Vseq	#mutable verification vector
	#Zseq	#neuron activation function input vector
	#Aseq	#neuron activation function output vector
	#tMin	#mutable time vector
	#tMid	#mutable time vector
	#tMax	#mutable time vector
	#Q
	#Z	#neuron activation function input
	#A	#neuron activation function output
	#tMidSeqSum	#combination variable
	#AseqSum	#combination variable

	#print("x.shape") 
	#print(x.shape)	
	
	AprevLayer = x
	
	for l in range(1, numberOfLayers+1):
		
		#print("\tl = " + str(l))
		
		#declare variables:
		#vectors:
		if(l == 1):
			#declare variables:
			tMinL0Row = tf.range(0, n_h[l-1], delta=1, dtype=tf.int32)	#n_h[l-1] = datasetNumFeatures
			tMidL0Row = tf.range(0, n_h[l-1], delta=1, dtype=tf.int32)
			tMaxL0Row = tf.range(0, n_h[l-1], delta=1, dtype=tf.int32)
			multiples = tf.constant([batchSize, 1], tf.int32) 
			tMin = tf.tile(tf.reshape(tMinL0Row, [1, n_h[l-1]]), multiples)
			tMid = tf.tile(tf.reshape(tMidL0Row, [1, n_h[l-1]]), multiples)
			tMax = tf.tile(tf.reshape(tMaxL0Row, [1, n_h[l-1]]), multiples)
		else:
			tMin = tMinNext
			tMid = tMidNext
			tMax = tMaxNext
			
		#combination vars;
		tMidSeqSum = tf.zeros([batchSize, n_h[l]], tf.int32)
		AseqSum = tf.zeros([batchSize, n_h[l]], tf.float32)
		ZseqSum = tf.zeros([batchSize, n_h[l]], tf.float32)
		
		for s in range(numberOfSequentialInputs):
			
			#print("\t\ts = " + str(s))
			
			#print("tsLxSx1" + str(tf.timestamp(name="tsLxSx1")))
			
			#this is where a slow down is occuring by a factor of approx 1000
			
			#if not useSparseTensors:
			#	Wseq[generateParameterNameSeq(l, s, "Wseq")] = tf.multiply(Wseq[generateParameterNameSeq(l, s, "Wseq")], tf.dtypes.cast(tf.dtypes.cast(Cseq[generateParameterNameSeq(l, s, "Cseq")], tf.int32), tf.float32))		#reset weights for unconnected weights to zero in case they have been updated by backprop

			if(tMinMidMaxUpdateMode == "fastApproximation"):
				#version 2 (fast: ~0.006s) (fast but inaccurate calculation of min/max; requires single connection per sequential input - ie max sparsity - for calculations to be accurate)
				tMinSeq = tf.matmul(tMin, tf.dtypes.cast(Cseq[generateParameterNameSeq(l, s, "Cseq")], tf.int32))
				tMidSeq = tf.matmul(tMid, tf.dtypes.cast(Cseq[generateParameterNameSeq(l, s, "Cseq")], tf.int32))
				tMaxSeq = tf.matmul(tMax, tf.dtypes.cast(Cseq[generateParameterNameSeq(l, s, "Cseq")], tf.int32))      
				CnumConnectionsVector = tf.math.reduce_sum(tf.dtypes.cast(Cseq[generateParameterNameSeq(l, s, "Cseq")], tf.int32), axis=0)
				multiples = tf.constant([batchSize,1], tf.int32) 
				CnumConnectionsMatrix = tf.tile(tf.reshape(CnumConnectionsVector, [1,n_h[l]]), multiples)
				tMinSeq = tf.dtypes.cast(tf.divide(tMinSeq, CnumConnectionsMatrix), tf.int32)   #element wise division
				tMidSeq = tf.dtypes.cast(tf.divide(tMidSeq, CnumConnectionsMatrix), tf.int32)  #element wise division
				tMaxSeq = tf.dtypes.cast(tf.divide(tMaxSeq, CnumConnectionsMatrix), tf.int32)  #element wise division
			elif(tMinMidMaxUpdateMode == "slowExact"):
				#version 1 (slow: ~6.0s) (slower by a factor of ~x1000)
				#              C     _n_
				#                  o|
				#                   |
				# tminTiled  _o_  *
				#	s*o|
				#          |
				#          |
				#          |
				#	
				#multiples = tf.constant([n_h[l-1],1], tf.int32) 
				#tMinTiled = tf.tile(tMin, multiples)
				#tMidTiled = tf.tile(tMid, multiples)
				#tMaxTiled = tf.tile(tMax, multiples)
				#tMinMasked = tf.matmul(tMinTiled, tf.dtypes.cast(Cseq[generateParameterNameSeq(l, s, "Cseq")], tf.int32))
				#tMidMasked = tf.matmul(tMidTiled, tf.dtypes.cast(Cseq[generateParameterNameSeq(l, s, "Cseq")], tf.int32))
				#tMaxMasked = tf.matmul(tMaxTiled, tf.dtypes.cast(Cseq[generateParameterNameSeq(l, s, "Cseq")], tf.int32))
				#tMinMasked3D = tf.reshape(tMinMasked, [n_h[l-1], batchSize, n_h[l]])
				#tMidMasked3D = tf.reshape(tMidMasked, [n_h[l-1], batchSize, n_h[l]])
				#tMaxMasked3D = tf.reshape(tMaxMasked, [n_h[l-1], batchSize, n_h[l]])
				#tMinSeq = tf.math.reduce_min(tMinMasked3D, axis=0)
				#tMidSeq = tf.math.reduce_mean(tMidMasked3D, axis=0)
				#tMaxSeq = tf.math.reduce_max(tMaxMasked3D, axis=0)	
				
				#version 3 (slow: ~0.6s) (slower by a factor of ~x100)
				#
				# tminTiled  _o_  *
				#	n *s   |
				#          |
				#          |
				#          |
				#
				# CTiled    _o__ 
				#    s *n  | 
				#          |
				#          |
				#          |
				#
				multiples = tf.constant([n_h[l],1], tf.int32) 
				tMinTiled = tf.tile(tMin, multiples)
				tMidTiled = tf.tile(tMid, multiples)
				tMaxTiled = tf.tile(tMax, multiples)
				tMinTiled3D = tf.reshape(tMinTiled, [n_h[l], batchSize, n_h[l-1]])
				tMidTiled3D = tf.reshape(tMidTiled, [n_h[l], batchSize, n_h[l-1]])
				tMaxTiled3D = tf.reshape(tMaxTiled, [n_h[l], batchSize, n_h[l-1]])
				Ctransposed = tf.transpose(Cseq[generateParameterNameSeq(l, s, "Cseq")])
				multiples = tf.constant([batchSize,1], tf.int32) 
				CTiled = tf.tile(Ctransposed, multiples)
				CTiled3D = tf.reshape(CTiled, [batchSize, n_h[l], n_h[l-1]])
				CTiled3DaxesAligned = tf.transpose(CTiled3D, [1, 0, 2])
				tMinMasked3D = tf.multiply(tMinTiled3D, tf.dtypes.cast(CTiled3DaxesAligned, tf.int32))	#check conversion
				tMidMasked3D = tf.multiply(tMidTiled3D, tf.dtypes.cast(CTiled3DaxesAligned, tf.int32))
				tMaxMasked3D = tf.multiply(tMaxTiled3D, tf.dtypes.cast(CTiled3DaxesAligned, tf.int32))
				tMinSeq = tf.math.reduce_min(tMinMasked3D, axis=2)
				tMidSeq = tf.math.reduce_mean(tMidMasked3D, axis=2)
				tMaxSeq = tf.math.reduce_max(tMaxMasked3D, axis=2)		
				tMinSeq = tf.reshape(tMinSeq, [n_h[l], batchSize])
				tMidSeq = tf.reshape(tMidSeq, [n_h[l], batchSize])
				tMaxSeq = tf.reshape(tMaxSeq, [n_h[l], batchSize])	
				tMinSeq = tf.transpose(tMinSeq, [1, 0])
				tMidSeq = tf.transpose(tMidSeq, [1, 0])
				tMaxSeq = tf.transpose(tMaxSeq, [1, 0])		
							
			#print("tsLxSx2" + str(tf.timestamp(name="tsLxSx2")))

			if(s == 0):
				Vseq = tf.fill([batchSize, n_h[l]], True)	#all values of Vseq0_l are always set to 1 as they have no sequential dependencies		
			else:
				if(sequentialityMode == "default"):
					#the first sub input of sequential input #2 must fire after the last subinput of sequential input #1
					Vseq = tf.math.greater(tMinSeq, tMaxSeqPrev)
				elif(sequentialityMode == "temporalCrossoverAllowed"):
					#the last sub input of sequential input #1 can fire after the first subinput of sequential input #2
					Vseq = tf.math.greater(tMaxSeq, tMaxSeqPrev)
				elif(sequentialityMode == "contiguousInputEnforced"):
					#the last sub input of sequential input #1 must fire immediately before the first subinput of sequentialInput #2
					Vseq = tf.math.equal(tMinSeq, tMaxSeqPrev+1)	#TODO: verify that the +1 here gets properly broadcasted
				Vseq = tf.math.logical_and(Vseq, VseqPrev)	#if previous sequentiality check fails, then all future sequentiality checks must fail
				
			VseqInt = tf.dtypes.cast(Vseq, tf.int32)
			VseqFloat = tf.dtypes.cast(VseqInt, tf.float32)
			
			if(useSparseTensors):
				Zseq = tf.add(tf.matmul(AprevLayer, tf.sparse.to_dense(Wseq[generateParameterNameSeq(l, s, "Wseq")])), Bseq[generateParameterNameSeq(l, s, "Bseq")])			
			else:
				Zseq = tf.add(tf.matmul(AprevLayer, Wseq[generateParameterNameSeq(l, s, "Wseq")]), Bseq[generateParameterNameSeq(l, s, "Bseq")])			
			Zseq = tf.multiply(VseqFloat, Zseq)	
			Aseq = tf.nn.sigmoid(Zseq)	#or relu
					
			tMidSeqSum = tf.math.add(tMidSeqSum, tMidSeq) 
			AseqSum = tf.math.add(AseqSum, Aseq)
			ZseqSum = tf.math.add(ZseqSum, Zseq)
			
			if(s == 0):
				tMinSeqFirst = tMinSeq
			#if(s == numberOfSequentialInputs-1):
			tMinNext = tMinSeqFirst
			tMidNext = tf.dtypes.cast(tf.math.divide(tMidSeqSum, s+1), tf.int32)	#OLD: AseqSum
			tMaxNext = tMaxSeq

			tMinSeqPrev = tMinSeq
			tMidSeqPrev = tMidSeq
			tMaxSeqPrev = tMaxSeq
			VseqPrev = Vseq
			
			if(s == 0):
				tMinSeqFirst = tMinSeq
				
				
		VseqLastFloat = VseqFloat
		AseqLast = Aseq
		ZseqLast = Zseq
				
		if(sequentialInputCombinationMode1 == 1):
			if(sequentialInputCombinationMode2 == "summation"):
				Q = AseqSum
				Q = tf.multiply(VseqLastFloat, AseqSum)	#supress activation if last sequentialy verification (V) check fails
				Z = ZseqSum
			elif(sequentialInputCombinationMode2 == "useLastSequentialInputOnly"):
				Q = AseqLast
				Z = ZseqLast
			else:
				print("BAD1")
			#Z = tf.add(tf.matmul(Q, W[generateParameterName(l, "W")]), B[generateParameterName(l, "B")])
			#A = tf.nn.sigmoid(Z)	#or relu
			A = Q	
		elif(sequentialInputCombinationMode1 == 2):
			Z = ZseqSum
			A = tf.nn.sigmoid(Z)
		elif(sequentialInputCombinationMode1 == 3):
			Z = AseqSum
			A = tf.nn.sigmoid(Z)

		AprevLayer = A
		
	return tf.nn.softmax(Z)

	

def defineNeuralNetworkParametersSANI():

	randomNormal = tf.initializers.RandomNormal()

	global useSparseTensors
	global sparsityRatioSeq
	global sequentialityMode
	global sequentialInputCombinationMode2
	global sequentialInputCombinationMode1
	
	global n_h
	global numberOfLayers
	global numberOfSequentialInputs
	
	for l in range(1, numberOfLayers+1):
		for s in range(numberOfSequentialInputs):

			#matrices:
			CseqNP = np.random.randint(-sparsityRatioSeq, 1+1, (n_h[l-1], n_h[l]))	#note +1 is required because np.random.randint generates int between min and max-1
			CseqNP = np.maximum(CseqNP, 0)
			CseqNP = np.array(CseqNP, dtype=int)
			CseqNPmask = (CseqNP != 0)
			WseqNP = np.random.rand(n_h[l-1], n_h[l])
			WseqNPsparseValues = WseqNP[CseqNPmask]
			WseqNPsparseIndices = np.asarray(np.where(CseqNPmask))
			WseqNPsparseIndices = np.transpose(WseqNPsparseIndices)
			WseqSparseValues = tf.convert_to_tensor(WseqNPsparseValues, dtype=tf.float32)

			WseqNPsparse = np.multiply(WseqNP, CseqNPmask)

			#sparse tensors:
			if(useSparseTensors):		
				Wseq[generateParameterNameSeq(l, s, "Wseq")] = tf.SparseTensor(indices=WseqNPsparseIndices, values=WseqSparseValues, dense_shape=[n_h[l-1], n_h[l]])	#, dtype=tf.int32
			else:
				Wseq[generateParameterNameSeq(l, s, "Wseq")] = tf.Variable(WseqNPsparse, dtype=tf.float32)		#tf.SparseTensor(indices=WseqNPsparseIndices, values=WseqSparseValues, dense_shape=[n_h[l-1], n_h[l]])
			Cseq[generateParameterNameSeq(l, s, "Cseq")] = tf.Variable(CseqNPmask, dtype=tf.bool)	#ORIG: tf.convert_to_tensor(CseqNPmask, dtype=tf.bool)
			Bseq[generateParameterNameSeq(l, s, "Bseq")] = tf.Variable(tf.zeros(n_h[l]), dtype=tf.float32)

		#matrices:
		#W[generateParameterName(l, "W")] = tf.Variable(randomNormal([numberOfSequentialInputs, n_h[l]], dtype=tf.float32))	#randomNormal	#note the architecture of this weight matrix is different than a normal weight matrix. Every set of numberOfSequentialInputs (e.g. 3) represents a unique set of sequential neuron inputs (artificial neurons), and are mapped to an independent neuron (real neuron)
		#B[generateParameterName(l, "B")] = tf.Variable(tf.zeros(n_h[l]), tf.float32)

				

