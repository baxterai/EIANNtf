"""ANNtf2_algorithmLIANN_PCAsimulation.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2021 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNtf2.py

# Usage:
see ANNtf2.py

# Description:
ANNtf algorithm LIANN PCA simulation

"""

import tensorflow as tf
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd

SVDmatrixTypeExamplesByNeurons = True
SVDmatrixTypeSpikeCoincidence = False

#uses np not tf;
def generateSVDinputMatrix(l, n_h, AprevLayer):
	if(SVDmatrixTypeExamplesByNeurons):
		SVDinputMatrix = AprevLayer.numpy()
	elif(SVDmatrixTypeSpikeCoincidence):
		SVDinputMatrix = generateSpikeCoincidenceMatrix(l, n_h, AprevLayer)
	return SVDinputMatrix
	
def generateSpikeCoincidenceMatrix(l, n_h, AprevLayer):
	#creates co-occurence count matrix of dimensionality (num_neurons, num_neurons)
	layerSize = n_h[l-1]	#or AprevLayer.shape[1]
	batchSize = AprevLayer.shape[0]
	spikeCoincidenceMatrix = np.empty([layerSize, layerSize])
	AprevLayerNP = AprevLayer.numpy()	#perform operations in numpy to save dev time
	for k1 in range(layerSize):
		for k2 in range(layerSize):
			spikeCoincidences = np.logical_and(AprevLayerNP[:,k1], AprevLayerNP[:,k2])
			spikeCoincidenceMatrix[k1, k2] = np.sum(spikeCoincidences)
	print("spikeCoincidenceMatrix = ", spikeCoincidenceMatrix)
	return spikeCoincidenceMatrix

def calculateSVD(M, k):		
	#print("M = ", M)		
	U, Sigma, VT = randomized_svd(M, n_components=k, n_iter=5, random_state=None)	#n_iter = ?
	return U, Sigma, VT

def calculateWeights(l, n_h, SVDinputMatrix, U, Sigma, VT):
	#incomplete
	#SVDinputMatrix: batchSize*prevLayerSize
	#U: batchSize*layerSize
	SVDinputMatrixT = np.transpose(SVDinputMatrix)
	layerSize = n_h[l]
	prevLayerSize = n_h[l-1]
	AWnumpy = np.matmul(SVDinputMatrixT, U)  	#incomplete: some function of SVDinputMatrix, U, Sigma, VT	#dimensions: prevLayerSize*layerSize
	AW = tf.convert_to_tensor(AWnumpy)
	return AW

