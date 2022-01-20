"""ANNtf2_algorithmLIANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNtf2.py

# Usage:
see ANNtf2.py

# Description:
ANNtf algorithm LIANN - define local inhibition artificial neural network (force neural independence)

Emulates unsupervised singular value decomposition (SVD/factor analysis) learning for all hidden layers

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf2_operations
import ANNtf2_globalDefs
import ANNtf2_algorithmLIANN_math
import copy

#debug parameters;
debugFastTrain = False
debugSmallBatchSize = False	#small batch size for debugging matrix output

#select learningAlgorithm (unsupervised learning algorithm for intermediate layers):
learningAlgorithmNone = True	#create a very large network (eg x10) neurons per layer, and perform final layer backprop only
learningAlgorithmCorrelation = False	#minimise correlation between layer neurons	#create a very large network (eg x10) neurons per layer, remove/reinitialise neurons that are highly correlated (redundant/not necessary to end performance), and perform final layer backprop only
learningAlgorithmPCAsimulation = False	#note layer construction is nonlinear (use ANNtf2_algorithmAEANN/autoencoder for nonlinear dimensionality reduction simulation)	#incomplete
learningAlgorithmStochasticCorrelation = False	#stochastic optimise weights based on objective function; minimise the correlation between layer neurons
learningAlgorithmShuffle = False	#randomly shuffle neuron weights until independence is detected
learningAlgorithmStochasticMaximiseAndEvenSignal = False	#stochastic optimise weights based on objective functions; #1: maximise the signal (ie successfully uninhibited) across multiple batches (entire dataset), #2: ensure that all layer neurons receive even activation across multiple batches (entire dataset)	
learningAlgorithmShufflePermanence = False	#increase the permanence of uninhibited neuron weights, and stocastically modify weights based on their impermanence
learningAlgorithmHebbian = False	#strengthen weights of successfully activated neurons

#intialise network properties (configurable);
positiveExcitatoryWeights = False	#requires testing	#required for biological plausibility of most learningAlgorithms
positiveExcitatoryWeightsActivationFunctionOffsetDisable = False	
supportSkipLayers = False #fully connected skip layer network	#TODO: add support for skip layers	#see ANNtf2_algorithmFBANN for template

supportMultipleNetworks = True

#intialise network properties;
largeBatchSize = False
generateLargeNetwork = False	#large number of layer neurons is required for learningAlgorithmHebbian:useZAcoincidenceMatrix
generateNetworkStatic = False

#select learningAlgorithmFinalLayer (supervised learning algorithm for final layer/testing):
learningAlgorithmFinalLayerBackpropHebbian = True	#only apply backprop (effective hebbian) learning at final layer

learningAlgorithmStochastic = False
if(learningAlgorithmStochasticCorrelation):
	learningAlgorithmStochastic = True
elif(learningAlgorithmStochasticMaximiseAndEvenSignal):
	learningAlgorithmStochastic = True

#network/activation parameters;
#forward excitatory connections;
W = {}
B = {}
if(supportMultipleNetworks):
	WallNetworksFinalLayer = None
	BallNetworksFinalLayer = None
if(learningAlgorithmStochastic):
	Wbackup = {}
	Bbackup = {}
useBinaryWeights = False

if(learningAlgorithmFinalLayerBackpropHebbian):
	positiveExcitatoryWeightsFinalLayer = False	#allow negative weights on final layer to emulate standard backprop/hebbian learning

estNetworkActivationSparsity = 0.5	#50% of neurons are expected to be active during standard propagation (no inhibition)

#intialise algorithm specific parameters;
inhibitionAlgorithmArtificial = False	#simplified inhibition algorithm implementation
inhibitionAlgorithmArtificialMoreThanXLateralNeuronActive = False	#inhibit layer if more than x lateral neurons active
inhibitionAlgorithmArtificialSparsity = False	#inhibition signal increases with number of simultaneously active neurons

Athreshold = False
AthresholdValue = 1.0	#do not allow output signal to exceed 1.0

learningRate = 0.0	#defined by defineTrainingParametersLIANN

generateDeepNetwork = False
generateVeryLargeNetwork = False

#intialise algorithm specific parameters;
enableInhibitionTrainSpecificLayerOnly = False
applyInhibitoryNetworkDuringTest = False
randomlyActivateWeightsDuringTrain = False

#learning algorithm customisation;
supportDimensionalityReductionLimitFrequency = False
if(learningAlgorithmNone):
	#note learningAlgorithmCorrelation requires supportSkipLayers - see LIANNtf_algorithmIndependentInput/AEANNtf_algorithmIndependentInput:learningAlgorithmLIANN for similar implementation
	#positiveExcitatoryWeights = True	#optional
	generateDeepNetwork = True	#optional	#used for algorithm testing
	generateVeryLargeNetwork = True
	generateNetworkStatic = True
elif(learningAlgorithmCorrelation):
	#note learningAlgorithmCorrelation requires supportSkipLayers - see LIANNtf_algorithmIndependentInput/AEANNtf_algorithmIndependentInput:learningAlgorithmLIANN for similar implementation
	#positiveExcitatoryWeights = True	#optional	
	enableInhibitionTrainSpecificLayerOnly = True	#optional?
	supportDimensionalityReduction = True	#mandatory
	if(supportDimensionalityReduction):
		supportDimensionalityReductionRandomise = True
		maxCorrelation = 0.95
		supportDimensionalityReductionLimitFrequency = True
		if(supportDimensionalityReductionLimitFrequency):
			supportDimensionalityReductionLimitFrequencyStep = 1000
	generateDeepNetwork = True	#optional	#used for algorithm testing
	generateVeryLargeNetwork = True
	generateNetworkStatic = True
elif(learningAlgorithmPCAsimulation):
	#positiveExcitatoryWeights = True	#optional
	enableInhibitionTrainSpecificLayerOnly = True	#optional?
	largeBatchSize = True	#1 PCA is performed across entire dataset [per layer]
elif(learningAlgorithmShuffle):
	#positiveExcitatoryWeights = True	#optional
	inhibitionAlgorithmArtificialMoreThanXLateralNeuronActive = True	#mandatory
	enableInhibitionTrainSpecificLayerOnly = True
	fractionIndependentInstancesAcrossBatchRequired = 0.3	#divide by number of neurons on layer	#if found x% of independent instances, then record neuron as independent (solidify weights)	#FUTURE: will depend on number of neurons on current layer and previous layer	#CHECKTHIS: requires calibration
	largeBatchSize = True
elif(learningAlgorithmStochastic):
	inhibitionAlgorithmArtificialMoreThanXLateralNeuronActive = True	#optional
	if(learningAlgorithmStochasticCorrelation):
		learningAlgorithmStochasticAlgorithm = "correlation"
		#positiveExcitatoryWeights = True	#optional
		#learning objective function: minimise the correlation between layer neurons
	elif(learningAlgorithmStochasticMaximiseAndEvenSignal):
		learningAlgorithmStochasticAlgorithm = "maximiseAndEvenSignal"
		#positiveExcitatoryWeights = True	#optional?
		#learning objective functions:
			#1: maximise the signal (ie successfully uninhibited) across multiple batches (entire dataset)
			#2: ensure that all layer neurons receive even activation across multiple batches (entire dataset)		
		metric1Weighting = 1.0
		metric2Weighting = 1000.0	#normalise metric2Weighting relative to metric1Weighting; eg metric1 =  0.9575, metric2 =  0.000863842
	enableInhibitionTrainSpecificLayerOnly = True
	numberStochasticIterations = 10
	updateParameterSubsetSimultaneously = False	#current tests indiciate this is not required/beneficial with significantly high batch size
	if(updateParameterSubsetSimultaneously):
		numberOfSubsetsTrialledPerBaseParameter = 10	#decreases speed, but provides more robust parameter updates
		parameterUpdateSubsetSize = 5	#decreases speed, but provides more robust parameter updates
	else:
		numberOfSubsetsTrialledPerBaseParameter = 1
		parameterUpdateSubsetSize = 1	
	NETWORK_PARAM_INDEX_TYPE = 0
	NETWORK_PARAM_INDEX_LAYER = 1
	NETWORK_PARAM_INDEX_H_CURRENT_LAYER = 2
	NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER = 3
	NETWORK_PARAM_INDEX_VARIATION_DIRECTION = 4
elif(learningAlgorithmShufflePermanence):
	inhibitionAlgorithmArtificialMoreThanXLateralNeuronActive = True	#optional
	#positiveExcitatoryWeights = True	#optional?
	enableInhibitionTrainSpecificLayerOnly = False	#CHECKTHIS (set True)
	applyInhibitoryNetworkDuringTest = True	#CHECKTHIS (set False)
	Wpermanence = {}
	Bpermanence = {}
	WpermanenceInitial = 0.1
	BpermanenceInitial = 0.1
	WpermanenceUpdateRate = 0.1
	BpermanenceUpdateRate = 0.1
	permanenceNumberBatches = 10	#if permanenceUpdateRate=1, average number of batches to reset W to random values
	solidificationRate = 0.1
elif(learningAlgorithmHebbian):
	tuneInhibitionNeurons = False	#optional
	useZAcoincidenceMatrix = True	#reduce connection weights for unassociated neurons
	positiveExcitatoryWeights = True	#mandatory (requires testing)
	positiveExcitatoryWeightsThresholds = True	#do not allow weights to exceed 1.0 / fall below 0.0 [CHECKTHIS]
	Athreshold = True	#prevents incremental increase in signal per layer
	alwaysApplyInhibition = False	
	if(useZAcoincidenceMatrix):
		alwaysApplyInhibition = True	#inhibition is theoretically allowed at all times with useZAcoincidenceMatrix as it simply biases the network against a correlation between layer k neurons (inhibition is not set up to only allow X/1 neuron to fire)
	if(alwaysApplyInhibition):
		#TODO: note network sparsity/inhibition must be configured such that at least one neuron fires per layer
		positiveExcitatoryWeightsActivationFunctionOffsetDisable = True	#activation function will always be applied to Z signal comprising positive+negative components	#CHECKTHIS
		inhibitionAlgorithmArtificialSparsity = True
		generateLargeNetwork = True 	#large is required because it will be sparsely activated due to constant inhibition
		generateNetworkStatic = True	#equal number neurons per layer for unsupervised layers/testing
		enableInhibitionTrainSpecificLayerOnly = False
		applyInhibitoryNetworkDuringTest = True
	else:
		inhibitionAlgorithmArtificialMoreThanXLateralNeuronActive = True	#optional
		enableInhibitionTrainSpecificLayerOnly = True
		applyInhibitoryNetworkDuringTest = False
	randomlyActivateWeightsDuringTrain = False	#randomly activate x weights (simulating input at simulataneous time interval t)
	if(randomlyActivateWeightsDuringTrain):
		randomlyActivateWeightsProbability = 1.0
	WinitialisationFactor = 1.0	#initialise network with relatively low weights	#network will be trained (weights will be increased) up until point where activation inhibited
	BinitialisationFactor = 1.0	#NOTUSED
	weightDecay = False
	if(useZAcoincidenceMatrix):
		useZAcoincidenceMatrix = True	#reduce connection weights for unassociated neurons
		if(useZAcoincidenceMatrix):
			#inhibitionAlgorithmArtificialMoreThanXLateralNeuronActive = False	#useZAcoincidenceMatrix requires real negative weights
			normaliseWeightUpdates = False
		else:
			normaliseWeightUpdates = True	#unimplemented: strengthen/update weights up to some maxima as determined by input signal strength (prevent runaway increase in weight strength up to 1.0)
	else:
		weightDecay = True	#constant neural net weight decay, such that network can be continuously trained
		weightDecayRate = 0.0	#defined by defineTrainingParametersLIANN		
		useZAcoincidenceMatrix = False
		normaliseWeightUpdates = False
		
	maxWeightUpdateThreshold = False	#max threshold weight updates to learningRate	
	#TODO: ensure learning algorithm does not result in runnaway weight increases


positiveExcitatoryWeightsActivationFunctionOffset = False
if(positiveExcitatoryWeights):
	if(positiveExcitatoryWeightsActivationFunctionOffsetDisable):
		positiveExcitatoryWeightsActivationFunctionOffset = False
	else:
		positiveExcitatoryWeightsActivationFunctionOffset = True
	normaliseInput = False	#TODO: verify that the normalisation operation will not disort the code's capacity to process a new data batch the same as an old data batch
	normalisedAverageInput = 1.0	#normalise input signal	#arbitrary
	if(positiveExcitatoryWeightsActivationFunctionOffset):
		positiveExcitatoryThreshold = 0.5	#1.0	#weights are centred around positiveExcitatoryThreshold, from 0.0 to positiveExcitatoryThreshold*2	#arbitrary
	Wmean = 0.5	#arbitrary
	WstdDev = 0.05	#stddev of weight initialisations	#CHECKTHIS
else:
	normaliseInput = False
	Wmean = 0.0
	WstdDev = 0.05	#stddev of weight initialisations
	

if(inhibitionAlgorithmArtificialMoreThanXLateralNeuronActive or inhibitionAlgorithmArtificialSparsity):
	inhibitionAlgorithmArtificial = True
	
if(inhibitionAlgorithmArtificial):
	if(inhibitionAlgorithmArtificialMoreThanXLateralNeuronActive):
		inhibitionAlgorithmMoreThanXLateralNeuronActiveFraction = True
		if(inhibitionAlgorithmMoreThanXLateralNeuronActiveFraction):
			inhibitionAlgorithmMoreThanXLateralNeuronActiveFractionValue = 0.25	#fraction of the layer active allowed before inhibition
		else:
			inhibitionAlgorithmMoreThanXLateralNeuronActiveValue = 1	#maximum (X) number of neurons activate allowed before inhibition
else:
	inhibitionFactor1 = 1.0	#pass through signal	#positiveExcitatoryThreshold	#CHECKTHIS: requires recalibration for activationFunction:positiveExcitatoryWeights
	inhibitionFactor2 = estNetworkActivationSparsity	#-(WstdDev)	#inhibition contributes a significant (nullifying) effect on layer activation	#CHECKTHIS: requires calibration
	if(randomlyActivateWeightsDuringTrain):
		inhibitionFactor1 = inhibitionFactor1
		inhibitionFactor2 = (inhibitionFactor2*randomlyActivateWeightsProbability)	#the lower the average activation, the lower the inhibition
	#TODO: inhibitionFactor1/inhibitionFactor2 requires recalibration for activationFunction:positiveExcitatoryWeights
	singleInhibitoryNeuronPerLayer = False	#simplified inhibitory layer
	#lateral inhibitory connections (incoming/outgoing);
	IWi = {}
	IBi = {}
	IWo = {}
	IWiWeights = inhibitionFactor1	#need at least 1/IWiWeights active neurons per layer for the inhibitory neuron to become activated	#CHECKTHIS: requires calibration	#WstdDev*2	#0.5	#0.3
	IWoWeights = inhibitionFactor2	#will depend on activation sparsity of network (higher the positive activation, higher the inhibition required)	#requires calibration such that more than x (e.g. 1) active neuron on a layer will inhibit the layer
	In_h = []


if(learningAlgorithmShuffle):
	Bindependent = {}	#independent neurons previously identified	#effective boolean (0.0 or 1.0)	#FUTURE: consider making this a continuous variable, such that the higher the independence the less the variable is randomly shuffled per training iteration
	
#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0


#note high batchSize is required for learningAlgorithmStochastic algorithm objective functions (>= 100)
def defineTrainingParameters(dataset):
	global learningRate
	global weightDecayRate	
	
	if(learningAlgorithmStochastic):
		learningRate = 0.001
	elif(learningAlgorithmHebbian):
		learningRate = 0.001
		weightDecayRate = learningRate/10.0	#CHECKTHIS	#will depend on learningRate
	else:
		learningRate = 0.005
	
	if(debugSmallBatchSize):
		batchSize = 10
	else:
		if(largeBatchSize):
			batchSize = 1000	#current implementation: batch size should contain all examples in training set
		else:
			batchSize = 100	#3	#100
	if(generateDeepNetwork):
		numEpochs = 100	#higher num epochs required for convergence
	else:
		numEpochs = 10	#100 #10
	if(debugFastTrain):
		trainingSteps = batchSize
	else:
		trainingSteps = 10000	#1000
		
	displayStep = 100
			
	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	


def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks
	if(not inhibitionAlgorithmArtificial):
		global In_h

	if(generateVeryLargeNetwork):
		firstHiddenLayerNumberNeurons = num_input_neurons*10
	else:
		if(generateLargeNetwork):
			firstHiddenLayerNumberNeurons = num_input_neurons*3
		else:
			firstHiddenLayerNumberNeurons = num_input_neurons
	if(generateDeepNetwork):
		numberOfLayers = 3
	else:
		numberOfLayers = 2
	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = defineNetworkParametersDynamic(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet, numberOfLayers, firstHiddenLayerNumberNeurons, generateNetworkStatic)		
	#n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf2_operations.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet, generateLargeNetwork=generateLargeNetwork, generateNetworkStatic=generateNetworkStatic)
	
	if(not inhibitionAlgorithmArtificial):
		if(singleInhibitoryNeuronPerLayer):
			In_h = [1] * len(n_h)	#create one inhibitory neuron per layer
		else:
			In_h = copy.copy(n_h)	#create one inhibitory neuron for every excitatory neuron
		
	return numberOfLayers
	

def defineNeuralNetworkParameters():

	print("numberOfNetworks", numberOfNetworks)
	
	global randomNormal
	randomNormal = tf.initializers.RandomNormal(mean=Wmean, stddev=WstdDev)
	#randomNormal = tf.initializers.RandomNormal()
	randomNormalFinalLayer = tf.initializers.RandomNormal()
	
	for networkIndex in range(1, numberOfNetworks+1):
		for l in range(1, numberOfLayers+1):
			#forward excitatory connections;
			EWlayer = randomNormal([n_h[l-1], n_h[l]]) 
			EBlayer = tf.zeros(n_h[l])
			if(positiveExcitatoryWeights):
				EWlayer = tf.abs(EWlayer)	#ensure randomNormal generated weights are positive
				if((l == numberOfLayers) and not positiveExcitatoryWeightsFinalLayer):
					EWlayer = randomNormalFinalLayer([n_h[l-1], n_h[l]])
			if(learningAlgorithmHebbian):
				EWlayer = tf.multiply(EWlayer, WinitialisationFactor)
				EBlayer = tf.multiply(EBlayer, BinitialisationFactor)
			W[generateParameterNameNetwork(networkIndex, l, "W")] = tf.Variable(EWlayer)
			B[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(EBlayer)
	
			if(learningAlgorithmShuffle):
				Bindependent[generateParameterNameNetwork(networkIndex, l, "Bindependent")] = tf.Variable(EBlayer)		#initialise all neurons to zero (false)
			elif(learningAlgorithmStochastic):
				Wbackup[generateParameterNameNetwork(networkIndex, l, "W")] = tf.Variable(W[generateParameterNameNetwork(networkIndex, l, "W")])
				Bbackup[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(B[generateParameterNameNetwork(networkIndex, l, "B")])			
			elif(learningAlgorithmShufflePermanence):
				EWlayerPermanence = tf.multiply(tf.ones([n_h[l-1], n_h[l]]), WpermanenceInitial)
				EBlayerPermanence = tf.multiply(tf.ones(n_h[l]), BpermanenceInitial)
				Wpermanence[generateParameterNameNetwork(networkIndex, l, "Wpermanence")] = tf.Variable(EWlayerPermanence)
				Bpermanence[generateParameterNameNetwork(networkIndex, l, "Bpermanence")] = tf.Variable(EBlayerPermanence)
			
			if(not inhibitionAlgorithmArtificial):			
				#lateral inhibitory connections (incoming/outgoing);
				#do not currently train inhibitory weights;
				IWilayer = tf.multiply(tf.ones([n_h[l], In_h[l]]), IWiWeights)		#CHECKTHIS: inhibitory neuron firing is a function of current (lateral) layer (not previous layer)
				IBilayer = tf.zeros(In_h[l])
				if(singleInhibitoryNeuronPerLayer):
					IWoWeightsL = IWoWeights
				else:
					IWoWeightsL = IWoWeights/In_h[l]	#normalise across number inhibitory neurons
				IWolayer = tf.multiply(tf.ones([In_h[l], n_h[l]]), IWoWeightsL)
				IWi[generateParameterNameNetwork(networkIndex, l, "IWi")] = tf.Variable(IWilayer)
				IBi[generateParameterNameNetwork(networkIndex, l, "IBi")] = tf.Variable(IBilayer)
				IWo[generateParameterNameNetwork(networkIndex, l, "IWo")] = tf.Variable(IWolayer)

	if(supportMultipleNetworks):
		if(numberOfNetworks > 1):
			global WallNetworksFinalLayer
			global BallNetworksFinalLayer
			WlayerF = randomNormal([n_h[numberOfLayers-1]*numberOfNetworks, n_h[numberOfLayers]])
			WallNetworksFinalLayer =  tf.Variable(WlayerF)
			Blayer = tf.zeros(n_h[numberOfLayers])
			BallNetworksFinalLayer	= tf.Variable(Blayer)	#not currently used
					
def neuralNetworkPropagation(x, networkIndex=1):
	return neuralNetworkPropagationLIANNtest(x, networkIndex)

def neuralNetworkPropagationLIANNtest(x, networkIndex=1, l=None):
	return neuralNetworkPropagationLIANNminimal(x, networkIndex, l)

def neuralNetworkPropagationLayer(x, networkIndex=1, l=None):
	return neuralNetworkPropagationLIANNminimal(x, networkIndex, l)
	#return neuralNetworkPropagationLIANN(x, None, networkIndex, trainWeights=False)

def neuralNetworkPropagationLIANNtrain(x, networkIndex=1):
	if(enableInhibitionTrainSpecificLayerOnly):
		for l in range(1, numberOfLayers+1):
			if(l < numberOfLayers):
				return neuralNetworkPropagationLIANN(x, networkIndex, trainWeights=True, layerToTrain=l)
	else:
		return neuralNetworkPropagationLIANN(x, networkIndex, trainWeights=True, layerToTrain=None)	

#if(supportMultipleNetworks):
def neuralNetworkPropagationAllNetworksFinalLayer(AprevLayer):
	Z = tf.add(tf.matmul(AprevLayer, WallNetworksFinalLayer), BallNetworksFinalLayer)	
	#Z = tf.matmul(AprevLayer, WallNetworksFinalLayer)	
	pred = tf.nn.softmax(Z)	
	return pred
	
	
#minimal code extracted from neuralNetworkPropagationLIANN;
def neuralNetworkPropagationLIANNminimal(x, networkIndex=1, l=None):

	enableInhibition = False
	randomlyActivateWeights = False

	if(l == None):
		maxLayer = numberOfLayers
	else:
		maxLayer = l
		
	AprevLayer = x
	ZprevLayer = x
	for l in range(1, maxLayer+1):
					
		A, Z, _ = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)

		if(learningAlgorithmFinalLayerBackpropHebbian):
			A = tf.stop_gradient(A)

		AprevLayer = A
		ZprevLayer = Z

	if(maxLayer == numberOfLayers):
		return tf.nn.softmax(Z)
	else:
		return A
	
def neuralNetworkPropagationLIANN(x, networkIndex=1, trainWeights=False, layerToTrain=None):

	if(normaliseInput):
		#TODO: verify that the normalisation operation will not disort the code's capacity to process a new data batch the same as an old data batch
		averageTotalInput = tf.math.reduce_mean(x)
		#print("averageTotalInput = ", averageTotalInput)
		x = tf.multiply(x, normalisedAverageInput/averageTotalInput)	#normalise input wrt positiveExcitatoryThreshold
		#averageTotalInput = tf.math.reduce_mean(x)

	if(trainWeights):
		if(enableInhibitionTrainSpecificLayerOnly):
			maxLayer = layerToTrain
		else:
			maxLayer = numberOfLayers
	else:
		maxLayer = numberOfLayers
			
	AprevLayer = x
	ZprevLayer = x
	for l in range(1, maxLayer+1):
					
		trainLayer = False
		enableInhibition = False
		randomlyActivateWeights = False
		if(trainWeights):
			if(enableInhibitionTrainSpecificLayerOnly):
				if(l == layerToTrain):
					#enableInhibition = False
					enableInhibition = True
					trainLayer = True
			else:
				if(l < numberOfLayers):
					enableInhibition = True
					trainLayer = True
			if(randomlyActivateWeightsDuringTrain):
				randomlyActivateWeights = True
		else:
			if(applyInhibitoryNetworkDuringTest):
				enableInhibition = True
	
		finalLayerHebbian = False	
		if(learningAlgorithmFinalLayerBackpropHebbian):
			if(l == numberOfLayers):
				finalLayerHebbian = True			
		if(finalLayerHebbian):
			enableInhibition = False
				
		
		if(finalLayerHebbian):
			A, Z, _ = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)
		else:
			if(trainLayer):
				#CHECKTHIS: verify learning algorithm (how to modify weights to maximise independence between neurons on each layer)

				if(learningAlgorithmCorrelation):
					neuralNetworkPropagationLIANNlearningAlgorithmCorrelation(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)
				elif(learningAlgorithmPCAsimulation):
					neuralNetworkPropagationLIANNlearningAlgorithmPCAsimulation(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)
				elif(learningAlgorithmShuffle):
					neuralNetworkPropagationLIANNlearningAlgorithmShuffle(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)
				elif(learningAlgorithmStochastic):
					neuralNetworkPropagationLIANNlearningAlgorithmStochastic(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)
				elif(learningAlgorithmShufflePermanence):
					neuralNetworkPropagationLIANNlearningAlgorithmShufflePermanence(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)
				elif(learningAlgorithmHebbian):
					neuralNetworkPropagationLIANNlearningAlgorithmHebbian(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)

				A, Z, _ = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition=(not enableInhibitionTrainSpecificLayerOnly), randomlyActivateWeights=False)	#in case !learningAlgorithmFinalLayerBackpropHebbian
			else:
				A, Z, _ = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)

			if(learningAlgorithmFinalLayerBackpropHebbian):
				A = tf.stop_gradient(A)
				
			AprevLayer = A
			ZprevLayer = Z

	return tf.nn.softmax(Z)

def neuralNetworkPropagationLIANNlearningAlgorithmCorrelation(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights):

	A, Z, _ = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l)
	
	#measure and minimise correlation between layer neurons;
	ANNtf2_algorithmLIANN_math.neuronActivationCorrelationMinimisation(networkIndex, n_h, l, A, randomNormal, Wf=W, Wfname="W", Wb=None, Wbname=None, updateAutoencoderBackwardsWeights=False, supportSkipLayers=supportSkipLayers, supportDimensionalityReductionRandomise=supportDimensionalityReductionRandomise, maxCorrelation=maxCorrelation)


def neuralNetworkPropagationLIANNlearningAlgorithmPCAsimulation(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights):
	#Afinal, Zfinal, _ = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)	#batched
	SVDinputMatrix = ANNtf2_algorithmLIANN_math.generateSVDinputMatrix(l, n_h, AprevLayer)
	U, Sigma, VT = ANNtf2_algorithmLIANN_math.calculateSVD(M=SVDinputMatrix, k=n_h[l])
	AW = ANNtf2_algorithmLIANN_math.calculateWeights(l, n_h, SVDinputMatrix, U, Sigma, VT)
	W[generateParameterNameNetwork(networkIndex, l, "W")] = AW

	#weights = U -> Sigma -> VT	[linear]
	#M_reduced = reduce_to_k_dim(M=spikeCoincidenceMatrix, k=n_h[l])
	
def neuralNetworkPropagationLIANNlearningAlgorithmShuffle(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights):
	layerHasDependentNeurons = True
	Bind = Bindependent[generateParameterNameNetwork(networkIndex, l, "Bindependent")]
	if(count_zero(Bind) > 0):	#more than 1 dependent neuron on layer
		layerHasDependentNeurons = True
	else:
		layerHasDependentNeurons = False

	while(layerHasDependentNeurons):
		Afinal, Zfinal, _ = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)	#batched

		AnumActive = tf.math.count_nonzero(Afinal, axis=1)	#batched
		Aindependent = tf.equal(AnumActive, 1)	#batched
		Aindependent = tf.dtypes.cast(Aindependent, dtype=tf.dtypes.float32)	#batched
		Aindependent = tf.expand_dims(Aindependent, 1)	#batched
		#print("Afinal = ", Afinal)
		#print("AnumActive = ", AnumActive)
		#print("Aindependent = ", Aindependent)

		Aactive = tf.greater(Afinal, 0)	#2D: batched, for every k neuron
		Aactive = tf.dtypes.cast(Aactive, dtype=tf.dtypes.float32) 	#2D: batched, for every k neuron
		#print("Aactive = ", Aactive)
		#ex

		AactiveAndIndependent = tf.multiply(Aactive, Aindependent)	#2D: batched, for every k neuron	
		AactiveAndIndependent = tf.reduce_sum(AactiveAndIndependent, axis=0) #for every k neuron

		AactiveAndIndependentPass = tf.greater(AactiveAndIndependent, fractionIndependentInstancesAcrossBatchRequired*n_h[l])	 #for every k neuron
		#print("AactiveAndIndependentPass = ", AactiveAndIndependentPass)

		BindBool = tf.dtypes.cast(Bind, dtype=tf.dtypes.bool)
		AactiveAndIndependentPassRequiresSolidifying = tf.logical_and(AactiveAndIndependentPass, tf.logical_not(BindBool))
		#print("AactiveAndIndependentPass = ", AactiveAndIndependentPass)
		#print("BindBool = ", BindBool)
		print("AactiveAndIndependentPassRequiresSolidifying = ", AactiveAndIndependentPassRequiresSolidifying)
		BindNew = tf.logical_or(BindBool, AactiveAndIndependentPassRequiresSolidifying)
		BdepNew = tf.logical_not(BindNew)

		#update layer weights (reinitialise weights for all dependent neurons);
		BindNew = tf.dtypes.cast(BindNew, dtype=tf.dtypes.float32)
		BdepNew = tf.dtypes.cast(BdepNew, dtype=tf.dtypes.float32)
		EWlayerDep = randomNormal([n_h[l-1], n_h[l]]) 
		if(positiveExcitatoryWeights):
			EWlayerDep = tf.abs(EWlayerDep)	#ensure randomNormal generated weights are positive
		EBlayerDep = tf.zeros(n_h[l])
		EWlayerDep = tf.multiply(EWlayerDep, BdepNew)	#requires broadcasting
		EBlayerDep = tf.multiply(EBlayerDep, BdepNew)				
		EWlayerInd = W[generateParameterNameNetwork(networkIndex, l, "W")] 
		EBlayerInd = B[generateParameterNameNetwork(networkIndex, l, "B")]
		EWlayerInd = tf.multiply(EWlayerInd, BindNew)	#requires broadcasting
		EBlayerInd = tf.multiply(EBlayerInd, BindNew)
		EWlayerNew = tf.add(EWlayerDep, EWlayerInd)
		EBlayerNew = tf.add(EBlayerDep, EBlayerInd)					
		W[generateParameterNameNetwork(networkIndex, l, "W")] = EWlayerNew
		B[generateParameterNameNetwork(networkIndex, l, "B")] = EBlayerNew	
		#print("EWlayerNew = ", EWlayerNew)				

		#print("BdepNew = ", BdepNew)
		#print("BindNew = ", BindNew)

		Bindependent[generateParameterNameNetwork(networkIndex, l, "Bindependent")] = BindNew	#update independence record
		Bind = BindNew
		if(count_zero(Bind) > 0):	#more than 1 dependent neuron on layer
			layerHasDependentNeurons = True
			#print("layerHasDependentNeurons: count_zero(Bind) = ", count_zero(Bind))
		else:
			layerHasDependentNeurons = False	
			#print("!layerHasDependentNeurons")
							
def neuralNetworkPropagationLIANNlearningAlgorithmStochastic(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights):

	if(learningAlgorithmStochastic):
		if(useBinaryWeights):
			variationDirections = 1
		else:
			variationDirections = 2
			
	#code from ANNtf2_algorithmLREANN_expSUANN;
	for s in range(numberStochasticIterations):
		for hIndexCurrentLayer in range(0, n_h[l]):
			for hIndexPreviousLayer in range(0, n_h[l-1]+1):
				if(hIndexPreviousLayer == n_h[l-1]):	#ensure that B parameter updates occur/tested less frequently than W parameter updates
					parameterTypeWorB = 0
				else:
					parameterTypeWorB = 1
				for variationDirectionInt in range(variationDirections):

					networkParameterIndexBase = (parameterTypeWorB, l, hIndexCurrentLayer, hIndexPreviousLayer, variationDirectionInt)

					metricBase = learningAlgorithmStochasticCalculateMetric(networkIndex, AprevLayer, ZprevLayer, l)

					for subsetTrialIndex in range(0, numberOfSubsetsTrialledPerBaseParameter):

						accuracyImprovementDetected = False

						currentSubsetOfParameters = []
						currentSubsetOfParameters.append(networkParameterIndexBase)

						for s in range(1, parameterUpdateSubsetSize):
							networkParameterIndex = getRandomNetworkParameter(networkIndex, currentSubsetOfParameters)
							currentSubsetOfParameters.append(networkParameterIndex)

						for s in range(0, parameterUpdateSubsetSize):
							networkParameterIndex = currentSubsetOfParameters[s]

							if(not useBinaryWeights):
								if(networkParameterIndex[NETWORK_PARAM_INDEX_VARIATION_DIRECTION] == 1):
									variationDiff = learningRate
								else:
									variationDiff = -learningRate		

							if(networkParameterIndex[NETWORK_PARAM_INDEX_TYPE] == 1):
								#Wnp = W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")].numpy()
								#currentVal = Wnp[networkParameterIndex[NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER], networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]]
								currentVal = W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")][networkParameterIndex[NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER], networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].numpy()

								#print("currentVal = ", currentVal)
								#print("W1 = ", W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")])
								if(useBinaryWeights):
									if(useBinaryWeightsReduceMemoryWithBool):
										newVal = not currentVal
									else:
										newVal = float(not bool(currentVal))
										#print("newVal = ", newVal)
								else:
									newVal = currentVal + variationDiff

								if(positiveExcitatoryWeights):
									newVal = max(newVal, 0)	#do not allow weights fall below zero [CHECKTHIS]	

								W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")][networkParameterIndex[NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER], networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].assign(newVal)

								#print("W2 = ", W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")])
							else:
								#Bnp = B[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "B")].numpy()
								#currentVal = Bnp[networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]]
								currentVal = B[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "B")][networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].numpy()

								if(useBinaryWeights):
									if(useBinaryWeightsReduceMemoryWithBool):
										newVal = not currentVal
									else:
										newVal = float(not bool(currentVal))
								else:
									newVal = currentVal + variationDiff

								if(positiveExcitatoryWeights):
									newVal = max(newVal, 0)	#do not allow weights fall below zero [CHECKTHIS]	

								B[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "B")][networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].assign(newVal)

						metricAfterStochasticUpdate = learningAlgorithmStochasticCalculateMetric(networkIndex, AprevLayer, ZprevLayer, l)
						#print("metricBase = ", metricBase)
						#print("metricAfterStochasticUpdate = ", metricAfterStochasticUpdate)

						if(metricAfterStochasticUpdate > metricBase):
							#print("(metricAfterStochasticUpdate > metricBase)")
							accuracyImprovementDetected = True
							metricBase = metricAfterStochasticUpdate
						#else:
							#print("(metricAfterStochasticUpdate < metricBase)")

						if(accuracyImprovementDetected):
							#retain weight update
							Wbackup[generateParameterNameNetwork(networkIndex, l, "W")].assign(W[generateParameterNameNetwork(networkIndex, l, "W")])
							Bbackup[generateParameterNameNetwork(networkIndex, l, "B")].assign(B[generateParameterNameNetwork(networkIndex, l, "B")])	
						else:
							#restore weights
							W[generateParameterNameNetwork(networkIndex, l, "W")].assign(Wbackup[generateParameterNameNetwork(networkIndex, l, "W")])
							B[generateParameterNameNetwork(networkIndex, l, "B")].assign(Bbackup[generateParameterNameNetwork(networkIndex, l, "B")])	

def neuralNetworkPropagationLIANNlearningAlgorithmShufflePermanence(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights):
	Afinal, Zfinal, _ = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)

	#update W/B permanence;
	Afinal2D = tf.reduce_mean(Afinal, axis=0)	#average across batch
	Afinal2D = tf.expand_dims(Afinal2D, axis=0)	#make compatible shape to W
	WpermanenceUpdate = tf.multiply(Afinal2D, WpermanenceUpdateRate)	#verify that broadcasting works
	WpermanenceNew = tf.add(Wpermanence[generateParameterNameNetwork(networkIndex, l, "Wpermanence")], WpermanenceUpdate)	#increase the permanence of neuron weights that successfully fired
	Wpermanence[generateParameterNameNetwork(networkIndex, l, "Wpermanence")] = WpermanenceNew
	print("WpermanenceUpdate = ", WpermanenceUpdate)

	#stochastically modify weights based on permanence values:
	Wupdate = randomNormal([n_h[l-1], n_h[l]])
	Wupdate = tf.divide(Wupdate, Wpermanence[generateParameterNameNetwork(networkIndex, l, "Wpermanence")])
	Wupdate = tf.divide(Wupdate, permanenceNumberBatches)
	Wnew = tf.add(W[generateParameterNameNetwork(networkIndex, l, "W")], Wupdate)
	if(positiveExcitatoryWeights):
		Wnew = tf.maximum(Wnew, 0)	#do not allow weights fall below zero [CHECKTHIS]
	W[generateParameterNameNetwork(networkIndex, l, "W")] = Wnew
	#print("Wupdate = ", Wupdate)

def neuralNetworkPropagationLIANNlearningAlgorithmHebbian(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights):
	AW = W[generateParameterNameNetwork(networkIndex, l, "W")] 
	Afinal, Zfinal, EWactive = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)
	#print("Zfinal = ", Zfinal)

	if(useZAcoincidenceMatrix):
		AWcontribution = tf.matmul(tf.transpose(ZprevLayer), Afinal)	#increase excitatory weights that contributed to the output signal	#hebbian
	else:
		AWcontribution = tf.matmul(tf.transpose(AprevLayer), Afinal)	#increase excitatory weights that contributed to the output signal	#hebbian

	if(randomlyActivateWeights):
		#do not apply weight updates to temporarily suppressed weights [CHECKTHIS];
		AWcontribution = tf.multiply(AWcontribution, EWactive)		

	if(normaliseWeightUpdates):
		print("ANNtf2_algorithmLIANN:neuralNetworkPropagationLIANN error - normaliseWeightUpdates: normaliseWeightUpdatesReduceConnectionWeightsForUnassociatedNeurons unimplemented")
	else:
		if(maxWeightUpdateThreshold):
			AWcontribution = tf.minimum(AWcontribution, 1.0)

	AWupdate = tf.multiply(AWcontribution, learningRate)
	#print("AWupdate = ", AWupdate)

	AW = tf.add(AW, AWupdate)	#apply weight updates

	if(weightDecay):
		#apply decay to all weights;
		AWdecay = -weightDecayRate
		#print("AWdecay = ", AWdecay)
		AW = tf.add(AW, AWdecay)
		#print("AWdecay = ", AWdecay)

	if(positiveExcitatoryWeightsThresholds):
		AW = tf.minimum(AW, 1.0)	#do not allow weights to exceed 1.0 [CHECKTHIS]
		AW = tf.maximum(AW, 0)	#do not allow weights fall below zero [CHECKTHIS]

	W[generateParameterNameNetwork(networkIndex, l, "W")] = AW





def forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition=False, randomlyActivateWeights=False):
	#forward excitatory connections;
	EWactive = None
	EW = W[generateParameterNameNetwork(networkIndex, l, "W")]
	if(randomlyActivateWeights):
		#print("EW = ", EW)
		EWactive = tf.less(tf.random.uniform(shape=EW.shape), randomlyActivateWeightsProbability)	#initialised from 0.0 to 1.0
		EWactive = tf.dtypes.cast(EWactive, dtype=tf.dtypes.float32) 
		#print("EWactive = ", EWactive)
		#EWactive = tf.dtypes.cast(tf.random.uniform(shape=EW.shape, minval=0, maxval=2, dtype=tf.dtypes.int32), dtype=tf.dtypes.float32)
		EW = tf.multiply(EW, EWactive)
	Z = tf.add(tf.matmul(AprevLayer, EW), B[generateParameterNameNetwork(networkIndex, l, "B")])
	A = activationFunction(Z, n_h[l-1])

	#lateral inhibitory connections (incoming/outgoing);
	if(enableInhibition):
		Afinal, Zfinal = forwardIterationInhibition(networkIndex, AprevLayer, ZprevLayer, l, A, Z)
	else:
		Zfinal = Z
		Afinal = A
	
	return Afinal, Zfinal, EWactive

def forwardIterationInhibition(networkIndex, AprevLayer, ZprevLayer, l, A, Z):
	if(inhibitionAlgorithmArtificial):
		if(inhibitionAlgorithmArtificialSparsity):
			prevLayerSize = n_h[l-1]
			inhibitionResult = tf.math.reduce_mean(AprevLayer, axis=1)	#or ZprevLayer?	#batched
			#print("inhibitionResult = ", inhibitionResult)
			inhibitionResult = tf.multiply(inhibitionResult, prevLayerSize)	#normalise by prev layer size		#batched
			inhibitionResult = tf.multiply(inhibitionResult, Wmean)	#normalise by average weight
			inhibitionResult = tf.expand_dims(inhibitionResult, axis=1)	#batched
			Zfinal = tf.subtract(Z, inhibitionResult)	#requires broadcasting
			#print("Z = ", Z)
			#print("Zfinal = ", Zfinal)
			Afinal = activationFunction(Zfinal, prevLayerSize=prevLayerSize)
		elif(inhibitionAlgorithmArtificialMoreThanXLateralNeuronActive):
			layerSize = n_h[l]
			numActiveLateralNeurons = tf.math.count_nonzero(A, axis=1)
			if(inhibitionAlgorithmMoreThanXLateralNeuronActiveFraction):
				numberActiveNeuronsAllowed = inhibitionAlgorithmMoreThanXLateralNeuronActiveFractionValue*layerSize
			else:
				numberActiveNeuronsAllowed = inhibitionAlgorithmMoreThanXLateralNeuronActiveValue
			numberActiveNeuronsAllowed = int(numberActiveNeuronsAllowed)
			#print("numActiveLateralNeurons = ", numActiveLateralNeurons)
			#print("numberActiveNeuronsAllowed = ", numberActiveNeuronsAllowed)
			inhibitionResult = tf.greater(numActiveLateralNeurons, numberActiveNeuronsAllowed)

			inhibitionResult = tf.logical_not(inhibitionResult)
			inhibitionResult = tf.dtypes.cast(inhibitionResult, dtype=tf.dtypes.float32)
			inhibitionResult = tf.expand_dims(inhibitionResult, axis=1)
			#print("numActiveLateralNeurons = ", numActiveLateralNeurons)
			#print("inhibitionResult = ", inhibitionResult)
			Zfinal = tf.multiply(Z, inhibitionResult)	#requires broadcasting
			Afinal = tf.multiply(A, inhibitionResult)	#requires broadcasting		
	else:
		#if((l < numberOfLayers) or positiveExcitatoryWeightsFinalLayer):

		#print("AprevLayer = ", AprevLayer)
		#print("Z = ", Z)

		IZi = tf.matmul(A, IWi[generateParameterNameNetwork(networkIndex, l, "IWi")])	#CHECKTHIS: inhibitory neuron firing is a function of current (lateral) layer (not previous layer)
		IAi = activationFunction(IZi, n_h[l-1])
		#print("IZi = ", IZi)
		#print("IAi = ", IAi)
		IZo = tf.matmul(IAi, IWo[generateParameterNameNetwork(networkIndex, l, "IWo")])
		#print("W = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
		#print("IZo = ", IZo)

		#final activations;
		Zfinal = tf.add(Z, IZo)
		#print("Zfinal = ", Zfinal)
		Afinal = activationFunction(Zfinal, n_h[l-1])
		
	if(Athreshold):
		Afinal = tf.minimum(Afinal, AthresholdValue)
		#print("Afinal = ", Afinal)
		
	return Afinal, Zfinal


									
def learningAlgorithmStochasticCalculateMetric(networkIndex, AprevLayer, ZprevLayer, l):
	randomlyActivateWeights = False
	if(randomlyActivateWeightsDuringTrain):
		randomlyActivateWeights = true
		
	if(learningAlgorithmStochasticCorrelation):
		enableInhibition = False
		A, Z, _ = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)
		metric = ANNtf2_algorithmLIANN_math.learningAlgorithmStochasticCalculateMetricCorrelation(A)
	elif(learningAlgorithmStochasticMaximiseAndEvenSignal):
		enableInhibition = True
		Afinal, Zfinal, _ = forwardIteration(networkIndex, AprevLayer, ZprevLayer, l, enableInhibition, randomlyActivateWeights)
		metric = ANNtf2_algorithmLIANN_math.learningAlgorithmStochasticCalculateMetricMaximiseAndEvenSignal(Afinal, metric1Weighting, metric2Weighting)
	return metric

	

def getRandomNetworkParameter(networkIndex, currentSubsetOfParameters):
	
	foundNewParameter = False
	while not foundNewParameter:
	
		variationDirection = random.randint(2)
		layer = random.randint(1, len(n_h))
		parameterTypeWorBtemp = random.randint(n_h[layer-1]+1)	#ensure that B parameter updates occur/tested less frequently than W parameter updates	#OLD: random.randint(2)	
		if(parameterTypeWorBtemp == n_h[layer-1]):
			parameterTypeWorB = 0
		else:
			parameterTypeWorB = 1
		hIndexCurrentLayer = random.randint(n_h[layer])	#randomNormal(n_h[l])
		hIndexPreviousLayer = random.randint(n_h[layer-1]) #randomNormal(n_h[l-1])
		networkParameterIndex = (parameterTypeWorB, layer, hIndexCurrentLayer, hIndexPreviousLayer, variationDirection)
	
		matches = [item for item in currentSubsetOfParameters if item == networkParameterIndex]
		if len(matches) == 0:
			foundNewParameter = True
			
	return networkParameterIndex				


def activationFunction(Z, prevLayerSize=None):
	return reluCustomPositive(Z, prevLayerSize)
	
def reluCustomPositive(Z, prevLayerSize=None):
	if(positiveExcitatoryWeightsActivationFunctionOffset):
		#CHECKTHIS: consider sigmoid instead of relu
		#offset required because negative weights are not used:
		
		#Zoffset = tf.ones(Z.shape)
		#Zoffset = tf.multiply(Zoffset, normalisedAverageInput)
		#Zoffset = tf.multiply(Zoffset, Wmean)
		#Zoffset = tf.multiply(Zoffset, prevLayerSize)
		Zpred = prevLayerSize*normalisedAverageInput*Wmean
		Zoffset = Zpred
		#print("Zoffset = ", Zoffset)
		
		Z = tf.subtract(Z, Zoffset) 
		A = tf.nn.relu(Z)
		A = tf.multiply(A, 2.0)	#double the slope of A to normalise the input:output signal
		#print("A = ", A)
	else:
		A = tf.nn.relu(Z)
		#A = tf.nn.sigmoid(Z)
	return A
	

def count_zero(M, axis=None):	#emulates count_nonzero behaviour
	if axis is not None:
		nonZeroElements = tf.math.count_nonzero(M, axis=axis)
		totalElements = tf.shape(M)[axis]
		zeroElements = tf.subtract(totalElements, nonZeroElements)
	else:
		totalElements = tf.size(M)
		nonZeroElements = tf.math.count_nonzero(M).numpy()
		zeroElements = tf.subtract(totalElements, nonZeroElements)
		zeroElements = zeroElements.numpy()
	return zeroElements



