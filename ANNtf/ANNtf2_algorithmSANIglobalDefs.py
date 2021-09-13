# -*- coding: utf-8 -*-
"""ANNtf2_algorithmSANIglobalDefs.py

# Requirements:
Python 3 and Tensorflow 2.1+ 

# License:
MIT License

# Usage:
see ANNtf2.py

# Description:

Define Sequentially Activated Neuronal Input (SANI) neural net - global defs

- Author: Richard Bruce Baxter - Copyright (c) 2020-2021 Baxter AI (baxterai.com)

"""

import tensorflow as tf
import numpy as np
import ANNtf2_globalDefs


algorithmSANI = "sharedModulesHebbian"
#algorithmSANI = "sharedModulesBinary"
#algorithmSANI = "sharedModules"
#algorithmSANI = "repeatedModules"

createSmallNetworkForDebug = True

useSequentialInputs = True
if(useSequentialInputs):
	numberOfSequentialInputs = 2	#2	#3	#1 - no sequential input requirement enforced
else:
	numberOfSequentialInputs = 1

if(algorithmSANI == "sharedModulesHebbian"):
	useHebbianLearningRule = True
	if(useHebbianLearningRule):
		useHebbianLearningRulePositiveWeights = True
		neuronActivationFiringThreshold = 1.0
		useHebbianLearningRuleApply = True
		if(useHebbianLearningRuleApply):
			#not currently compatible with supportFullConnectivity:supportFeedback
			#these parameters require calibration:
			hebbianLearningRate = 0.01
			minimumConnectionWeight = 0.0
			maximumConnectionWeight = 1.0
	supportFullConnectivity = True	#full connectivity between layers	
	supportFeedback = False	#optional 
else:
	useHebbianLearningRule = False
	supportFullConnectivity = False	#unimplemented (could be added in future)
	supportFeedback = False	#unimplemented (could be added in future)
#supportFeedback note: activation/A must be maintained across multiple iteration forward propagation through layers
	#currently requires SANIsharedModules=True
	#SANIsharedModules=False would need to be upgraded to perform multiple forward pass iterations
	
veryLargeInt = 99999999


if(algorithmSANI == "sharedModulesHebbian"):
	SANIsharedModules = True	#optional
elif(algorithmSANI == "sharedModulesBinary"):
	SANIsharedModules = True	#mandatory (only coded implementation)
elif(algorithmSANI == "sharedModules"):
	SANIsharedModules = True	#mandatory	(only coded implementation)
elif(algorithmSANI == "repeatedModules"): 	
	SANIsharedModules = False	#mandatory (only coded implementation)
#SANIsharedModules note: uses shifting input x feed, enabling identical input subsets (eg phrases/subreferencesets) irrespective of their sentence position to be sent to same modules/neurons

if(algorithmSANI == "sharedModulesHebbian"):
	sequentialInputActivationThreshold = 1.0	#CHECKTHIS (requires optimisation)
elif(algorithmSANI == "sharedModulesBinary"):
	pass
elif(algorithmSANI == "sharedModules"):
	sequentialInputActivationThreshold = 0.1	#CHECKTHIS (requires optimisation)
elif(algorithmSANI == "repeatedModules"):
	pass

if(algorithmSANI == "sharedModulesHebbian"):
	allowMultipleSubinputsPerSequentialInput = True	#implied variable (granted via full connectivity)
	inputNumberFeaturesForCurrentWordOnly = True	#optional

elif(algorithmSANI == "sharedModulesBinary"):
	allowMultipleSubinputsPerSequentialInput = True	#required #originally set as False
	inputNumberFeaturesForCurrentWordOnly = True	#mandatory (only coded implementation)

	resetSequentialInputsIfOnlyFirstInputValid = True	#see GIA_TXT_REL_TRANSLATOR_NEURAL_NETWORK_SEQUENCE_GRAMMAR development history for meaning and algorithmic implications of this feature
	if(resetSequentialInputsIfOnlyFirstInputValid):
		doNotResetNeuronOutputUntilAllSequentialInputsActivated = True

	useSparseTensors = True	#mandatory
elif(algorithmSANI == "sharedModules"):
	allowMultipleSubinputsPerSequentialInput = False

	if(allowMultipleSubinputsPerSequentialInput):
		allowMultipleContributingSubinputsPerSequentialInput = False	#whether only 1 subinput can be fired to activate a sequential input	
		if(allowMultipleContributingSubinputsPerSequentialInput):
			inputNumberFeaturesForCurrentWordOnly = False	#the convolutional window (kernel) captures x words every time is slided to the right
		else:
			inputNumberFeaturesForCurrentWordOnly = True
	else:
		allowMultipleContributingSubinputsPerSequentialInput = False	#mandatory
		inputNumberFeaturesForCurrentWordOnly = True

	resetSequentialInputsIfOnlyFirstInputValid = True	#see GIA_TXT_REL_TRANSLATOR_NEURAL_NETWORK_SEQUENCE_GRAMMAR development history for meaning and algorithmic implications of this feature
	if(resetSequentialInputsIfOnlyFirstInputValid):
		if(allowMultipleContributingSubinputsPerSequentialInput):
			averageTimeChangeOfNewInputRequiredForReset = 1
		doNotResetNeuronOutputUntilAllSequentialInputsActivated = True

	if(allowMultipleSubinputsPerSequentialInput):
		if(allowMultipleContributingSubinputsPerSequentialInput):
			useSparseTensors = False	#optional
		else:
			useSparseTensors = True		#mandatory	#FUTURE: upgrade code to remove this requirement 
	else:
		useSparseTensors = True	#mandatory	#sparse tensors are used
 
elif(algorithmSANI == "repeatedModules"): 	
	allowMultipleSubinputsPerSequentialInput = False
	useSparseTensors = True	#mandatory
	inputNumberFeaturesForCurrentWordOnly = False	#NA (not used)



#set parameters oneSequentialInputHasOnlyOneSubinput:
if(algorithmSANI == "sharedModulesHebbian"):
	oneSequentialInputHasOnlyOneSubinput = False
elif(algorithmSANI == "sharedModulesBinary"):		
	expectNetworkConvergence = False
	if(expectNetworkConvergence):
		#if(numberOfSequentialInputs == 2):
		oneSequentialInputHasOnlyOneSubinput = True	#conditional probability determination of events
	else:
		oneSequentialInputHasOnlyOneSubinput = False
elif(algorithmSANI == "sharedModules"):	
	if(useSparseTensors):	#FUTURE: upgrade code to remove this requirement
		if(allowMultipleSubinputsPerSequentialInput):
			#if(numberOfSequentialInputs == 2):
			oneSequentialInputHasOnlyOneSubinput = True	#conditional probability determination of events
		else:
			oneSequentialInputHasOnlyOneSubinput = False
	else:
		oneSequentialInputHasOnlyOneSubinput = False
elif(algorithmSANI == "repeatedModules"):
	oneSequentialInputHasOnlyOneSubinput = False
if(oneSequentialInputHasOnlyOneSubinput):
	firstSequentialInputHasOnlyOneSubinput = True #use combination of allowMultipleSubinputsPerSequentialInput for different sequential inputs;  #1[#2] sequential input should allow multiple subinputs, #2[#1] sequential input should allow single subinput
	if(firstSequentialInputHasOnlyOneSubinput):
		lastSequentialInputHasOnlyOneSubinput = False
	else:
		lastSequentialInputHasOnlyOneSubinput = True

#set parameters enforceTcontiguityConstraints:
if(algorithmSANI == "sharedModulesHebbian"):
	enforceTcontiguityConstraints = False
elif(algorithmSANI == "sharedModulesBinary"):		
	enforceTcontiguityConstraints = True
elif(algorithmSANI == "sharedModules"):	
	if(allowMultipleSubinputsPerSequentialInput):
		if(useSparseTensors):
			if(not allowMultipleContributingSubinputsPerSequentialInput):
				enforceTcontiguityConstraints = True
			else:
				enforceTcontiguityConstraints = False
		else:
			enforceTcontiguityConstraints = False
	else:
		enforceTcontiguityConstraints = True
elif(algorithmSANI == "repeatedModules"): 
	enforceTcontiguityConstraints = False
if(enforceTcontiguityConstraints):
	enforceTcontiguityBetweenSequentialInputs = True
	enforceTcontiguityTakeEncapsulatedSubinputWithMinTvalue = True	#method to decide between subinput selection/parse tree generation
	enforceTcontiguityTakeEncapsulatedSubinputWithMaxTvalueEqualsW = True
	enforceTcontiguityStartAndEndOfSequence = True
	

if(inputNumberFeaturesForCurrentWordOnly):
	numberOfWordsInConvolutionalWindowSeen = 1	#always 1
else:
	numberOfWordsInConvolutionalWindowSeen = 10
	
#set parameters performSummationOfSubInputsWeighted/useLastSequentialInputOnly/numberOfWordsInConvolutionalWindowSeen:
if(algorithmSANI == "sharedModulesHebbian"):
	performSummationOfSubInputs = True	#mandatory (implied)
	if(performSummationOfSubInputs):
		performSummationOfSubInputsWeighted = True	#required
		performSummationOfSubInputsNonlinear = False	#optional
		performSummationOfSubInputsBinary = True	#simple thresholding function (activations are normalised to 1.0)
	performSummationOfSequentialInputs = True	#optional
	if(performSummationOfSequentialInputs):
		performSummationOfSequentialInputsWeighted = False	#optional (otherwise just sum them together or take a pass condition)
		performSummationOfSequentialInputsNonlinear = False	#optional		
		if(performSummationOfSequentialInputsWeighted):
			#sequentialInputCombinationModeSummation = 3
			sequentialInputCombinationModeSummation = 4
		else:
			#sequentialInputCombinationModeSummation = 1
			sequentialInputCombinationModeSummation = 2
			sequentialInputCombinationModeSummationAveraged = True
	else:
		pass	#if(performSummationOfSubInputsBinary): simple thresholding function (activations are normalised to 1.0)
elif(algorithmSANI == "sharedModulesBinary"):		
	performSummationOfSubInputsWeighted = False	#required
	useLastSequentialInputOnly = True	#implied variable (not used)
	performSummationOfSequentialInputsWeighted = False	#mandatory (implied)
elif(algorithmSANI == "sharedModules"):	
	if(allowMultipleSubinputsPerSequentialInput):
		performSummationOfSequentialInputs = True	#optional

		if(allowMultipleContributingSubinputsPerSequentialInput):
			#[multiple contributing subinputs per sequential input] #each sequential input can detect a pattern of activation from the previous layer
			performSummationOfSubInputs = True	#mandatory (implied)
			performSummationOfSubInputsWeighted = True	#mandatory?
			performSummationOfSubInputsNonlinear = True
		else:
			performSummationOfSubInputs = False	#optional though by algorithm design: False
			performSummationOfSubInputsWeighted = False	#will take (True: most weighted) (False: any) active time contiguous subinput
			performSummationOfSubInputsNonlinear = False

		if(performSummationOfSequentialInputs):
			performSummationOfSequentialInputsWeighted = False #True	#determines if backprop is required to update weight matrix associated with sequential inputs
			performSummationOfSequentialInputsNonlinear = False	#True	#applies nonlinear function to weighting
			performSummationOfSequentialInputsVerify = True	#verify that all (last) sequential inputs are activated	#CHECKTHIS
		else:
			performSummationOfSequentialInputsWeighted = False
			performSummationOfSequentialInputsNonlinear = False
			performSummationOfSequentialInputsVerify = False

		if(performSummationOfSequentialInputs):
			if(performSummationOfSequentialInputsWeighted):
				#sequentialInputCombinationModeSummation = 3
				sequentialInputCombinationModeSummation = 4
			else:
				#sequentialInputCombinationModeSummation = 1
				sequentialInputCombinationModeSummation = 2
				sequentialInputCombinationModeSummationAveraged = True	
		else:
			useLastSequentialInputOnly = True	#implied variable (not used)

	else:
		performSummationOfSubInputsWeighted = False

		performSummationOfSequentialInputs = True

		if(performSummationOfSequentialInputs):
			performSummationOfSequentialInputsWeighted = True	#does backprop require to update weight matrix associated with sequential inputs?
			performSummationOfSequentialInputsNonlinear = True
			performSummationOfSequentialInputsVerify = True	#verify that all (last) sequential inputs are activated	#CHECKTHIS
		else:
			performSummationOfSequentialInputsWeighted = False
			performSummationOfSequentialInputsNonlinear = False
			performSummationOfSequentialInputsVerify = True	#verify that all (last) sequential inputs are activated	#CHECKTHIS


		if(performSummationOfSequentialInputs):
			if(performSummationOfSequentialInputsWeighted):
				sequentialInputCombinationModeSummation = 3
			else:
				sequentialInputCombinationModeSummation = 1	
		else:
			useLastSequentialInputOnly = True	#implied variable (not used)	
elif(algorithmSANI == "repeatedModules"): 
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

		
#set parameters record:
if(algorithmSANI == "sharedModulesHebbian"):	
	recordNetworkWeights = False	#need to modify weights not just record them
	if(recordNetworkWeights):
		recordSubInputsWeighted = False	#batchSize must equal 1
		recordSequentialInputsWeighted = False
		recordNeuronsWeighted = False
elif(algorithmSANI == "sharedModulesBinary"):
	if(not ANNtf2_globalDefs.testHarness):	
		recordNetworkWeights = True
		if(recordNetworkWeights):
			recordSubInputsWeighted = True
			recordSequentialInputsWeighted = False	#may not be necessary (only used if can split neuron sequential inputs)
			recordNeuronsWeighted = True
			#FUTURE: prune network neurons/connections based on the relative strength of these weights
	else:
		recordNetworkWeights = False
		recordSubInputsWeighted = False
		recordSequentialInputsWeighted = False
		recordNeuronsWeighted = False
elif(algorithmSANI == "sharedModules"):
	if(useSparseTensors):
		if(allowMultipleSubinputsPerSequentialInput):		
			recordNetworkWeights = True
			if(recordNetworkWeights):
				recordSubInputsWeighted = True
				recordSequentialInputsWeighted = False	#may not be necessary (only used if can split neuron sequential inputs)
				recordNeuronsWeighted = True
				#FUTURE: prune network neurons/connections based on the relative strength of these weights
		else:
			recordNetworkWeights = False	#not yet coded
	else:
		recordNetworkWeights = False	#not yet coded
elif(algorithmSANI == "repeatedModules"): 	
	recordNetworkWeights = False


if(algorithmSANI == "sharedModulesHebbian"):
	supportSkipLayers = True
	maxNumberSubinputsPerSequentialInput = -1	#NA
elif(algorithmSANI == "sharedModulesBinary"):	
	if(expectNetworkConvergence):
		maxNumberSubinputsPerSequentialInput = 50	#~approx equal number of prev layer neurons/2 (FUTURE: make dynamic based on layer index)	#number of prior/future events in which to calculate a conditional probability
	else:
		maxNumberSubinputsPerSequentialInput = 1	#sparsity
	supportSkipLayers = True
elif(algorithmSANI == "sharedModules"):
	if(allowMultipleSubinputsPerSequentialInput):
		if(useSparseTensors):
			supportSkipLayers = True
			if(oneSequentialInputHasOnlyOneSubinput):
				maxNumberSubinputsPerSequentialInput = 50	#~approx equal number of prev layer neurons/2	#number of prior/future events in which to calculate a conditional probability
			else:
				maxNumberSubinputsPerSequentialInput = 3	#sparsity
		else:
			supportSkipLayers = True
	else:
		supportSkipLayers = True
else:
	supportSkipLayers = True
	maxNumberSubinputsPerSequentialInput = -1	#NA

	



