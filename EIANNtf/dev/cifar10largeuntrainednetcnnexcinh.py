# -*- coding: utf-8 -*-
"""CIFAR10largeUntrainedNetCNNexcInh.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1AGiE11i41ksQWhaXgxBvU6cYEOCG0Zo5

# CIFAR10 Large Untrained Net CNN Exc Inh

Derived from  https://keras.io/zh/examples/cifar10_cnn_tfaugment2d/
"""

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, Lambda, MaxPooling2D
from keras import backend as K
from keras.src.utils.np_utils import to_categorical
import os

if K.backend() != 'tensorflow':
    raise RuntimeError('This example can only run with the '
                       'TensorFlow backend, '
                       'because it requires TF-native augmentation APIs')

import tensorflow as tf
import numpy as np
import math

inhibitoryNeuronOutputPositive = True
if(inhibitoryNeuronOutputPositive):
    inhibitoryNeuronSwitchActivation = True
else:
    inlineImplementation = False	#False: only implementation  #True: excitatory/inhibitory neurons are on same sublayer, False: add inhibitory neurons to separate preceding sublayer
    if(not inlineImplementation):
        positiveWeightImplementation = False    #False: only current coded implementation
        inhibitoryNeuronNormalisationFactorStatic = False    #True: normalise intermediary inhibitory neuron layer based on h0/h1 num neurons, False: normalise based on h0/h1 activations
        excitatoryNeuronThreshold = 0.0   #orig: 0.0

useSparsity = False
if(useSparsity):
  sparsityProbabilityOfConnection = 0.1 #1-sparsity
#addSkipLayers = False  #skip layers not supported by keras model.add definition format

inputLayerExcitatoryOnly = True #True: only current coded implementation

generateUntrainedNetwork = False
if(generateUntrainedNetwork):
    numberOfHiddenLayers = 2  #default = 2, if 0 then useSVM=True
    preFinalDenseLayer = False
else:
    numberOfHiddenLayers = 2  #default = 4, if 0 then useSVM=True
    preFinalDenseLayer = False


if(numberOfHiddenLayers > 1):
    addSkipLayers = False   #optional
else:
    addSkipLayers = False   #mandatory

layerSizeBase = 32  #default: 32

batch_size = 32
epochs = 5  #100

debugNoEIneurons = False
debugPreTrainWeights = True
debugPreTrainOutputs = True
debugPostTrainWeights = True
debugPostTrainOutputs = True
if(debugNoEIneurons):
    numberOfHiddenLayers = 4  #default = 4, if 0 then useSVM=True
    preFinalDenseLayer = True

if(generateUntrainedNetwork):
    #only train the last layer
    generateLargeNetwork = True
else:
    generateLargeNetwork = False

if(generateLargeNetwork):
    largeNetworkRatio = 10    #100
    generateLargeNetworkExpansion = False
    if(generateLargeNetworkExpansion):
        generateLargeNetworkRatioExponential = False
else:
    generateLargeNetworkRatio = False
    largeNetworkRatio = 1

def getLayerRatio(layerIndex):
    layerRatio = 1
    if(generateLargeNetwork):
        if(generateLargeNetworkExpansion):
            if(generateLargeNetworkRatioExponential):
                layerRatio = largeNetworkRatio**layerIndex
            else:
                layerRatio = largeNetworkRatio * layerIndex
        else:
            layerRatio = largeNetworkRatio
    else:
        layerRatio = 1
    return int(layerRatio)

def kernelInitializerWithSparsity(shape, dtype=None):
    initialisedWeights = tf.random.normal(shape, dtype=dtype) #change to glorot_uniform?
    sparsityMatrixMask = tf.random.uniform(shape, minval=0.0, maxval=1.0, dtype=tf.dtypes.float32)
    sparsityMatrixMask = tf.math.less(sparsityMatrixMask, sparsityProbabilityOfConnection)
    sparsityMatrixMask = tf.cast(sparsityMatrixMask, dtype=tf.dtypes.float32)
    initialisedWeights = tf.multiply(initialisedWeights, sparsityMatrixMask)
    return initialisedWeights

if(useSparsity):
     kernelInitializer = kernelInitializerWithSparsity
else:
    kernelInitializer = 'glorot_uniform'

"""## Load data"""

num_classes = 10
num_predictions = 20
save_dir = '/tmp/saved_models'
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
print("input_shape = ", input_shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

"""## Define model"""

def EIactivation(Z):
    A = K.maximum(Z, excitatoryNeuronThreshold)-excitatoryNeuronThreshold  #ReLU
    return A

def EIactivationExcitatory(Z):
    if(inlineImplementation):
        if(positiveWeightImplementation):
            return EIactivation(Z)
        else:
             print("EIactivationExcitatory error: requires positiveWeightImplementation")
    else:
        print("EIactivationExcitatory error: requires inlineImplementation")

def EIactivationInhibitory(Z):
    if(inlineImplementation):
        if(positiveWeightImplementation):
            return -EIactivation(Z)   #ReLU with negative output
        else:
             print("EIactivationInhibitory error: requires positiveWeightImplementation")
    else:
        print("inlineImplementation error: requires inlineImplementation")

def EIweightInitializer(shape, dtype=None):
    if(inlineImplementation):
        if(positiveWeightImplementation):
            w = tf.math.abs(tf.random.normal(shape, dtype=dtype))
        else:
            if(integrateWeights):
                if(integrateWeightsInitialiseZero):
                    w = tf.zeros(shape, dtype=dtype)    #tf.math.abs(tf.random.normal(shape, dtype=dtype))
                else:
                    #print("shape = ", shape)
                    w = tf.math.abs(tf.random.normal(shape, dtype=dtype))
                    wEIsize = w.shape[2]//2
                    wSignE = tf.ones([w.shape[0], w.shape[1], wEIsize, w.shape[3]])
                    wSignI = tf.ones([w.shape[0], w.shape[1], wEIsize, w.shape[3]])
                    wSignI = tf.multiply(wSignI, -1)
                    wSign = tf.concat([wSignE, wSignI], axis=2)
                    w = tf.multiply(w, wSign)
            else:
                print("EIweightInitializer error: requires !positiveWeightImplementation:integrateWeights")
    else:
        print("EIweightInitializer error: requires inlineImplementation")

    return w

def EIweightInitializerExcitatory(shape, dtype=None):
    if(positiveWeightImplementation):
        print("EIweightInitializerExcitatory error: requires !positiveWeightImplementation")
    else:
        return tf.math.abs(tf.random.normal(shape, dtype=dtype))

def EIweightInitializerInhibitory(shape, dtype=None):
    if(positiveWeightImplementation):
        print("EIweightInitializerExcitatory error: requires !positiveWeightImplementation")
    else:
        return tf.math.negative(tf.math.abs(tf.random.normal(shape, dtype=dtype)))

def EIweightInitialisedAverage(shape):
    return tf.reduce_mean(tf.math.abs(tf.random.normal(shape)))

class negative(tf.keras.constraints.Constraint):
    #based on https://www.tensorflow.org/api_docs/python/tf/keras/constraints/Constraint
    def __init__(self):
        pass
    def __call__(self, w):
        return w * tf.cast(tf.math.less_equal(w, 0.), w.dtype)

class positiveOrNegative(tf.keras.constraints.Constraint):
    #based on https://www.tensorflow.org/api_docs/python/tf/keras/constraints/Constraint
    def __init__(self):
        pass
    def __call__(self, w):
        w_shape = w.shape
        #print("w_shape = ", w_shape)
        wEIsize = w.shape[2]//2
        wE = w[:, :, 0:wEIsize]
        wI = w[:, :, wEIsize:]
        wEcheck = tf.greater_equal(wE, 0)
        wIcheck = tf.less_equal(wI, 0)
        wEcheck = tf.cast(wEcheck, tf.float32)
        wIcheck = tf.cast(wIcheck, tf.float32)
        wE = tf.multiply(wE, wEcheck)
        wI = tf.multiply(wI, wIcheck)
        w = tf.concat([wE, wI], axis=2)
        return w

if(not inhibitoryNeuronOutputPositive):
    if(not inlineImplementation):
        EIweightConstraintPositive = tf.keras.constraints.non_neg()
        EIweightConstraintNegative = negative()
        constrainBiases = False
        if(constrainBiases):
            EIbiasConstraintPositive = tf.keras.constraints.non_neg()
            EIbiasConstraintNegative = negative()
        else:
            EIbiasConstraintPositive = None
            EIbiasConstraintNegative = None
        EIweightConstraintLastLayer = None
        EIbiasConstraintLastLayer = None


def createEIlayer(layerIndex, h0, numChannels, previousNumChannels, firstLayer=False, maxpool2d=None, dropout=None):
    layerRatio = getLayerRatio(2)
    if(debugNoEIneurons):
        h1 = tf.keras.layers.Conv2D(numChannels, (3,3), padding='same')(h0)
        h1 = tf.keras.layers.Activation(EIactivation)(h1)
        if(maxpool2d is not None):
            h1 = tf.keras.layers.MaxPool2D(pool_size=maxpool2d)(h1)
        if(dropout is not None):
            h1 = tf.keras.layers.Dropout(dropout)(h1)
    else:
        if(inhibitoryNeuronOutputPositive):
            h1E = tf.keras.layers.Conv2D(numChannels, (5,5), padding='same')(h0)
            h1I = tf.keras.layers.Conv2D(numChannels, (5,5), padding='same')(h0)
            h1E = tf.keras.layers.Activation('relu')(h1E)
            if(inhibitoryNeuronSwitchActivation):
                h1I = tf.keras.layers.Activation('relu')(-h1I)
            else:
                h1I = tf.keras.layers.Activation('relu')(h1I)
            h1 = tf.keras.layers.Concatenate()([h1E, h1I])
            if(maxpool2d is not None):
                h1 = tf.keras.layers.MaxPool2D(pool_size=maxpool2d)(h1)
            if(dropout is not None):
                h1 = tf.keras.layers.Dropout(dropout)(h1)
        else:
            if(not inlineImplementation):
                h1I = tf.keras.layers.Conv2D(previousNumChannels, (5,5), padding='same', kernel_initializer=EIweightInitializerExcitatory, kernel_constraint=EIweightConstraintPositive, bias_constraint=EIbiasConstraintPositive)(h0) #inhibitory interneuron (excitatory inputs)
                if(not inhibitoryNeuronNormalisationFactorStatic):  #disabled because modifies activation levels
                    h1I = tf.keras.layers.Activation(EIactivation)(h1I)
                h1I = h1I*calculateInhibitoryNeuronNormalisationFactor(h0, h1I, numChannels, previousNumChannels, firstLayer)
                h1Ee = tf.keras.layers.Conv2D(numChannels, (5,5), padding='same', kernel_initializer=EIweightInitializerExcitatory, kernel_constraint=EIweightConstraintPositive, bias_constraint=EIbiasConstraintPositive)(h0) #excitatory neuron excitatory inputs
                h1Ei = tf.keras.layers.Conv2D(numChannels, (5,5), padding='same', kernel_initializer=EIweightInitializerInhibitory, kernel_constraint=EIweightConstraintNegative, bias_constraint=EIbiasConstraintNegative)(h1I) #excitatory neuron inhibitory inputs
                h1E = tf.keras.layers.Add()([h1Ee, h1Ei])
                h1E = tf.keras.layers.Activation(EIactivation)(h1E)
                h1 = h1E
    return h1

def calculateAverageWeight(numChannels, previousNumChannels):
    shape = [previousNumChannels,numChannels]
    averageWeight = EIweightInitialisedAverage(shape)
    #avg = x*sqrt(pi/2) = 1.25331413732 #https://stats.stackexchange.com/questions/363240/mean-of-absgauss-as-a-function-of-the-standard-deviation
    return averageWeight

def calculateInhibitoryNeuronNormalisationFactor(h0, h1I, numChannels, previousNumChannels, firstLayer=False):
    if(inhibitoryNeuronNormalisationFactorStatic):
        previousNumChannels = previousNumChannels*input_shape[1]*input_shape[2]
        numChannels = numChannels*input_shape[1]*input_shape[2]
        averageLayerActivation = 0.5    #this is not correct
        averageWeight = calculateAverageWeight(numChannels, previousNumChannels)
        if(firstLayer):
            #assume input layer unequal activation/nonactivation level
            averageLayerActivation = np.mean(x_train)
        print("previousNumChannels = ", previousNumChannels)
        print("averageLayerActivation = ", averageLayerActivation)
        print("averageWeight = ", averageWeight)
        h1InormalisationFactor = (1/previousNumChannels*averageWeight)*averageLayerActivation
    else:
        h1InormalisationFactor = tf.reduce_mean(h0)/tf.reduce_mean(h1I)
    return h1InormalisationFactor

def concatEIneurons(h):
    if(inhibitoryNeuronOutputPositive):
        return h
    else:
        if(inlineImplementation):
            if(positiveWeightImplementation):
                return h
            else:
                if(integrateWeights):
                    pass
                else:
                    hE, hI = h
                    h = tf.keras.layers.Concatenate()([hE, hI])
                return h
        else:
            return h

x = tf.keras.layers.Input(shape=input_shape)
h0 = x
hLast = h0

previousNumChannels = input_shape[2]   #3
if(numberOfHiddenLayers >= 1):
    numChannels = layerSizeBase*1*getLayerRatio(1)
    h1 = createEIlayer(1, h0, numChannels, previousNumChannels, firstLayer=True)
    hLast = h1
    previousNumChannels = numChannels
if(numberOfHiddenLayers >= 2):
    numChannels = layerSizeBase*1*getLayerRatio(2)
    h2 = createEIlayer(2, h1, numChannels, previousNumChannels, maxpool2d=(2,2), dropout=0.25)
    hLast = h2
    previousNumChannels = numChannels
if(numberOfHiddenLayers >= 3):
    numChannels = layerSizeBase*2*getLayerRatio(3)
    h3 = createEIlayer(3, h2, numChannels, previousNumChannels)
    hLast = h3
    previousNumChannels = numChannels
if(numberOfHiddenLayers >= 4):
    numChannels = layerSizeBase*2*getLayerRatio(4)
    h4 = createEIlayer(4, h3, numChannels, previousNumChannels, maxpool2d=(2,2), dropout=0.25)
    hLast = h4
    previousNumChannels = numChannels

if(addSkipLayers):
    mList = []
    if(numberOfHiddenLayers >= 1):
        m1 = tf.keras.layers.Flatten()(concatEIneurons(h1))
        mList.append(m1)
    if(numberOfHiddenLayers >= 2):
        m2 = tf.keras.layers.Flatten()(concatEIneurons(h2))
        mList.append(m2)
    if(numberOfHiddenLayers >= 3):
        m3 = tf.keras.layers.Flatten()(concatEIneurons(h3))
        mList.append(m3)
    if(numberOfHiddenLayers >= 4):
        m4 = tf.keras.layers.Flatten()(concatEIneurons(h4))
        mList.append(m4)
    hLast = tf.keras.layers.concatenate(mList)
else:
    hLast = concatEIneurons(hLast)

hLast = tf.keras.layers.Flatten()(hLast)
if(preFinalDenseLayer):
    numChannels = 512*largeNetworkRatio
    hLast = tf.keras.layers.Dense(numChannels, activation='relu', kernel_initializer=kernelInitializer)(hLast)
    hLast = tf.keras.layers.Dropout(0.5)(hLast)

if(generateUntrainedNetwork):
    hLast = tf.keras.layers.Lambda(lambda x: tf.keras.backend.stop_gradient(x))(hLast)

y = tf.keras.layers.Dense(num_classes, activation='softmax')(hLast)
model = tf.keras.Model(x, y)

print(model.summary())
#printModelSummary(model)

opt = tf.keras.optimizers.RMSprop(epsilon=1e-08)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    #orig: model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

"""## Train model"""

if(debugPreTrainWeights):
    testwritefile = open('weightsPreTrain.txt', 'w')
    for layerIndex, layer in enumerate(model.layers):
        heading = "\n" + "layer = " + str(layerIndex) + "\n"
        testwritefile.write(heading)
        weights = layer.get_weights()
        #weightsAvg = np.mean(weights[0])
        #print(heading)
        #print(weights)
        weightsS =  str(weights)
        testwritefile.write(weightsS)
    testwritefile.close()

if(debugPreTrainOutputs):
    testwritefile = open('outputPreTrain.txt', 'w')
    xTrainFirstSample = np.expand_dims(x_train[0], axis=0)
    for layerIndex, layer in enumerate(model.layers):
        heading = "\n" + "layer = " + str(layerIndex) + "\n"
        testwritefile.write(heading)
        func = K.function([model.get_layer(index=0).input], layer.output)
        layerOutput = func([xTrainFirstSample])  # input_data is a numpy array
        #print(heading)
        #print(layerOutput)
        layerOutputS =  str(layerOutput)
        testwritefile.write(layerOutputS)
    testwritefile.close()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

print(model.summary())

if(debugPostTrainWeights):
    testwritefile = open('weightsPostTrain.txt', 'w')
    for layerIndex, layer in enumerate(model.layers):
        heading = "\n" + "layer = " + str(layerIndex) + "\n"
        testwritefile.write(heading)
        weights = layer.get_weights()
        #print(heading)
        #print(weights)
        weightsS =  str(weights)
        testwritefile.write(weightsS)
    testwritefile.close()

if(debugPostTrainOutputs):
    testwritefile = open('outputPostTrain.txt', 'w')
    xTrainFirstSample = np.expand_dims(x_train[0], axis=0)
    for layerIndex, layer in enumerate(model.layers):
        heading = "\n" + "layer = " + str(layerIndex) + "\n"
        testwritefile.write(heading)
        func = K.function([model.get_layer(index=0).input], layer.output)
        layerOutput = func([xTrainFirstSample])  # input_data is a numpy array
        #print(heading)
        #print(layerOutput)
        layerOutputS =  str(layerOutput)
        testwritefile.write(layerOutputS)
    testwritefile.close()

"""## Evaluate model"""

# Save model and weights
#if not os.path.isdir(save_dir):
#    os.makedirs(save_dir)
#model_path = os.path.join(save_dir, model_name)
#model.save(model_path)
#print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])