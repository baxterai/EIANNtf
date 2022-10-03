# -*- coding: utf-8 -*-
"""MNISTlargeUntrainedNetCNNexcInh.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Uts0ul0iHzAHM5nC_LcJm5eHAAsRcCjI

# MNIST Large Untrained Net CNN Exc Inh

Derived from https://www.kaggle.com/code/amyjang/tensorflow-mnist-cnn-tutorial/notebook
"""

import tensorflow as tf
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import backend as K

print(tf.__version__)

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

positiveWeightImplementation = False	#orig: True
if(not positiveWeightImplementation):
    integrateWeights = True
    if(integrateWeights):
        integrateWeights1 = False    #explicitly declare E/I neurons
        integrateWeights2 = True    #implicitly declare E/I neurons   
        integrateWeightsInitialiseZero = True   #improves training performance
 
preFinalDenseLayer = False

generateUntrainedNetwork = False
if(generateUntrainedNetwork):
    #only train the last layer
    numberOfHiddenLayers = 2    #default: 2    #if 0 then useSVM=True
else:
    numberOfHiddenLayers = 2   #default: 4

if(numberOfHiddenLayers > 1):
    addSkipLayers = False   #optional
else:
    addSkipLayers = False   #mandatory

layerSizeBase = 32  #default: 32

batch_size = 64 #default: 64
epochs = 5  #1  #5

debugNoEIneurons = False
debugPreTrainWeights = True
debugPreTrainOutputs = True
debugPostTrainWeights = True
debugPostTrainOutputs = True

"""## Load data"""

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

input_shape = (28, 28, 1)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train=x_train / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test=x_test/255.0

y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

plt.imshow(x_train[100][:,:,0])
print(y_train[100])

"""## Define model"""

num_classes = 10

def activation(x):
    return K.maximum(x, 0)  #ReLU

def activationExcitatory(x):
    return K.maximum(x, 0)  #ReLU

def activationInhibitory(x):
    if(positiveWeightImplementation):
        return -(K.maximum(x, 0))   #ReLU with negative output
    else:
        return K.maximum(x, 0)  #ReLU

def neuronInitializer(shape, dtype=None):
    if(positiveWeightImplementation):
        print("neuronInitializer error: requires !positiveWeightImplementation:integrateWeights")
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
            print("neuronInitializer error: requires !positiveWeightImplementation:integrateWeights")
    return w

def excitatoryNeuronInitializer(shape, dtype=None):
    return tf.math.abs(tf.random.normal(shape, dtype=dtype))

def inhibitoryNeuronInitializer(shape, dtype=None):
    if(positiveWeightImplementation):
        return tf.math.abs(tf.random.normal(shape, dtype=dtype))
    else:
        return tf.math.negative(tf.math.abs(tf.random.normal(shape, dtype=dtype)))

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

if(positiveWeightImplementation):
    weightConstraint = tf.keras.constraints.non_neg()
    constrainBiases = True   #ensure positive biases also
    if(constrainBiases):
        biasConstraint = tf.keras.constraints.non_neg()
        constrainBiasesLastLayer = False
        if(constrainBiasesLastLayer):
            biasConstraintLastLayer = tf.keras.constraints.non_neg()
        else:
            biasConstraintLastLayer = None
    else:
        biasConstraint = None
        biasConstraintLastLayer = None
else:
    if(integrateWeights):
        weightConstraint = positiveOrNegative()
        biasConstraint = None
    else:
        weightConstraintPositive = tf.keras.constraints.non_neg()
        weightConstraintNegative = negative()
        constrainBiases = False
        if(constrainBiases):
            biasConstraintPositive = tf.keras.constraints.non_neg()
            biasConstraintNegative = negative()
        else:
            biasConstraintPositive = None
            biasConstraintNegative = None

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

def createEIlayer(layerIndex, h0, firstLayer=False, maxpool2d=None, dropout=None):
    layerRatio = getLayerRatio(2)
    if(debugNoEIneurons):
        h1E = tf.keras.layers.Conv2D(layerSizeBase*layerRatio, (5,5), padding='same')(h0)
        h1I = tf.keras.layers.Conv2D(layerSizeBase*layerRatio, (5,5), padding='same')(h0)
        h1E = tf.keras.layers.Activation(activationExcitatory)(h1E)
        h1I = tf.keras.layers.Activation(activationInhibitory)(h1I)
        h1 = tf.keras.layers.Concatenate(axis=2)([h1E, h1I])
    else:
        if(positiveWeightImplementation):
            h1E = tf.keras.layers.Conv2D(layerSizeBase*layerRatio, (5,5), kernel_initializer=excitatoryNeuronInitializer, kernel_constraint=weightConstraint, bias_constraint=biasConstraint, padding='same')(h0)
            h1I = tf.keras.layers.Conv2D(layerSizeBase*layerRatio, (5,5), kernel_initializer=inhibitoryNeuronInitializer, kernel_constraint=weightConstraint, bias_constraint=biasConstraint, padding='same')(h0)
            h1E = tf.keras.layers.Activation(activationExcitatory)(h1E)
            h1I = tf.keras.layers.Activation(activationInhibitory)(h1I)
            h1 = tf.keras.layers.Concatenate()([h1E, h1I])
            if(maxpool2d is not None):
                h1 = tf.keras.layers.MaxPool2D(strides=maxpool2d)(h1)
            if(dropout is not None):
                h1 = tf.keras.layers.Dropout(dropout)(h1)
        else:
            if(integrateWeights):
                if(integrateWeights1):
                    if(firstLayer):
                        h1E = tf.keras.layers.Conv2D(layerSizeBase*layerRatio, (5,5), padding='same')(h0)
                        h1I = tf.keras.layers.Conv2D(layerSizeBase*layerRatio, (5,5), padding='same')(h0)
                    else:
                        h1E = tf.keras.layers.Conv2D(layerSizeBase*layerRatio, (5,5), kernel_initializer=neuronInitializer, kernel_constraint=weightConstraint, bias_constraint=biasConstraint, padding='same')(h0)
                        h1I = tf.keras.layers.Conv2D(layerSizeBase*layerRatio, (5,5), kernel_initializer=neuronInitializer, kernel_constraint=weightConstraint, bias_constraint=biasConstraint, padding='same')(h0)
                    h1E = tf.keras.layers.Activation(activation)(h1E)
                    h1I = tf.keras.layers.Activation(activation)(h1I)
                    h1 = tf.keras.layers.Concatenate(axis=2)([h1E, h1I])
                elif(integrateWeights2):
                    if(firstLayer):
                        h1 = tf.keras.layers.Conv2D(layerSizeBase*layerRatio*2, (5,5), padding='same')(h0)
                    else:
                        h1 = tf.keras.layers.Conv2D(layerSizeBase*layerRatio*2, (5,5), kernel_initializer=neuronInitializer, kernel_constraint=weightConstraint, bias_constraint=biasConstraint, padding='same')(h0)
                    h1 = tf.keras.layers.Activation(activation)(h1)
                if(maxpool2d is not None):
                    h1 = tf.keras.layers.MaxPool2D(strides=maxpool2d)(h1)
                if(dropout is not None):
                    h1 = tf.keras.layers.Dropout(dropout)(h1)
            else:
                if(firstLayer):
                    h1E = tf.keras.layers.Conv2D(layerSizeBase*layerRatio, (5,5), kernel_initializer=excitatoryNeuronInitializer, padding='same')(h0)
                    h1I = tf.keras.layers.Conv2D(layerSizeBase*layerRatio, (5,5), kernel_initializer=inhibitoryNeuronInitializer, padding='same')(h0)
                else:
                    h0E, h0I = h0
                    h1Ee = tf.keras.layers.Conv2D(layerSizeBase*layerRatio, (5,5), kernel_initializer=excitatoryNeuronInitializer, kernel_constraint=weightConstraintPositive, bias_constraint=biasConstraintPositive, padding='same')(h0E) #excitatory neuron excitatory inputs
                    h1Ei = tf.keras.layers.Conv2D(layerSizeBase*layerRatio, (5,5), kernel_initializer=excitatoryNeuronInitializer, kernel_constraint=weightConstraintNegative, bias_constraint=biasConstraintNegative, padding='same')(h0I) #excitatory neuron inhibitory inputs
                    h1Ie = tf.keras.layers.Conv2D(layerSizeBase*layerRatio, (5,5), kernel_initializer=inhibitoryNeuronInitializer, kernel_constraint=weightConstraintPositive, bias_constraint=biasConstraintPositive, padding='same')(h0E) #inhibitory neuron excitatory inputs
                    h1Ii = tf.keras.layers.Conv2D(layerSizeBase*layerRatio, (5,5), kernel_initializer=inhibitoryNeuronInitializer, kernel_constraint=weightConstraintNegative, bias_constraint=biasConstraintNegative, padding='same')(h0I) #inhibitory neuron inhibitory inputs
                    h1E = tf.keras.layers.Add()([h1Ee, h1Ei])
                    h1I = tf.keras.layers.Add()([h1Ie, h1Ii])
                h1E = tf.keras.layers.Activation(activationExcitatory)(h1E)
                h1I = tf.keras.layers.Activation(activationInhibitory)(h1I)
                if(maxpool2d is not None):
                    h1E = tf.keras.layers.MaxPool2D(strides=maxpool2d)(h1E)
                    h1I = tf.keras.layers.MaxPool2D(strides=maxpool2d)(h1I)
                if(dropout is not None):
                    h1E = tf.keras.layers.Dropout(dropout)(h1E)
                    h1I = tf.keras.layers.Dropout(dropout)(h1I)
                h1 = (h1E, h1I)
    return h1

def concatEIneurons(h):
    if(positiveWeightImplementation):
        return h
    else: 
        if(integrateWeights):
            pass
        else:
            hE, hI = h
            h = tf.keras.layers.Concatenate()([hE, hI])
        return h


x = tf.keras.layers.Input(shape=input_shape)
h0 = x
hLast = h0
if(numberOfHiddenLayers >= 1):
    h1 = createEIlayer(1, h0, firstLayer=True)
    hLast = h1
if(numberOfHiddenLayers >= 2):
    h2 = createEIlayer(2, h1, maxpool2d=(1,1), dropout=0.25)
    hLast = h2
if(numberOfHiddenLayers >= 3):
    h3 = createEIlayer(3, h2)
    hLast = h3
if(numberOfHiddenLayers >= 4):
    h4 = createEIlayer(4, h3, maxpool2d=(2,2))
    hLast = h4
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
    hLast = tf.keras.layers.Dense(128, activation='relu')(hLast)
    hLast = tf.keras.layers.Dropout(0.5)(hLast)
if(generateUntrainedNetwork):
    hLast = tf.keras.layers.Lambda(lambda x: tf.keras.backend.stop_gradient(x))(hLast)
y = tf.keras.layers.Dense(num_classes, activation='softmax')(hLast)
model = tf.keras.Model(x, y)

print(model.summary())

model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), loss='categorical_crossentropy', metrics=['acc'])

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.995):
      print("\nReached 99.5% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

if(debugPreTrainWeights):
    testwritefile = open('weightsPreTrain.txt', 'w')
    for layerIndex, layer in enumerate(model.layers):
        heading = "layerWeights = " + str(layerIndex) + "\n"
        #print(heading)
        testwritefile.write(heading)
        weights = layer.get_weights()
        #print(weights)
        weightsS =  str(weights)
        testwritefile.write(weightsS)
    testwritefile.close()

if(debugPreTrainOutputs):
    testwritefile = open('outputPreTrain.txt', 'w')
    xTrainFirstSample = np.expand_dims(x_train[0], axis=0)
    for layerIndex, layer in enumerate(model.layers):
        heading = "layerOutputs = " + str(layerIndex) + "\n"
        #print(heading)
        testwritefile.write(heading)
        func = K.function([model.get_layer(index=0).input], layer.output)
        layerOutput = func([xTrainFirstSample])  # input_data is a numpy array
        #print(layerOutput)
        layerOutputS =  str(layerOutput)
        testwritefile.write(layerOutputS)
    testwritefile.close()

"""## Train model"""

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    callbacks=[callbacks])

if(debugPostTrainWeights):
    testwritefile = open('weightsPostTrain.txt', 'w')
    for layerIndex, layer in enumerate(model.layers):
        heading = "layerWeights = " + str(layerIndex) + "\n"
        #print(heading)
        testwritefile.write(heading)
        weights = layer.get_weights()
        #print(weights)
        weightsS =  str(weights)
        testwritefile.write(weightsS)
    testwritefile.close()

if(debugPostTrainOutputs):
    testwritefile = open('outputPostTrain.txt', 'w')
    xTrainFirstSample = np.expand_dims(x_train[0], axis=0)
    for layerIndex, layer in enumerate(model.layers):
        heading = "layerOutputs = " + str(layerIndex) + "\n"
        #print(heading)
        testwritefile.write(heading)
        func = K.function([model.get_layer(index=0).input], layer.output)
        layerOutput = func([xTrainFirstSample])  # input_data is a numpy array
        #print(layerOutput)
        layerOutputS =  str(layerOutput)
        testwritefile.write(layerOutputS)
    testwritefile.close()

"""## Evaluate model"""

fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training Loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation Loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training Accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation Accuracy")
legend = ax[1].legend(loc='best', shadow=True)

test_loss, test_acc = model.evaluate(x_test, y_test)