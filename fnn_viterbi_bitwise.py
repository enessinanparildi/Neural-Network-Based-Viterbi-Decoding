# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:03:55 2017

@author: Fahrettin
"""

import tensorflow as tf
import numpy as np
from random import randint
import math as math
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


# Another strategy involves doing prediction bit per bit in binary classification sense. But unlike
# neuralviterbiseqprection.py , we need to train state number of distinct classifier for every state so the algoritm decide
# which of the next classifier will be used according to previous prediction


def generateInfoSeq(blocklength):
    infoseq = np.zeros(blocklength)
    for i in range(blocklength):
        infoseq[i] = randint(0, 1)
    return infoseq


def encode57(infoseq):
    encodedseq = np.zeros(len(infoseq) * 2)
    encodedseq[0] = infoseq[0]
    encodedseq[1] = (infoseq[0] + infoseq[1]) % 2
    encodedseq[2] = (infoseq[1]) % 2
    encodedseq[3] = (infoseq[1] + infoseq[0]) % 2
    for i in range(2, len(infoseq)):
        encodedseq[2 * i] = (infoseq[i] + infoseq[i - 2]) % 2
        encodedseq[2 * i + 1] = (infoseq[i] + infoseq[i - 1] + infoseq[i - 2]) % 2
    return encodedseq


# Encoder have a capacity of start at spesified state
def encode133171(bits, statenum):
    arr = extractBitssingle(statenum)
    bits = np.append(arr, bits)

    encoded_bits = np.zeros(len(bits) * 2)
    encoded_bits[0] = (bits[0]) % 2
    encoded_bits[1] = (bits[0]) % 2

    encoded_bits[2] = (bits[1]) % 2
    encoded_bits[3] = (bits[1] + bits[0]) % 2

    encoded_bits[4] = (bits[2] + bits[0]) % 2
    encoded_bits[5] = (bits[2] + bits[1] + bits[0]) % 2

    encoded_bits[6] = (bits[3] + bits[1] + bits[0]) % 2
    encoded_bits[7] = (bits[3] + bits[2] + bits[1] + bits[0]) % 2

    encoded_bits[8] = (bits[4] + bits[2] + bits[1] + 0 + 0) % 2
    encoded_bits[9] = (bits[4] + bits[3] + bits[2] + bits[1]) % 2

    encoded_bits[10] = (bits[5] + bits[3] + bits[2] + bits[0]) % 2
    encoded_bits[11] = (bits[5] + bits[4] + bits[3] + bits[2]) % 2

    for i in range(6, len(bits)):
        encoded_bits[2 * i] = (
            bits[i] + bits[i - 2] + bits[i - 3] + bits[i - 5] + bits[i - 6]
        ) % 2
        encoded_bits[2 * i + 1] = (
            bits[i] + bits[i - 1] + bits[i - 2] + bits[i - 3] + bits[i - 6]
        ) % 2

    return encoded_bits


# modulate and add noise
def modulateAwgn(encodedseq, sigma):
    encodedseq[np.where(encodedseq == 0)] = -1
    distortedseq = encodedseq + np.random.normal(0, sigma, len(encodedseq))
    return distortedseq


def generateInputVectors(length, number):
    randomtestset = np.random.randint(2, size=(number, length))
    return randomtestset


# Blocklength is number of bit the system will try to decode at once
constraintlength = 6
blocklength = 512
totallength = constraintlength + blocklength


def extractBitssingle(arrval):
    bits = np.zeros(6)
    val = [int(x) for x in list("{0:0b}".format(arrval))]
    bits[len(bits) - len(val) : len(bits)] = np.array(val)
    return bits


def calHammingDistanceDecimal(val1, val2):
    set1 = np.arange(blocklength)
    set2 = 2**set1
    biterr = 0
    for i in range(blocklength):
        if val1 % (2 * set2[i]) != 0 and val2 % (2 * set2[i]) == 0:
            val1 = val1 - set2[i]
            biterr = biterr + 1
        elif val1 % (2 * set2[i]) == 0 and val2 % (2 * set2[i]) != 0:
            val2 = val2 - set2[i]
            biterr = biterr + 1
        elif val1 % (2 * set2[i]) != 0 and val2 % (2 * set2[i]) != 0:
            val2 = val2 - set2[i]
            val1 = val1 - set2[i]
    return biterr


def generateTrainData(blocklength, trainingsize, noisevector, state):
    traininginfoseq = generateInputVectors(blocklength, trainingsize)
    #    traininginfoseq[:,0]=1

    traininginfoseq = traininginfoseq.astype(np.float32)
    trainingencodedseq = np.array(
        list(
            map(
                encode133171,
                traininginfoseq.tolist(),
                state * np.ones(len(traininginfoseq), dtype=np.uint8),
            )
        )
    )

    for i in range(trainingsize):
        ind = np.random.randint(len(noisevector), size=1)
        trainingencodedseq[i, :] = modulateAwgn(
            trainingencodedseq[i, :], noisevector[ind]
        )
    return (traininginfoseq, trainingencodedseq)


def generateTestData(blocklength, testsize, noisevector, noiseindex, state):
    testinfoseq = generateInputVectors(blocklength, testsize)
    #    testinfoseq[:,0]=1

    testcodedseq = np.array(
        list(
            map(
                encode133171,
                testinfoseq.tolist(),
                state * np.ones(len(testinfoseq), dtype=np.uint8),
            )
        )
    )
    for i in range(testsize):
        testcodedseq[i, :] = modulateAwgn(testcodedseq[i, :], noisevector[noiseindex])
    return (testinfoseq, testcodedseq)


tf.reset_default_graph()

# Network Parameters node numbers
n_hidden_1 = 30  #
n_hidden_2 = 100  #
n_hidden_3 = 100
n_hidden_4 = 100
# number of input layer node must have same dimensionality as input size
n_input = 2 * totallength

# How many blocklength of block will be used for training ans test more trainingsize may give better result
trainingsize = 300000
testsize = 300000

# Prapare different snr values
snrdbvec = np.arange(-2, 9)
SNR = 10 ** (snrdbvec / 10)
N0 = 1 / SNR
sigma = np.sqrt(N0 * 0.5)

# test vectors will go through this level of snr lower index will give high error since it has more noise power
chosentestsnrindex = 0
chosenbitposition = 0
startingstate = 0

learning_rate = 0.0002
training_epochs = 300
batch_size = 1000
display_step = 1

# Saving directory
trainedmodeldirectory = "C:\\Users\\Fahrettin\\neuralviterbi\\mymodelfirstbitpredict"
testmodeldirectory = "C:\\Users\\Fahrettin\\neuralviterbi\\mymodelfirstbitpredict.meta"
# False means train new one that will be saved spesified directory
testoldmodel = False

beta = 0.0001

tf.reset_default_graph()
sess = tf.InteractiveSession()

# Generate information and encoded stream
traininfoseq, traincodedseq = generateTrainData(
    blocklength, trainingsize, sigma, startingstate
)
testinfoseq, testcodedseq = generateTestData(
    blocklength, testsize, sigma, chosentestsnrindex, startingstate
)

# Spesified bit positiomn of info sequnce will be binary label this is whar are we trying to predict

trainlabels = traininfoseq[:, chosenbitposition]
testlabels = testinfoseq[:, chosenbitposition]

# Generate one hot represantion of labels this is accepted inpt type of network
onehottraininglabels = np.zeros([trainingsize, 2], dtype=np.float16)
onehottestlabels = np.zeros([testsize, 2], dtype=np.float16)
for i in range(trainingsize):
    onehottraininglabels[i, int(trainlabels[i])] = 1
    if i < len(testlabels):
        onehottestlabels[i, int(testlabels[i])] = 1

trainlabels = onehottraininglabels
testlabels = onehottestlabels

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, 2])
keep_prob = tf.placeholder("float")


# Create model
def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights["h1"]), biases["b1"])
    layer_1 = tf.nn.relu(layer_1)

    #    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    #    layer_2 = tf.nn.relu(layer_2)

    #    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    #    layer_3 = tf.nn.relu(layer_3)
    #
    #
    #    layer_4 = tf.add(tf.matmul(layer_3, weights['h3']), biases['b3'])
    #    layer_4 = tf.nn.relu(layer_4)

    out_layer = tf.matmul(layer_1, weights["out"]) + biases["out"]
    out_layer = tf.nn.relu(out_layer)

    return out_layer


# Store layers weight & bias
weights = {
    "h1": tf.Variable(
        tf.truncated_normal([n_input, n_hidden_1], stddev=math.sqrt(2.0 / (n_input)))
    ),
    "h2": tf.Variable(
        tf.truncated_normal(
            [n_hidden_1, n_hidden_2], stddev=math.sqrt(2.0 / (n_hidden_1))
        )
    ),
    "h3": tf.Variable(
        tf.truncated_normal(
            [n_hidden_2, n_hidden_3], stddev=math.sqrt(2.0 / (n_hidden_2))
        )
    ),
    "h4": tf.Variable(
        tf.truncated_normal(
            [n_hidden_3, n_hidden_4], stddev=math.sqrt(2.0 / (n_hidden_3))
        )
    ),
    "out": tf.Variable(
        tf.truncated_normal([n_hidden_1, 2], stddev=math.sqrt(2.0 / (n_hidden_4)))
    ),
}
biases = {
    "b1": tf.Variable(tf.zeros([n_hidden_1])),
    "b2": tf.Variable(tf.zeros([n_hidden_2])),
    "b3": tf.Variable(tf.zeros([n_hidden_3])),
    "b4": tf.Variable(tf.zeros([n_hidden_4])),
    "out": tf.Variable(tf.zeros([2])),
}

# Construct model
# pred = multilayer_perceptron_withdropout(x, weights, biases, keep_prob)
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer

cost = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y, name=None)
)

regularizer = tf.nn.l2_loss(weights["h1"]) + tf.nn.l2_loss(weights["h2"])

cost = tf.reduce_mean(cost + beta * regularizer)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Launch the graph
if testoldmodel:
    with tf.Session() as session:
        saver = tf.train.import_meta_graph(testmodeldirectory)
        saver.restore(session, tf.train.latest_checkpoint("./"))

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        g = tf.argmax(pred, 1)
        # test prediction values
        predictions = g.eval(feed_dict={x: testcodedseq})
        testprediction = pred.eval(feed_dict={x: testcodedseq})
        testpredictioncopy = testprediction

        trainingprediction = pred.eval(feed_dict={x: traincodedseq})
else:
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(trainingsize / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x = traincodedseq[i * batch_size : (i + 1) * batch_size, :]
                batch_y = trainlabels[i * batch_size : (i + 1) * batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

                #                trainpredictions = pred.eval(feed_dict = {x:trainingencodedseq})
                #                ber = calBer(trainpredictions,traininginfoseq)
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                #                print("ber:" , ber )
                print(
                    "Epoch:", "%04d" % (epoch + 1), "cost=", "{:.9f}".format(avg_cost)
                )
        print("Optimization Finished!")
        saver.save(sess, trainedmodeldirectory)
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        g = tf.argmax(pred, 1)
        predictions = g.eval(feed_dict={x: testcodedseq})
        conf = pred.eval(feed_dict={x: testcodedseq})
        testprediction = pred.eval(feed_dict={x: testcodedseq})
        testpredictioncopy = testprediction
        trainingprediction = pred.eval(feed_dict={x: traincodedseq})

# Calculate errors
packeterrnum = 0
testlabels = testinfoseq[:, chosenbitposition]
testlabels = testlabels.astype(int)
for i in range(len(predictions)):
    if predictions[i] != testlabels[i]:
        packeterrnum = packeterrnum + 1

frameerror = packeterrnum / len(predictions)
print("error = ", frameerror)
