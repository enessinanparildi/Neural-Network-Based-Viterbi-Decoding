# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:13:12 2017

@author: Enes
"""

import tensorflow as tf
import numpy as np
from random import randint
import math as math
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn import svm


# For details refer to C:\Users\Fahrettin\neuralviterbi\ANN_decoder_Sujan_final
# I have tried to replicate result of this paper but it gives no good result maybe simpler encoder will give better
# It needs more work on it


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


def encode133171(bits):
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


def modulateAwgn(encodedseq, sigma):
    encodedseq[np.where(encodedseq == 0)] = -1
    distortedseq = encodedseq + np.random.normal(0, sigma, len(encodedseq))
    return distortedseq


def generateInputVectors(length, number):
    randomtestset = np.random.randint(2, size=(number, length))
    return randomtestset


# Convert decimal value to bit vector
def extractBitssingle(arrval):
    bits = np.zeros(6)
    val = [int(x) for x in list("{0:0b}".format(arrval))]
    bits[len(bits) - len(val) : len(bits)] = np.array(val)
    return bits


# For a spesified bit length calculate hamming distance of two decimal numbers
def calHammingDistanceDecimal(val1, val2, blocklength):
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


# Generate training stream
def generateTrainData(blocklength, trainingsize, noisevector):
    traininginfoseq = generateInputVectors(blocklength, trainingsize)

    traininginfoseq = traininginfoseq.astype(np.float32)
    trainingencodedseq = np.array(list(map(encode133171, traininginfoseq.tolist())))

    for i in range(trainingsize):
        # snr of training data can be configured here random or fixed value
        ind = np.random.randint(len(noisevector), size=1)
        ind = 2
        trainingencodedseq[i, :] = modulateAwgn(
            trainingencodedseq[i, :], noisevector[ind]
        )
    return (traininginfoseq, trainingencodedseq)


# Generate test stream


def generateTestData(blocklength, testsize, noisevector, noiseindex, state):
    testinfoseq = generateInputVectors(blocklength, testsize)

    testcodedseq = np.array(list(map(encode133171, testinfoseq.tolist())))
    for i in range(testsize):
        testcodedseq[i, :] = modulateAwgn(testcodedseq[i, :], noisevector[noiseindex])
    return (testinfoseq, testcodedseq)


# Get single training or test instance from stream
def getSingleInstance(windowlength, codedseq, infoseq, offset):
    codedseq = np.transpose(codedseq)
    infoseq = np.transpose(infoseq)
    batchx = codedseq[offset : offset + windowlength, 0]
    batchy = np.zeros(2)
    val = infoseq[int(offset * 0.5), :]
    batchy[int(val)] = 1
    return (batchx, batchy)


def model_graph():
    tf.reset_default_graph()

    # Stream length
    totaltraininglengthinfo = 1000100
    totaltraininglengthcoded = 2000200
    totaltestlengthinfo = 1000100
    totaltestlengthcoded = 2000200
    windowlength = 200
    trainingsize = 1
    testsize = 1
    # Training and test instance number
    totaltraininginstance = int((totaltraininglengthcoded - windowlength) * 0.5)
    totaltestinstance = int((totaltestlengthcoded - windowlength) * 0.5)
    # Network Parameters
    n_hidden_1 = 500  #
    n_hidden_2 = 500  #
    n_hidden_3 = 500
    n_hidden_4 = 500
    n_input = windowlength

    snrdbvec = np.arange(-2, 9)
    SNR = 10 ** (snrdbvec / 10)
    N0 = 1 / SNR
    sigma = np.sqrt(N0 * 0.5)

    chosentestsnrindex = 1

    learning_rate = 0.0004
    training_epochs = 50
    batch_size = 100
    display_step = 1

    # Model saving directory
    trainedmodeldirectory = "C:\\Users\\Fahrettin\\neuralviterbi\\mymodelstate5"
    testmodeldirectory = "C:\\Users\\Fahrettin\\neuralviterbi\\mymodelstate9.meta"

    testoldmodel = False

    # Regularazitoan constant
    beta = 0.00000

    tf.reset_default_graph()
    session = tf.InteractiveSession()

    # Get streams where we will generate data
    traininfoseq, traincodedseq = generateTrainData(
        totaltraininglengthinfo, trainingsize, sigma
    )
    testinfoseq, testcodedseq = generateTestData(
        totaltraininglengthinfo, testsize, sigma, chosentestsnrindex
    )

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, 2])
    keep_prob = tf.placeholder("float")

    # Create model for adding extra layer uncomment
    def multilayer_perceptron(x, weights, biases):
        layer_1 = tf.add(tf.matmul(x, weights["h1"]), biases["b1"])
        layer_1 = tf.nn.tanh(layer_1)

        layer_2 = tf.add(tf.matmul(layer_1, weights["h2"]), biases["b2"])
        layer_2 = tf.nn.tanh(layer_2)

        layer_3 = tf.add(tf.matmul(layer_2, weights["h3"]), biases["b3"])
        layer_3 = tf.nn.tanh(layer_3)

        layer_4 = tf.add(tf.matmul(layer_3, weights["h4"]), biases["b4"])
        layer_4 = tf.nn.tanh(layer_4)

        out_layer = tf.matmul(layer_4, weights["out"]) + biases["out"]
        out_layer = tf.nn.tanh(out_layer)

        return out_layer

    # Store layers weight & bias
    weights = {
        "h1": tf.Variable(
            tf.truncated_normal(
                [n_input, n_hidden_1], stddev=math.sqrt(2.0 / (n_input))
            )
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
            tf.truncated_normal([n_hidden_4, 2], stddev=math.sqrt(2.0 / (n_hidden_4)))
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
    offset = 0

    if testoldmodel:
        # Testing already trained model
        testdata = np.zeros((totaltestinstance, windowlength))
        testlabels = np.zeros((totaltestinstance, 2))
        for i in range(totaltestinstance):
            testdata[i, :], testlabels[i, :] = getSingleInstance(
                windowlength, testcodedseq, testinfoseq, 2 * i
            )

        traindata = np.zeros((totaltraininginstance, windowlength))
        trainlabels = np.zeros((totaltraininginstance, 2))
        for i in range(totaltestinstance):
            traindata[i, :], trainlabels[i, :] = getSingleInstance(
                windowlength, traincodedseq, traininfoseq, 2 * i
            )

        with tf.Session() as session:
            saver = tf.train.import_meta_graph(
                "C:\\Users\\Fahrettin\\neuralviterbi\\mymodelstate133171.meta"
            )
            saver.restore(
                session, "C:\\Users\\Fahrettin\\neuralviterbi\\mymodelstate133171"
            )

            g = tf.argmax(pred, 1)
            testprediction = g.eval(feed_dict={x: testdata})
            trainingprediction = g.eval(feed_dict={x: traindata})
            testlike = pred.eval(feed_dict={x: testdata})

    else:
        with tf.Session() as session:
            saver = tf.train.import_meta_graph(
                "C:\\Users\\Fahrettin\\neuralviterbi\\mymodelstate133171bigger2.meta"
            )
            saver.restore(
                session,
                "C:\\Users\\Fahrettin\\neuralviterbi\\mymodelstate133171bigger2",
            )
            #        session.run(init)
            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.0
                total_batch = int(totaltraininginstance / batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x = np.zeros((int(batch_size), windowlength))
                    batch_y = np.zeros((int(batch_size), 2))

                    for j in range(batch_size):
                        a = 2 * j + offset
                        batch_x[j, :], batch_y[j, :] = getSingleInstance(
                            windowlength, traincodedseq, traininfoseq, offset
                        )

                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = session.run(
                        [optimizer, cost], feed_dict={x: batch_x, y: batch_y}
                    )

                    offset = 2 * (i + 1) * batch_size

                    avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % display_step == 0:
                    #                print("ber:" , ber )
                    print(
                        "Epoch:",
                        "%04d" % (epoch + 1),
                        "cost=",
                        "{:.9f}".format(avg_cost),
                    )
            print("Optimization Finished!")
            saver.save(
                session,
                "C:\\Users\\Fahrettin\\neuralviterbi\\mymodelstate133171bigger2",
            )
            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            g = tf.argmax(pred, 1)

            # Prapare the test and traşning data here and get the result
            testdata = np.zeros((totaltestinstance, windowlength))
            testlabels = np.zeros((totaltestinstance, 2))

            for i in range(totaltestinstance):
                testdata[i, :], testlabels[i, :] = getSingleInstance(
                    windowlength, testcodedseq, testinfoseq, 2 * i
                )

            traindata = np.zeros((totaltraininginstance, windowlength))
            trainlabels = np.zeros((totaltraininginstance, 2))
            for i in range(totaltestinstance):
                traindata[i, :], trainlabels[i, :] = getSingleInstance(
                    windowlength, traincodedseq, traininfoseq, 2 * i
                )

                # These will be saved prediction result
            testprediction = g.eval(feed_dict={x: testdata})
            trainingprediction = g.eval(feed_dict={x: traindata})

    # Calculate errors
    packeterrnum = 0
    testlabels = testlabels[:, 1]
    testlabels = testlabels.astype(int)
    for i in range(len(testprediction)):
        if testprediction[i] != testlabels[i]:
            packeterrnum = packeterrnum + 1

    frameerror = packeterrnum / len(testprediction)
    print(" test error = ", frameerror)

    packeterrnum = 0
    trainlabels = trainlabels[:, 1]
    trainlabels = trainlabels.astype(int)
    for i in range(len(trainingprediction)):
        if trainingprediction[i] != trainlabels[i]:
            packeterrnum = packeterrnum + 1

    frameerror = packeterrnum / len(trainingprediction)
    print(" train error = ", frameerror)


#
