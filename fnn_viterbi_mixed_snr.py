# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 09:08:39 2017

@author: Fahrettin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 09:30:09 2017

@author: Enes
"""

import tensorflow as tf
import numpy as np
from random import randint
import math as math

# Network Parameters
n_hidden_1 = 128  #
n_hidden_2 = 64  #
n_hidden_3 = 32
blocklength = 7
n_input = 2 * blocklength
n_classes = 2**blocklength
ALLSET = 2**blocklength
numofclasses = ALLSET

numoftrainingsamples = numofclasses * 1000
instanceforeachsampletraining = int(numoftrainingsamples / numofclasses)
numoftestsamples = numofclasses * 10000
instanceforeachsampletest = int(numoftestsamples / ALLSET)

testsetwordnum = 5000
randomtestsetlen = blocklength * testsetwordnum

# Parameters
learning_rate = 0.0001
training_epochs = 300
batch_size = 256
display_step = 1
beta = 0.001

snrdbvec = np.arange(-2, 10)
SNR = 10 ** (snrdbvec / 10)
N0 = 1 / SNR
sigma = np.sqrt(N0 * 0.5)

chosentestsnrindex = 0

train = True
testoldmodel = False
usedropoutmodel = False
userandomtestset = False
snrnumeachclass = int(instanceforeachsampletraining / len(snrdbvec))


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


def extractBits(arr, predictions):
    predictionbits = np.zeros((len(predictions), blocklength), dtype=np.uint8)

    for i in range(len(predictions)):
        bits = np.zeros(blocklength)
        val = [int(x) for x in list("{0:0b}".format(arr[i]))]
        bits[len(bits) :] = np.array(val)
        predictionbits[i, :] = bits

    return predictionbits


blocklength = 7
rate = 2
allset = 2**blocklength
numofclasses = allset


def extractBitssingle(arrval):
    bits = np.zeros(blocklength)
    val = [int(x) for x in list("{0:0b}".format(arrval))]
    bits[0 : len(val)] = np.array(val)
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


def binaryToDecimal(arr):
    c = 2 ** np.arange(arr.shape[0])
    v = arr.dot(c)
    return v


def calculateBer(predictions, labels, bitnum):
    biterr = sum(list(map(calHammingDistanceDecimal, predictions, labels)))
    ber = biterr / (bitnum * blocklength)
    return ber


def calculateAccuracy(predictions, testlabels):
    packeterrnum = 0
    testlabels = testlabels.astype(int)
    for i in range(len(predictions)):
        if predictions[i] != testlabels[i]:
            packeterrnum = packeterrnum + 1

    frameerror = packeterrnum / len(predictions)
    print("accuracy = ", 1 - frameerror)


def get_dataset():
    # Prapare test ans training data
    classbits = np.zeros((allset, blocklength))
    for i in range(allset):
        classbits[i, :] = np.array(
            list(np.binary_repr(i, width=blocklength)), dtype=np.uint8
        )

    fullencodedset = np.zeros((allset, blocklength * 2))
    for i in range(allset):
        fullencodedset[i, :] = encode133171(classbits[i, :])

    partialencodedset = np.zeros((numofclasses, blocklength * 2))
    for i in range(numofclasses):
        partialencodedset[i, :] = encode133171(classbits[i, :])

    trainingset = np.zeros((numoftrainingsamples, blocklength * 2))
    traininglabels = np.zeros(numoftrainingsamples)

    testset = np.zeros((numoftestsamples, blocklength * 2))
    testlabels = np.zeros(numoftestsamples)

    for k in range(allset):
        for i in range(len(snrdbvec)):
            for j in range(snrnumeachclass):
                trainingset[
                    instanceforeachsampletraining * k + snrnumeachclass * i + j, :
                ] = modulateAwgn(fullencodedset[k, :], sigma[i])
        traininglabels[
            k * instanceforeachsampletraining : (k + 1) * instanceforeachsampletraining
        ] = k

    for i in range(allset):
        for j in range(instanceforeachsampletest):
            testset[instanceforeachsampletest * i + j, :] = modulateAwgn(
                fullencodedset[i, :], sigma[chosentestsnrindex]
            )
            testlabels[instanceforeachsampletest * i + j] = i

    randomtestset = np.random.randint(2, size=randomtestsetlen)
    encodedrandomtestset = encode133171(randomtestset)
    encodedrandomtestset = modulateAwgn(encodedrandomtestset, sigma[chosentestsnrindex])
    encodedrandomtestset = encodedrandomtestset.reshape(testsetwordnum, 14)
    binaryrandomtestsetlabels = randomtestset.reshape(testsetwordnum, 7)
    decimalrandomtestsetlabels = np.zeros((testsetwordnum), dtype=np.uint16)

    for i in range(len(decimalrandomtestsetlabels)):
        decimalrandomtestsetlabels[i] = binaryToDecimal(binaryrandomtestsetlabels[i, :])

    traininglabels = traininglabels.astype(int)
    randinds = np.random.permutation(len(testlabels))
    testset = testset[randinds, :]
    testlabels = testlabels[randinds]
    testlabels = testlabels.astype(int)

    # create one hot encoding for labels
    onehottraininglabels = np.zeros([len(trainingset), n_classes], dtype=np.float16)
    onehottestlabels = np.zeros([len(testset), n_classes], dtype=np.float16)
    onehotrandomtestlabels = np.zeros([testsetwordnum, n_classes], dtype=np.float16)
    for i in range(len(trainingset)):
        onehottraininglabels[i, traininglabels[i]] = 1
        if i < len(testlabels):
            onehottestlabels[i, testlabels[i]] = 1

    trainingset = trainingset.astype(np.float16)
    testset = testset.astype(np.float16)

    for i in range(testsetwordnum):
        onehotrandomtestlabels[i, decimalrandomtestsetlabels[i]] = 1

    return (
        trainingset,
        testset,
        onehottraininglabels,
        onehotrandomtestlabels,
        encodedrandomtestset,
    )


def create_model_graph(trainingset, testset, encodedrandomtestset):
    tf.reset_default_graph()

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    keep_prob = tf.placeholder("float")

    # Create model
    def multilayer_perceptron(x, weights, biases):
        layer_1 = tf.add(tf.matmul(x, weights["h1"]), biases["b1"])
        layer_1 = tf.nn.tanh(layer_1)

        layer_2 = tf.add(tf.matmul(layer_1, weights["h2"]), biases["b2"])
        layer_2 = tf.nn.tanh(layer_2)

        layer_3 = tf.add(tf.matmul(layer_2, weights["h3"]), biases["b3"])
        layer_3 = tf.nn.tanh(layer_3)

        out_layer = tf.matmul(layer_3, weights["out"]) + biases["out"]
        out_layer = tf.nn.tanh(out_layer)

        return out_layer

    def multilayer_perceptron_withdropout(x, weights, biases, keep_prob):
        layer_1 = tf.add(tf.matmul(x, weights["h1"]), biases["b1"])
        layer_1 = tf.nn.tanh(layer_1)
        layer_1 = tf.nn.dropout(layer_1, keep_prob)

        layer_2 = tf.add(tf.matmul(layer_1, weights["h2"]), biases["b2"])
        layer_2 = tf.nn.tanh(layer_2)
        layer_2 = tf.nn.dropout(layer_2, keep_prob)

        layer_3 = tf.add(tf.matmul(layer_2, weights["h3"]), biases["b3"])
        layer_3 = tf.nn.tanh(layer_3)
        layer_3 = tf.nn.dropout(layer_3, keep_prob)

        out_layer = tf.matmul(layer_3, weights["out"]) + biases["out"]
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
        "out": tf.Variable(
            tf.truncated_normal(
                [n_hidden_3, n_classes], stddev=math.sqrt(2.0 / (n_hidden_3))
            )
        ),
    }
    biases = {
        "b1": tf.Variable(tf.zeros([n_hidden_1])),
        "b2": tf.Variable(tf.zeros([n_hidden_2])),
        "b3": tf.Variable(tf.zeros([n_hidden_3])),
        "out": tf.Variable(tf.zeros([n_classes])),
    }

    # Construct model
    pred = multilayer_perceptron_withdropout(x, weights, biases, keep_prob)
    testmodel = multilayer_perceptron(x, weights, biases)

    if not usedropoutmodel:
        pred = testmodel

        # Define loss and optimizer
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
    )
    regularizer = (
        tf.nn.l2_loss(weights["h1"])
        + tf.nn.l2_loss(weights["h2"])
        + tf.nn.l2_loss(weights["h3"])
    )
    cost = tf.reduce_mean(cost + beta * regularizer)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    if train:
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.0
                total_batch = int(numoftrainingsamples / batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x = trainingset[i * batch_size : (i + 1) * batch_size, :]
                    batch_y = onehottraininglabels[
                        i * batch_size : (i + 1) * batch_size
                    ]
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run(
                        [optimizer, cost], feed_dict={x: batch_x, y: batch_y}
                    )
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print(
                        "Epoch:",
                        "%04d" % (epoch + 1),
                        "cost=",
                        "{:.9f}".format(avg_cost),
                    )
            print("Optimization Finished!")
            saver.save(sess, trainedmodeldirectory)
            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            g = tf.argmax(pred, 1)
            predictions = g.eval(feed_dict={x: testset})
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({x: testset, y: onehottestlabels}))

    if testoldmodel:
        if userandomtestset:
            with tf.Session() as session:
                saver = tf.train.import_meta_graph(testmodeldirectory)
                saver.restore(session, tf.train.latest_checkpoint("./"))
                correct_prediction = tf.equal(tf.argmax(testmodel, 1), tf.argmax(y, 1))
                g = tf.argmax(testmodel, 1)
                predictions = g.eval(feed_dict={x: encodedrandomtestset})
                # Calculate accuracy
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print(
                    "Accuracy:",
                    accuracy.eval({x: encodedrandomtestset, y: onehotrandomtestlabels}),
                )

        else:
            with tf.Session() as session:
                saver = tf.train.import_meta_graph(testmodeldirectory)
                saver.restore(session, tf.train.latest_checkpoint("./"))
                correct_prediction = tf.equal(tf.argmax(testmodel, 1), tf.argmax(y, 1))
                g = tf.argmax(testmodel, 1)
                predictions = g.eval(feed_dict={x: testset})
                # Calculate accuracy
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print("Accuracy:", accuracy.eval({x: testset, y: onehottestlabels}))

        calculateAccuracy(predictions, decimalrandomtestsetlabels)
        ber = calculateBer(predictions, decimalrandomtestsetlabels, randomtestsetlen)

        calculateAccuracy(predictions, testlabels)
        ber = calculateBer(predictions, testlabels, numoftestsamples)

        print("ber: ", ber)
