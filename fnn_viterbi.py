# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 09:30:09 2017

@author: Enes
"""

import tensorflow as tf
import numpy as np
from random import randint
import math as math

# This is the final version of block prediction type strategy where we are trying to predict whole info sequence at once
# in multiclass classification sense works excellent when low number of bits is decoded like 7-8 obviosly not practical
# for large bit length since class size becomes 2^(bit length) for small bit length it works very nice since we can train it
# with the whole codebook unlike other methods

BLOCKLENGTH = 7
RATE = 2
ALLSET = 2**BLOCKLENGTH
NUM_OF_CLASSES = ALLSET

# Network Parameters node numbers
N_HIDDEN_1 = 128  #
N_HIDDEN_2 = 64  #
N_HIDDEN_3 = 32
N_INPUT = 2 * BLOCKLENGTH
N_CLASSES = 2**BLOCKLENGTH

# tf Graph input


# take 500 sample for each class
numoftrainingsamples = NUM_OF_CLASSES * 500
instanceforeachsampletraining = int(numoftrainingsamples / NUM_OF_CLASSES)
numoftestsamples = NUM_OF_CLASSES * 10000
instanceforeachsampletest = int(numoftestsamples / NUM_OF_CLASSES)

# Parameters

# set snr this system works on single snr now
snrdb = 2
SNR = 10 ** (snrdb / 10)
N0 = 1 / SNR
SIGMA = np.sqrt(N0 * 0.5)


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


def train_test_dataset():
    # All possible bit combinations for spesified length
    classbits = np.zeros((ALLSET, BLOCKLENGTH))
    for i in range(ALLSET):
        classbits[i, :] = np.array(
            list(np.binary_repr(i, width=BLOCKLENGTH)), dtype=np.uint8
        )

    # Encoded version of all possibilities
    fullencodedset = np.zeros((ALLSET, BLOCKLENGTH * 2))
    for i in range(ALLSET):
        fullencodedset[i, :] = encode133171(classbits[i, :])

    # Only important if we decide to train it using subset of codebook
    partialencodedset = np.zeros((NUM_OF_CLASSES, BLOCKLENGTH * 2))
    for i in range(NUM_OF_CLASSES):
        partialencodedset[i, :] = encode133171(classbits[i, :])

    trainingset = np.zeros((numoftrainingsamples, BLOCKLENGTH * 2))
    traininglabels = np.zeros(numoftrainingsamples)

    testset = np.zeros((numoftestsamples, BLOCKLENGTH * 2))
    testlabels = np.zeros(numoftestsamples)

    # generate training data and test data with spesific snr value
    for i in range(NUM_OF_CLASSES):
        for j in range(instanceforeachsampletraining):
            trainingset[instanceforeachsampletraining * i + j, :] = modulateAwgn(
                partialencodedset[i, :], SIGMA
            )
            traininglabels[instanceforeachsampletraining * i + j] = i

    for i in range(ALLSET):
        for j in range(instanceforeachsampletest):
            testset[instanceforeachsampletest * i + j, :] = modulateAwgn(
                fullencodedset[i, :], SIGMA
            )
            testlabels[instanceforeachsampletest * i + j] = i

    traininglabels = traininglabels.astype(int)
    testlabels = testlabels.astype(int)

    # create one hot encoding for labels
    onehottraininglabels = np.zeros([len(trainingset), N_CLASSES], dtype=np.float32)
    onehottestlabels = np.zeros([len(testset), N_CLASSES], dtype=np.float32)
    for i in range(len(trainingset)):
        onehottraininglabels[i, traininglabels[i]] = 1
        if i < len(testlabels):
            onehottestlabels[i, testlabels[i]] = 1

    trainingset = trainingset.astype(np.float16)
    testset = testset.astype(np.float16)
    return trainingset, onehottraininglabels, testset, onehottestlabels, classbits


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


def get_weights():
    # Store layers weight & bias
    weights = {
        "h1": tf.Variable(
            tf.truncated_normal(
                [N_INPUT, N_HIDDEN_1], stddev=math.sqrt(2.0 / (N_INPUT))
            )
        ),
        "h2": tf.Variable(
            tf.truncated_normal(
                [N_HIDDEN_1, N_HIDDEN_2], stddev=math.sqrt(2.0 / (N_HIDDEN_1))
            )
        ),
        "h3": tf.Variable(
            tf.truncated_normal(
                [N_HIDDEN_2, N_HIDDEN_3], stddev=math.sqrt(2.0 / (N_HIDDEN_2))
            )
        ),
        "out": tf.Variable(
            tf.truncated_normal(
                [N_HIDDEN_3, N_CLASSES], stddev=math.sqrt(2.0 / (N_HIDDEN_3))
            )
        ),
    }
    biases = {
        "b1": tf.Variable(tf.zeros([N_HIDDEN_1])),
        "b2": tf.Variable(tf.zeros([N_HIDDEN_2])),
        "b3": tf.Variable(tf.zeros([N_HIDDEN_3])),
        "out": tf.Variable(tf.zeros([N_CLASSES])),
    }
    return weights, biases


def run_network(trainingset, onehottraininglabels, testset, onehottestlabels):
    learning_rate = 0.0001
    training_epochs = 200
    batch_size = 256
    display_step = 1
    beta = 0.001
    weights, biases = get_weights()

    x = tf.placeholder("float", [None, N_INPUT])
    y = tf.placeholder("float", [None, N_CLASSES])

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

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
                batch_y = onehottraininglabels[i * batch_size : (i + 1) * batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print(
                    "Epoch:", "%04d" % (epoch + 1), "cost=", "{:.9f}".format(avg_cost)
                )
        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        g = tf.argmax(pred, 1)
        # Obtain prediction
        predictions = g.eval(feed_dict={x: testset})
        # Calculate accuracy this does not show correct result
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: testset, y: onehottestlabels}))
        return predictions


def calculate_error(testlabels, predictions, classbits):
    # Calculate frame error corrects if network guess whole bits correct
    packeterrnum = 0
    testlabels = testlabels.astype(int)
    for i in range(len(predictions)):
        if predictions[i] != testlabels[i]:
            packeterrnum = packeterrnum + 1

    frameerror = packeterrnum / len(predictions)
    print("accuracy = ", 1 - frameerror)

    # Calculate bit error here
    result = predictions
    resultbits = np.zeros((len(result), BLOCKLENGTH))
    for i in range(len(result)):
        resultbits[i, :] = np.array(
            list(np.binary_repr(result[i], width=BLOCKLENGTH)), dtype=np.uint8
        )

    biterrnum = 0
    for i in range(NUM_OF_CLASSES):
        arr = np.tile(classbits[i, :], (instanceforeachsampletest, 1))
        for j in range(instanceforeachsampletest):
            biterrnum = biterrnum + np.sum(
                np.logical_xor(
                    arr[j, :], resultbits[instanceforeachsampletest * i + j, :]
                )
            )

    ber = biterrnum / (numoftestsamples * BLOCKLENGTH)
    print("ber: ", ber)


def main():
    (
        trainingset,
        onehottraininglabels,
        testset,
        onehottestlabels,
        classbits,
    ) = train_test_dataset()
    predictions = run_network(
        trainingset, onehottraininglabels, testset, onehottestlabels
    )
    calculate_error(onehottestlabels, predictions, classbits)
