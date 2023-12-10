# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 09:09:41 2017

@author: Fahrettin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 10:55:03 2017

@author: Enes
"""

import numpy as np
import tensorflow as tf
from random import randint
import math as math


# This version is trying to predict with a different kind of neural network called recurrant network and special kind of it
# sequence to sequeunce mapping , it is used in machine translation I thought in theory this can be utilized to decode, but in
# practice it has many problems maybe more work will make it work , I could not spend too much time on it ,
# this is final version of rnnviterbi.py
# for deatils refer to https://github.com/ematvey/tensorflow-seq2seq-tutorials input generation is same as others for other look
# links


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


def extractBits(arr):
    predictionbits = np.zeros((len(predictions), blocklength), dtype=np.uint8)

    for i in range(len(predictions)):
        bits = np.zeros(blocklength)
        val = [int(x) for x in list("{0:0b}".format(arr[i]))]
        bits[len(bits) :] = np.array(val)
        predictionbits[i, :] = bits

    return predictionbits


blocklength = 100


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


def generateInputVectors(length, number):
    randomtestset = np.random.randint(2, size=(number, length))
    return randomtestset


def hardCode(seq):
    seq[np.where(seq < 0)] = 0
    seq[np.where(seq > 0)] = 1
    return seq


def create_model_graph():
    tf.reset_default_graph()

    trainingsize = 100000
    testsize = 1000

    # Parameters
    learning_rate = 0.01
    training_epochs = 6
    batch_size = 100
    display_step = 1
    beta = 0.001

    snrdbvec = np.arange(0, 9)
    SNR = 10 ** (snrdbvec / 10)
    N0 = 1 / SNR
    sigma = np.sqrt(N0 * 0.5)

    chosentestsnrindex = 2

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    traininginfoseq = generateInputVectors(blocklength, trainingsize)
    trainingencodedseq = np.array(list(map(encode133171, traininginfoseq.tolist())))

    snrnumeachclass = int(trainingsize / len(snrdbvec))

    for i in range(trainingsize):
        trainingencodedseq[i, :] = modulateAwgn(
            trainingencodedseq[i, :], sigma[chosentestsnrindex]
        )

    traininginfoseq[np.where(traininginfoseq == 0)] = -1

    testinfoseq = generateInputVectors(blocklength, testsize)
    testcodedseq = np.array(list(map(encode133171, testinfoseq.tolist())))
    for i in range(testsize):
        testcodedseq[i, :] = modulateAwgn(testcodedseq[i, :], sigma[chosentestsnrindex])

    testinfoseq[np.where(testinfoseq == 0)] = -1

    PAD = 0
    EOS = 1

    vocab_size = 1

    encoder_hidden_units = 20
    decoder_hidden_units = encoder_hidden_units

    encoder_inputs = tf.placeholder(
        shape=(2 * blocklength, None), dtype=tf.float32, name="encoder_inputs"
    )
    decoder_targets = tf.placeholder(
        shape=(2 * blocklength, None), dtype=tf.float32, name="decoder_targets"
    )
    decoder_inputs = tf.placeholder(
        shape=(2 * blocklength, None), dtype=tf.float32, name="decoder_inputs"
    )

    decoder_targets = tf.expand_dims(decoder_targets, 2)
    encoder_inputs = tf.expand_dims(encoder_inputs, 2)
    decoder_inputs = tf.expand_dims(decoder_inputs, 2)

    encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
        encoder_cell,
        encoder_inputs,
        dtype=tf.float32,
        time_major=True,
    )

    del encoder_outputs

    decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

    decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
        decoder_cell,
        decoder_inputs,
        initial_state=encoder_final_state,
        dtype=tf.float32,
        time_major=True,
        scope="plain_decoder",
    )

    decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)

    decoder_prediction = tf.argmax(decoder_logits, 2)

    loss = tf.reduce_mean(tf.squared_difference(decoder_targets, decoder_logits))

    train_op = tf.train.AdamOptimizer().minimize(loss)

    batch_size = 1000


def next_feed_training(batch_size, epochind):
    encoder_inputs_ = trainingencodedseq[
        epochind * batch_size : (epochind + 1) * batch_size, :
    ]
    decoder_targets_ = np.append(
        traininginfoseq[epochind * batch_size : (epochind + 1) * batch_size, :],
        np.tile(np.concatenate(([EOS], np.zeros(99))), (1000, 1)),
        axis=1,
    )
    decoder_inputs_ = np.append(
        np.tile([EOS], (1000, 1)),
        traininginfoseq[epochind * batch_size : (epochind + 1) * batch_size, :],
        axis=1,
    )
    decoder_inputs_ = np.append(decoder_inputs_, np.zeros((1000, 99)), axis=1)

    encoder_inputs_ = np.transpose(encoder_inputs_)
    encoder_inputs_ = np.expand_dims(encoder_inputs_, axis=2)
    decoder_targets_ = np.transpose(decoder_targets_)
    decoder_targets_ = np.expand_dims(decoder_targets_, axis=2)
    decoder_inputs_ = np.transpose(decoder_inputs_)
    decoder_inputs_ = np.expand_dims(decoder_inputs_, axis=2)

    return {
        encoder_inputs: encoder_inputs_,
        decoder_inputs: decoder_inputs_,
        decoder_targets: decoder_targets_,
    }


def next_feed_test(batch_size, epochind):
    encoder_inputs_ = testcodedseq[
        epochind * batch_size : (epochind + 1) * batch_size, :
    ]
    decoder_targets_ = np.append(
        testinfoseq[epochind * batch_size : (epochind + 1) * batch_size, :],
        np.tile(np.concatenate(([EOS], np.zeros(99))), (1000, 1)),
        axis=1,
    )
    decoder_inputs_ = np.append(
        np.tile([EOS], (1000, 1)),
        testinfoseq[epochind * batch_size : (epochind + 1) * batch_size, :],
        axis=1,
    )
    decoder_inputs_ = np.append(decoder_inputs_, np.zeros((1000, 99)), axis=1)

    encoder_inputs_ = np.transpose(encoder_inputs_)
    encoder_inputs_ = np.expand_dims(encoder_inputs_, axis=2)
    decoder_targets_ = np.transpose(decoder_targets_)
    decoder_targets_ = np.expand_dims(decoder_targets_, axis=2)
    decoder_inputs_ = np.transpose(decoder_inputs_)
    decoder_inputs_ = np.expand_dims(decoder_inputs_, axis=2)

    return {
        encoder_inputs: encoder_inputs_,
        decoder_inputs: decoder_inputs_,
        decoder_targets: decoder_targets_,
    }


def run_graph():
    trainingpredictedset = np.zeros((trainingsize, blocklength))
    sess.run(tf.global_variables_initializer())
    epochnum = 5
    batches_in_epoch = int(trainingsize / batch_size)
    max_batches = batches_in_epoch * epochnum
    batchindex = 0
    ind = 0
    loss_track = []
    try:
        for batch in range(max_batches):
            fd = next_feed_training(batch_size, batchindex)
            _, l = sess.run([train_op, loss], fd)
            loss_track.append(l)
            batchindex = batchindex + 1
            if batch == 0 or batchindex - batches_in_epoch == 0:
                batchindex = 0
                print("batch {}".format(batch))
                print("  minibatch loss: {}".format(sess.run(loss, fd)))
                predict_ = sess.run(decoder_prediction, fd)
                ind = ind + 1
                for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                    print("  sample {}:".format(i + 1))
                    print("    input     > {}".format(inp))
                    print("    predicted > {}".format(pred))
                    if i >= 2:
                        break
                print()
        testdata = next_feed_test(testsize, 0)
        testprediction = sess.run(decoder_prediction, testdata)

    except KeyboardInterrupt:
        print("training interrupted")

    import matplotlib.pyplot as plt

    plt.plot(loss_track)
    print(
        "loss {:.4f} after {} examples (batch_size={})".format(
            loss_track[-1], len(loss_track) * batch_size, batch_size
        )
    )

    testprediction = np.transpose(testprediction)
    testprediction = testprediction[:, 0:100]
    biterrnum = 0
    for j in range(len(testprediction)):
        biterrnum = biterrnum + np.sum(
            np.logical_xor(testprediction[j, :], testinfoseq[j, :])
        )

    ber = biterrnum / (len(testprediction) * blocklength)

    print("ber: ", ber)
    print("snrdb : ", snrdbvec[chosentestsnrindex])
