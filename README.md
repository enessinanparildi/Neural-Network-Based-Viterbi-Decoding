# Neural-Network-Based-Viterbi-Decoding
Decode convolutional codes used in communications efficiently by utilizing feed-forward and recurrent neural networks. Feed-forward and recurrent neural networks are implemented using Tensorflow. Implementation of convolutional coding and noisy channel are also included. 

This repository contains implementations of neural network-based Viterbi decoders for convolutional codes.

## Contents

- `neural_viterbi.py`: Main implementation of neural Viterbi decoder using feedforward neural networks
- `rnn_viterbi.py`: Experimental implementation using recurrent neural networks 
- `seq2seq_viterbi.py`: Experimental implementation using sequence-to-sequence models
- `bit_predictor.py`: Implementation that predicts bits individually

## Key Features

- Implements (7,5) and (133,171) convolutional encoders
- Trains neural networks to decode noisy encoded sequences
- Supports various SNR levels for training and testing
- Includes functions for generating training/test data
- Calculates bit error rate and frame error rate

## Usage

The main script to run is `neural_viterbi.py`. It allows configuring parameters like:

- Network architecture (number of layers, units)
- Training settings (learning rate, batch size, epochs)
- SNR levels
- Block length
- Training/test set sizes

To train a new model:

```python
train = True
model = create_model_graph()
run_graph(model)

testoldmodel = True
model = create_model_graph()
run_graph(model)
