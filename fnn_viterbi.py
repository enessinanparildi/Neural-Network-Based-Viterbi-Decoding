#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Neural Network-Based Viterbi Decoder using Feedforward Neural Networks.

This module implements a neural network decoder for convolutional codes using
multiclass classification. The network predicts entire information sequences
at once, treating decoding as a classification problem where each class
represents a possible bit sequence.

Author: Enes, Fahrettin
Date: July 10, 2017
Improved: December 2024
"""

from typing import Tuple, Dict, Optional
import math
import tensorflow as tf
import numpy as np
from random import randint


# ==============================================================================
# CONFIGURATION AND GLOBAL CONSTANTS
# ==============================================================================

class DecoderConfig:
    """Configuration container for the neural Viterbi decoder."""
    
    def __init__(
        self,
        block_length: int = 7,
        rate: int = 2,
        snr_db: float = 2.0,
        n_hidden_1: int = 128,
        n_hidden_2: int = 64,
        n_hidden_3: int = 32,
        learning_rate: float = 0.0001,
        training_epochs: int = 200,
        batch_size: int = 256,
        beta_regularization: float = 0.001,
        samples_per_class_train: int = 500,
        samples_per_class_test: int = 10000,
    ):
        """
        Initialize decoder configuration.
        
        Args:
            block_length: Number of information bits per block
            rate: Code rate (output bits per input bit)
            snr_db: Signal-to-noise ratio in decibels
            n_hidden_1: Number of neurons in first hidden layer
            n_hidden_2: Number of neurons in second hidden layer
            n_hidden_3: Number of neurons in third hidden layer
            learning_rate: Learning rate for Adam optimizer
            training_epochs: Number of training epochs
            batch_size: Mini-batch size for training
            beta_regularization: L2 regularization coefficient
            samples_per_class_train: Training samples per class
            samples_per_class_test: Test samples per class
        """
        # Block and encoding parameters
        self.block_length = block_length
        self.rate = rate
        self.all_classes = 2 ** block_length
        self.num_classes = self.all_classes
        
        # Network architecture
        self.n_input = rate * block_length
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.n_hidden_3 = n_hidden_3
        self.n_classes = self.all_classes
        
        # Training parameters
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.beta_regularization = beta_regularization
        self.display_step = 1
        
        # Dataset sizes
        self.num_training_samples = self.num_classes * samples_per_class_train
        self.samples_per_class_train = samples_per_class_train
        self.num_test_samples = self.num_classes * samples_per_class_test
        self.samples_per_class_test = samples_per_class_test
        
        # Channel parameters
        self.snr_db = snr_db
        self.snr = 10 ** (snr_db / 10.0)
        self.n0 = 1.0 / self.snr
        self.sigma = np.sqrt(self.n0 * 0.5)


# ==============================================================================
# DATA GENERATION AND ENCODING
# ==============================================================================

def generate_info_sequence(block_length: int) -> np.ndarray:
    """
    Generate a random binary information sequence.
    
    Args:
        block_length: Length of the information sequence
        
    Returns:
        Binary array of shape (block_length,) with random 0s and 1s
        
    Example:
        >>> seq = generate_info_sequence(7)
        >>> seq.shape
        (7,)
        >>> all(bit in [0, 1] for bit in seq)
        True
    """
    info_seq = np.zeros(block_length, dtype=np.int32)
    for i in range(block_length):
        info_seq[i] = randint(0, 1)
    return info_seq


def encode_57(info_seq: np.ndarray) -> np.ndarray:
    """
    Encode information sequence using (7,5) convolutional encoder.
    
    This is a rate-1/2 convolutional encoder with constraint length 3.
    Generator polynomials: G1 = 7 (octal), G2 = 5 (octal)
    
    Args:
        info_seq: Binary information sequence
        
    Returns:
        Encoded sequence with length 2 * len(info_seq)
        
    Note:
        This encoder has memory of 2 bits and uses modulo-2 addition
    """
    length = len(info_seq)
    encoded_seq = np.zeros(length * 2, dtype=np.int32)
    
    # First two output pairs (special cases)
    encoded_seq[0] = info_seq[0]
    encoded_seq[1] = (info_seq[0] + info_seq[1]) % 2
    encoded_seq[2] = info_seq[1]
    encoded_seq[3] = (info_seq[1] + info_seq[0]) % 2
    
    # Remaining output pairs (general case)
    for i in range(2, length):
        encoded_seq[2 * i] = (info_seq[i] + info_seq[i - 2]) % 2
        encoded_seq[2 * i + 1] = (
            info_seq[i] + info_seq[i - 1] + info_seq[i - 2]
        ) % 2
    
    return encoded_seq


def encode_133171(bits: np.ndarray) -> np.ndarray:
    """
    Encode information sequence using (133,171) convolutional encoder.
    
    This is a rate-1/2 convolutional encoder with constraint length 7.
    Generator polynomials: G1 = 133 (octal), G2 = 171 (octal)
    
    Args:
        bits: Binary information sequence
        
    Returns:
        Encoded sequence with length 2 * len(bits)
        
    Note:
        This encoder has memory of 6 bits (constraint length - 1)
        and provides better error correction than (7,5) encoder
    """
    length = len(bits)
    encoded_bits = np.zeros(length * 2, dtype=np.int32)
    
    # Initialize first 6 encoded pairs based on constraint length
    # Each pair depends on different combinations of previous bits
    
    # Bit 0 encoding
    encoded_bits[0] = bits[0] % 2
    encoded_bits[1] = bits[0] % 2
    
    # Bit 1 encoding
    encoded_bits[2] = bits[1] % 2
    encoded_bits[3] = (bits[1] + bits[0]) % 2
    
    # Bit 2 encoding
    encoded_bits[4] = (bits[2] + bits[0]) % 2
    encoded_bits[5] = (bits[2] + bits[1] + bits[0]) % 2
    
    # Bit 3 encoding
    encoded_bits[6] = (bits[3] + bits[1] + bits[0]) % 2
    encoded_bits[7] = (bits[3] + bits[2] + bits[1] + bits[0]) % 2
    
    # Bit 4 encoding
    encoded_bits[8] = (bits[4] + bits[2] + bits[1]) % 2
    encoded_bits[9] = (bits[4] + bits[3] + bits[2] + bits[1]) % 2
    
    # Bit 5 encoding
    encoded_bits[10] = (bits[5] + bits[3] + bits[2] + bits[0]) % 2
    encoded_bits[11] = (bits[5] + bits[4] + bits[3] + bits[2]) % 2
    
    # General case for remaining bits (steady state)
    # Uses full constraint length of 7
    for i in range(6, length):
        # G1 = 133 (octal) = 1011011 (binary)
        encoded_bits[2 * i] = (
            bits[i] + bits[i - 2] + bits[i - 3] + bits[i - 5] + bits[i - 6]
        ) % 2
        
        # G2 = 171 (octal) = 1111001 (binary)
        encoded_bits[2 * i + 1] = (
            bits[i] + bits[i - 1] + bits[i - 2] + bits[i - 3] + bits[i - 6]
        ) % 2
    
    return encoded_bits


# ==============================================================================
# CHANNEL SIMULATION
# ==============================================================================

def modulate_awgn(encoded_seq: np.ndarray, sigma: float) -> np.ndarray:
    """
    Modulate encoded sequence and add Additive White Gaussian Noise (AWGN).
    
    Performs BPSK modulation (0 -> -1, 1 -> +1) and adds Gaussian noise
    to simulate a noisy communication channel.
    
    Args:
        encoded_seq: Binary encoded sequence (0s and 1s)
        sigma: Standard deviation of Gaussian noise
        
    Returns:
        Noisy real-valued sequence after BPSK modulation and AWGN
        
    Note:
        - BPSK (Binary Phase Shift Keying): Simple modulation scheme
        - 0 bits are mapped to -1, 1 bits are mapped to +1
        - Gaussian noise with variance sigma^2 is added to each symbol
    """
    # Create a copy to avoid modifying original array
    modulated_seq = encoded_seq.copy().astype(np.float32)
    
    # BPSK modulation: 0 -> -1, 1 -> +1
    modulated_seq[np.where(modulated_seq == 0)] = -1
    
    # Add Gaussian noise with mean 0 and standard deviation sigma
    noise = np.random.normal(0, sigma, len(modulated_seq))
    distorted_seq = modulated_seq + noise
    
    return distorted_seq


# ==============================================================================
# DATASET PREPARATION
# ==============================================================================

def generate_class_bits(block_length: int) -> np.ndarray:
    """
    Generate all possible bit combinations for given block length.
    
    Args:
        block_length: Number of bits per block
        
    Returns:
        Array of shape (2^block_length, block_length) containing all
        possible binary sequences
        
    Example:
        >>> bits = generate_class_bits(3)
        >>> bits.shape
        (8, 3)
        >>> bits[0]  # First combination: [0, 0, 0]
        >>> bits[7]  # Last combination: [1, 1, 1]
    """
    num_classes = 2 ** block_length
    class_bits = np.zeros((num_classes, block_length), dtype=np.uint8)
    
    for i in range(num_classes):
        # Convert decimal number to binary string with fixed width
        binary_str = np.binary_repr(i, width=block_length)
        # Convert string to array of integers
        class_bits[i, :] = np.array(list(binary_str), dtype=np.uint8)
    
    return class_bits


def create_one_hot_labels(
    labels: np.ndarray,
    num_classes: int
) -> np.ndarray:
    """
    Convert class labels to one-hot encoded format.
    
    Args:
        labels: Array of class indices
        num_classes: Total number of classes
        
    Returns:
        One-hot encoded array of shape (len(labels), num_classes)
        
    Example:
        >>> labels = np.array([0, 2, 1])
        >>> one_hot = create_one_hot_labels(labels, 3)
        >>> one_hot[0]  # [1, 0, 0]
        >>> one_hot[1]  # [0, 0, 1]
    """
    num_samples = len(labels)
    one_hot = np.zeros((num_samples, num_classes), dtype=np.float32)
    
    for i in range(num_samples):
        one_hot[i, labels[i]] = 1.0
    
    return one_hot


def train_test_dataset(
    config: DecoderConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate complete training and test datasets.
    
    Creates datasets by:
    1. Generating all possible bit sequences (codebook)
    2. Encoding each sequence
    3. Adding noise at specified SNR
    4. Creating multiple noisy samples per class
    
    Args:
        config: Decoder configuration object
        
    Returns:
        Tuple containing:
        - training_set: Noisy encoded training samples
        - training_labels: One-hot encoded training labels
        - test_set: Noisy encoded test samples
        - test_labels: One-hot encoded test labels
        - class_bits: All possible bit combinations (codebook)
    """
    # Generate all possible bit combinations (codebook)
    class_bits = generate_class_bits(config.block_length)
    all_classes = len(class_bits)
    
    # Encode all possible sequences
    full_encoded_set = np.zeros(
        (all_classes, config.block_length * 2),
        dtype=np.float32
    )
    for i in range(all_classes):
        full_encoded_set[i, :] = encode_133171(class_bits[i, :])
    
    # Initialize training and test arrays
    training_set = np.zeros(
        (config.num_training_samples, config.block_length * 2),
        dtype=np.float32
    )
    training_labels = np.zeros(config.num_training_samples, dtype=np.int32)
    
    test_set = np.zeros(
        (config.num_test_samples, config.block_length * 2),
        dtype=np.float32
    )
    test_labels = np.zeros(config.num_test_samples, dtype=np.int32)
    
    # Generate training data
    # For each class, create multiple noisy samples
    for i in range(config.num_classes):
        for j in range(config.samples_per_class_train):
            sample_idx = config.samples_per_class_train * i + j
            # Add noise to encoded sequence
            training_set[sample_idx, :] = modulate_awgn(
                full_encoded_set[i, :],
                config.sigma
            )
            training_labels[sample_idx] = i
    
    # Generate test data
    for i in range(all_classes):
        for j in range(config.samples_per_class_test):
            sample_idx = config.samples_per_class_test * i + j
            # Add noise to encoded sequence
            test_set[sample_idx, :] = modulate_awgn(
                full_encoded_set[i, :],
                config.sigma
            )
            test_labels[sample_idx] = i
    
    # Convert labels to one-hot encoding
    one_hot_training_labels = create_one_hot_labels(
        training_labels,
        config.n_classes
    )
    one_hot_test_labels = create_one_hot_labels(
        test_labels,
        config.n_classes
    )
    
    return (
        training_set,
        one_hot_training_labels,
        test_set,
        one_hot_test_labels,
        class_bits
    )


# ==============================================================================
# NEURAL NETWORK MODEL
# ==============================================================================

def initialize_weights(config: DecoderConfig) -> Tuple[Dict, Dict]:
    """
    Initialize weights and biases for the neural network.
    
    Uses truncated normal initialization with Xavier scaling for better
    gradient flow during training.
    
    Args:
        config: Decoder configuration object
        
    Returns:
        Tuple of (weights_dict, biases_dict) containing TensorFlow variables
        
    Note:
        Xavier initialization: stddev = sqrt(2 / n_in)
        This helps prevent vanishing/exploding gradients
    """
    weights = {
        'h1': tf.Variable(
            tf.truncated_normal(
                [config.n_input, config.n_hidden_1],
                stddev=math.sqrt(2.0 / config.n_input)
            ),
            name='weights_h1'
        ),
        'h2': tf.Variable(
            tf.truncated_normal(
                [config.n_hidden_1, config.n_hidden_2],
                stddev=math.sqrt(2.0 / config.n_hidden_1)
            ),
            name='weights_h2'
        ),
        'h3': tf.Variable(
            tf.truncated_normal(
                [config.n_hidden_2, config.n_hidden_3],
                stddev=math.sqrt(2.0 / config.n_hidden_2)
            ),
            name='weights_h3'
        ),
        'out': tf.Variable(
            tf.truncated_normal(
                [config.n_hidden_3, config.n_classes],
                stddev=math.sqrt(2.0 / config.n_hidden_3)
            ),
            name='weights_out'
        ),
    }
    
    biases = {
        'b1': tf.Variable(tf.zeros([config.n_hidden_1]), name='bias_h1'),
        'b2': tf.Variable(tf.zeros([config.n_hidden_2]), name='bias_h2'),
        'b3': tf.Variable(tf.zeros([config.n_hidden_3]), name='bias_h3'),
        'out': tf.Variable(tf.zeros([config.n_classes]), name='bias_out'),
    }
    
    return weights, biases


def multilayer_perceptron(
    x: tf.Tensor,
    weights: Dict[str, tf.Variable],
    biases: Dict[str, tf.Variable]
) -> tf.Tensor:
    """
    Build a 3-layer feedforward neural network.
    
    Architecture:
        Input -> Hidden1 (tanh) -> Hidden2 (tanh) -> Hidden3 (tanh) -> Output (tanh)
    
    Args:
        x: Input tensor of shape [batch_size, n_input]
        weights: Dictionary of weight variables
        biases: Dictionary of bias variables
        
    Returns:
        Output tensor of shape [batch_size, n_classes]
        
    Note:
        - Uses tanh activation for better gradient flow
        - No activation on output layer (logits for softmax)
    """
    # Hidden layer 1: W1*x + b1, then tanh activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)
    
    # Hidden layer 2: W2*h1 + b2, then tanh activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.tanh(layer_2)
    
    # Hidden layer 3: W3*h2 + b3, then tanh activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.tanh(layer_3)
    
    # Output layer: W_out*h3 + b_out, then tanh activation
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    out_layer = tf.nn.tanh(out_layer)
    
    return out_layer


# ==============================================================================
# TRAINING AND EVALUATION
# ==============================================================================

def run_network(
    config: DecoderConfig,
    training_set: np.ndarray,
    training_labels: np.ndarray,
    test_set: np.ndarray,
    test_labels: np.ndarray
) -> np.ndarray:
    """
    Train and evaluate the neural network decoder.
    
    Args:
        config: Decoder configuration
        training_set: Training input data
        training_labels: Training labels (one-hot encoded)
        test_set: Test input data
        test_labels: Test labels (one-hot encoded)
        
    Returns:
        Predictions on test set
    """
    # Initialize weights and biases
    weights, biases = initialize_weights(config)
    
    # Create placeholders for input and output
    x = tf.placeholder("float", [None, config.n_input], name='input')
    y = tf.placeholder("float", [None, config.n_classes], name='output')
    
    # Construct model
    pred = multilayer_perceptron(x, weights, biases)
    
    # Define loss function
    # Cross-entropy loss for multiclass classification
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
    )
    
    # Add L2 regularization to prevent overfitting
    regularizer = (
        tf.nn.l2_loss(weights['h1']) +
        tf.nn.l2_loss(weights['h2']) +
        tf.nn.l2_loss(weights['h3'])
    )
    cost = tf.reduce_mean(cost + config.beta_regularization * regularizer)
    
    # Define optimizer
    optimizer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate
    ).minimize(cost)
    
    # Initialize variables
    init = tf.global_variables_initializer()
    
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        
        # Training cycle
        for epoch in range(config.training_epochs):
            avg_cost = 0.0
            total_batch = int(config.num_training_samples / config.batch_size)
            
            # Loop over all batches
            for i in range(total_batch):
                batch_x = training_set[
                    i * config.batch_size : (i + 1) * config.batch_size, :
                ]
                batch_y = training_labels[
                    i * config.batch_size : (i + 1) * config.batch_size
                ]
                
                # Run optimization and compute loss
                _, c = sess.run(
                    [optimizer, cost],
                    feed_dict={x: batch_x, y: batch_y}
                )
                
                # Compute average loss
                avg_cost += c / total_batch
            
            # Display logs per epoch step
            if epoch % config.display_step == 0:
                print(
                    f"Epoch: {epoch + 1:04d} | "
                    f"cost = {avg_cost:.9f}"
                )
        
        print("Optimization Finished!")
        
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        predictions_tensor = tf.argmax(pred, 1)
        
        # Get predictions
        predictions = predictions_tensor.eval(feed_dict={x: test_set})
        
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(f"Accuracy: {accuracy.eval({x: test_set, y: test_labels}):.4f}")
        
        return predictions


def calculate_error(
    test_labels: np.ndarray,
    predictions: np.ndarray,
    class_bits: np.ndarray,
    config: DecoderConfig
) -> None:
    """
    Calculate and display bit error rate (BER) and frame error rate (FER).
    
    Args:
        test_labels: Ground truth labels (one-hot encoded)
        predictions: Predicted class labels
        class_bits: All possible bit combinations
        config: Decoder configuration
    """
    # Convert one-hot labels to class indices
    test_labels_idx = np.argmax(test_labels, axis=1).astype(int)
    
    # Calculate Frame Error Rate (FER)
    # A frame error occurs when prediction doesn't match true label
    frame_errors = np.sum(predictions != test_labels_idx)
    fer = frame_errors / len(predictions)
    accuracy = 1.0 - fer
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Frame Error Rate: {fer:.6f}")
    
    # Calculate Bit Error Rate (BER)
    # Convert predictions to bit representations
    result_bits = np.zeros((len(predictions), config.block_length), dtype=np.uint8)
    for i in range(len(predictions)):
        binary_str = np.binary_repr(predictions[i], width=config.block_length)
        result_bits[i, :] = np.array(list(binary_str), dtype=np.uint8)
    
    # Count bit errors
    bit_errors = 0
    for i in range(config.num_classes):
        # Get all samples for this class
        start_idx = config.samples_per_class_test * i
        end_idx = start_idx + config.samples_per_class_test
        
        # True bits for this class
        true_bits = class_bits[i, :]
        
        # Compare with predictions
        for j in range(config.samples_per_class_test):
            pred_bits = result_bits[start_idx + j, :]
            # XOR gives 1 for differing bits
            bit_errors += np.sum(np.logical_xor(true_bits, pred_bits))
    
    # Calculate BER
    total_bits = config.num_test_samples * config.block_length
    ber = bit_errors / total_bits
    
    print(f"Bit Error Rate: {ber:.6f}")
    print(f"Total Bit Errors: {bit_errors}/{total_bits}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """
    Main function to run the neural Viterbi decoder.
    
    Workflow:
    1. Initialize configuration
    2. Generate training and test datasets
    3. Build and train neural network
    4. Evaluate performance
    """
    print("=" * 70)
    print("Neural Network-Based Viterbi Decoder")
    print("=" * 70)
    
    # Initialize configuration
    config = DecoderConfig(
        block_length=7,
        rate=2,
        snr_db=2.0,
        n_hidden_1=128,
        n_hidden_2=64,
        n_hidden_3=32,
        learning_rate=0.0001,
        training_epochs=200,
        batch_size=256,
        beta_regularization=0.001,
        samples_per_class_train=500,
        samples_per_class_test=10000,
    )
    
    print("\nConfiguration:")
    print(f"  Block Length: {config.block_length}")
    print(f"  SNR: {config.snr_db} dB")
    print(f"  Number of Classes: {config.num_classes}")
    print(f"  Training Samples: {config.num_training_samples}")
    print(f"  Test Samples: {config.num_test_samples}")
    print(f"  Network: {config.n_input} -> {config.n_hidden_1} -> "
          f"{config.n_hidden_2} -> {config.n_hidden_3} -> {config.n_classes}")
    
    # Generate datasets
    print("\nGenerating datasets...")
    (
        training_set,
        training_labels,
        test_set,
        test_labels,
        class_bits,
    ) = train_test_dataset(config)
    
    print(f"Training set shape: {training_set.shape}")
    print(f"Test set shape: {test_set.shape}")
    
    # Train and evaluate
    print("\nTraining neural network...")
    predictions = run_network(
        config,
        training_set,
        training_labels,
        test_set,
        test_labels
    )
    
    # Calculate errors
    print("\nEvaluating performance...")
    calculate_error(test_labels, predictions, class_bits, config)
    
    print("\n" + "=" * 70)
    print("Decoding complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
