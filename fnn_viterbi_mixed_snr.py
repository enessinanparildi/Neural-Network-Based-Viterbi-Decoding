#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mixed SNR Neural Viterbi Decoder using Feedforward Neural Networks.

This module implements a neural network decoder trained on multiple SNR values
simultaneously. By training across a range of SNR conditions, the decoder
learns to be robust to varying noise levels, making it more practical for
real-world scenarios where channel conditions fluctuate.

Key Features:
    - Multi-SNR training for robustness
    - Supports both full and random test sets
    - Dropout regularization option
    - Configurable network architecture

Author: Enes
Date: July 12, 2017
Improved: December 2024
"""

from typing import Tuple, Optional, Dict, List
import math
import tensorflow as tf
import numpy as np
from random import randint


# ==============================================================================
# CONFIGURATION AND GLOBAL CONSTANTS
# ==============================================================================

class MixedSNRDecoderConfig:
    """Configuration container for mixed SNR neural decoder."""
    
    def __init__(
        self,
        block_length: int = 7,
        rate: int = 2,
        n_hidden_1: int = 128,
        n_hidden_2: int = 64,
        n_hidden_3: int = 32,
        learning_rate: float = 0.0001,
        training_epochs: int = 300,
        batch_size: int = 256,
        beta_regularization: float = 0.001,
        samples_per_class_train: int = 1000,
        samples_per_class_test: int = 10000,
        snr_db_range: Tuple[int, int] = (-2, 10),
        test_snr_index: int = 0,
        use_dropout: bool = False,
        dropout_keep_prob: float = 0.5,
        use_random_test_set: bool = False,
        random_test_set_size: int = 5000,
    ):
        """
        Initialize mixed SNR decoder configuration.
        
        Args:
            block_length: Number of information bits per block
            rate: Code rate (output bits per input bit)
            n_hidden_1: Number of neurons in first hidden layer
            n_hidden_2: Number of neurons in second hidden layer
            n_hidden_3: Number of neurons in third hidden layer
            learning_rate: Learning rate for Adam optimizer
            training_epochs: Number of training epochs
            batch_size: Mini-batch size for training
            beta_regularization: L2 regularization coefficient
            samples_per_class_train: Training samples per class
            samples_per_class_test: Test samples per class
            snr_db_range: Range of SNR values for training (min, max)
            test_snr_index: Index of SNR value for testing
            use_dropout: Whether to use dropout regularization
            dropout_keep_prob: Probability of keeping neurons (if using dropout)
            use_random_test_set: Use random test set instead of exhaustive
            random_test_set_size: Size of random test set (if used)
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
        
        # Dropout parameters
        self.use_dropout = use_dropout
        self.dropout_keep_prob = dropout_keep_prob
        
        # Dataset configuration
        self.samples_per_class_train = samples_per_class_train
        self.samples_per_class_test = samples_per_class_test
        self.num_training_samples = self.num_classes * samples_per_class_train
        self.num_test_samples = self.num_classes * samples_per_class_test
        
        # SNR parameters
        snr_db_vec = np.arange(snr_db_range[0], snr_db_range[1])
        self.snr_db_vec = snr_db_vec
        self.snr_vec = 10 ** (snr_db_vec / 10.0)
        self.n0_vec = 1.0 / self.snr_vec
        self.sigma_vec = np.sqrt(self.n0_vec * 0.5)
        self.test_snr_index = test_snr_index
        
        # Calculate samples per SNR for training
        # Distribute training samples evenly across all SNR values
        self.samples_per_snr = int(samples_per_class_train / len(snr_db_vec))
        
        # Random test set parameters
        self.use_random_test_set = use_random_test_set
        self.random_test_set_size = random_test_set_size
        self.random_test_length = block_length * random_test_set_size


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
    """
    info_seq = np.zeros(block_length, dtype=np.int32)
    for i in range(block_length):
        info_seq[i] = randint(0, 1)
    return info_seq


def encode_57(info_seq: np.ndarray) -> np.ndarray:
    """
    Encode information sequence using (7,5) convolutional encoder.
    
    This is a rate-1/2 encoder with constraint length 3.
    Generator polynomials: G1 = 7 (octal), G2 = 5 (octal)
    
    Args:
        info_seq: Binary information sequence
        
    Returns:
        Encoded sequence with length 2 * len(info_seq)
    """
    length = len(info_seq)
    encoded_seq = np.zeros(length * 2, dtype=np.int32)
    
    # Initialize encoding
    encoded_seq[0] = info_seq[0]
    encoded_seq[1] = (info_seq[0] + info_seq[1]) % 2
    encoded_seq[2] = info_seq[1]
    encoded_seq[3] = (info_seq[1] + info_seq[0]) % 2
    
    # Steady-state encoding
    for i in range(2, length):
        encoded_seq[2 * i] = (info_seq[i] + info_seq[i - 2]) % 2
        encoded_seq[2 * i + 1] = (
            info_seq[i] + info_seq[i - 1] + info_seq[i - 2]
        ) % 2
    
    return encoded_seq


def encode_133171(bits: np.ndarray) -> np.ndarray:
    """
    Encode information sequence using (133,171) convolutional encoder.
    
    This is a rate-1/2 encoder with constraint length 7.
    Generator polynomials: G1 = 133 (octal), G2 = 171 (octal)
    
    Args:
        bits: Binary information sequence
        
    Returns:
        Encoded sequence with length 2 * len(bits)
    """
    length = len(bits)
    encoded_bits = np.zeros(length * 2, dtype=np.int32)
    
    # Initialize encoding for first 6 bits
    encoded_bits[0] = bits[0] % 2
    encoded_bits[1] = bits[0] % 2
    
    encoded_bits[2] = bits[1] % 2
    encoded_bits[3] = (bits[1] + bits[0]) % 2
    
    encoded_bits[4] = (bits[2] + bits[0]) % 2
    encoded_bits[5] = (bits[2] + bits[1] + bits[0]) % 2
    
    encoded_bits[6] = (bits[3] + bits[1] + bits[0]) % 2
    encoded_bits[7] = (bits[3] + bits[2] + bits[1] + bits[0]) % 2
    
    encoded_bits[8] = (bits[4] + bits[2] + bits[1]) % 2
    encoded_bits[9] = (bits[4] + bits[3] + bits[2] + bits[1]) % 2
    
    encoded_bits[10] = (bits[5] + bits[3] + bits[2] + bits[0]) % 2
    encoded_bits[11] = (bits[5] + bits[4] + bits[3] + bits[2]) % 2
    
    # Steady-state encoding
    for i in range(6, length):
        # G1 = 133 (octal) = 1011011 (binary)
        encoded_bits[2 * i] = (
            bits[i] + bits[i - 2] + bits[i - 3] + 
            bits[i - 5] + bits[i - 6]
        ) % 2
        
        # G2 = 171 (octal) = 1111001 (binary)
        encoded_bits[2 * i + 1] = (
            bits[i] + bits[i - 1] + bits[i - 2] + 
            bits[i - 3] + bits[i - 6]
        ) % 2
    
    return encoded_bits


# ==============================================================================
# CHANNEL SIMULATION
# ==============================================================================

def modulate_awgn(encoded_seq: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply BPSK modulation and add Additive White Gaussian Noise (AWGN).
    
    Performs BPSK modulation (0 -> -1, 1 -> +1) and adds Gaussian noise
    to simulate a noisy communication channel.
    
    Args:
        encoded_seq: Binary encoded sequence (0s and 1s)
        sigma: Standard deviation of Gaussian noise
        
    Returns:
        Noisy real-valued sequence after BPSK modulation and AWGN
    """
    # Create copy and apply BPSK modulation
    modulated_seq = encoded_seq.copy().astype(np.float32)
    modulated_seq[np.where(modulated_seq == 0)] = -1
    
    # Add Gaussian noise
    noise = np.random.normal(0, sigma, len(modulated_seq))
    distorted_seq = modulated_seq + noise
    
    return distorted_seq


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def binary_to_decimal(binary_array: np.ndarray) -> int:
    """
    Convert binary array to decimal number.
    
    Args:
        binary_array: Binary array (MSB first)
        
    Returns:
        Decimal representation
        
    Example:
        >>> binary_to_decimal(np.array([1, 0, 1]))
        5
    """
    coefficients = 2 ** np.arange(binary_array.shape[0])
    value = binary_array.dot(coefficients)
    return int(value)


def extract_bits_single(decimal_value: int, block_length: int) -> np.ndarray:
    """
    Convert decimal value to binary array.
    
    Args:
        decimal_value: Decimal number to convert
        block_length: Length of binary array
        
    Returns:
        Binary array of specified length
        
    Example:
        >>> extract_bits_single(5, 7)
        array([0, 0, 0, 0, 1, 0, 1])
    """
    bits = np.zeros(block_length, dtype=np.uint8)
    binary_str = format(decimal_value, f'0{block_length}b')
    bits[:len(binary_str)] = [int(b) for b in binary_str]
    return bits


def calculate_hamming_distance_decimal(
    val1: int,
    val2: int,
    block_length: int
) -> int:
    """
    Calculate Hamming distance between two decimal values.
    
    The Hamming distance is the number of bit positions in which
    the two values differ.
    
    Args:
        val1: First decimal value
        val2: Second decimal value
        block_length: Number of bits to compare
        
    Returns:
        Number of differing bits
        
    Example:
        >>> calculate_hamming_distance_decimal(5, 3, 3)  # 101 vs 011
        2
    """
    # Use XOR to find differing bits, then count them
    xor_result = val1 ^ val2
    return bin(xor_result).count('1')


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
    """
    num_classes = 2 ** block_length
    class_bits = np.zeros((num_classes, block_length), dtype=np.uint8)
    
    for i in range(num_classes):
        binary_str = np.binary_repr(i, width=block_length)
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
    """
    num_samples = len(labels)
    one_hot = np.zeros((num_samples, num_classes), dtype=np.float16)
    
    for i in range(num_samples):
        one_hot[i, labels[i]] = 1.0
    
    return one_hot


def get_dataset(config: MixedSNRDecoderConfig) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
    np.ndarray, np.ndarray
]:
    """
    Generate training and test datasets with mixed SNR values.
    
    This function creates a robust training set by including samples
    at multiple SNR values, allowing the network to learn noise-resistant
    features.
    
    Args:
        config: Mixed SNR decoder configuration
        
    Returns:
        Tuple containing:
        - training_set: Noisy encoded training samples
        - test_set: Noisy encoded test samples (single SNR)
        - training_labels: One-hot encoded training labels
        - random_test_labels: One-hot labels for random test set
        - random_test_set: Random test samples
        - class_bits: All possible bit combinations
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
    
    # Initialize training set
    training_set = np.zeros(
        (config.num_training_samples, config.block_length * 2),
        dtype=np.float32
    )
    training_labels = np.zeros(config.num_training_samples, dtype=np.int32)
    
    # Initialize standard test set
    test_set = np.zeros(
        (config.num_test_samples, config.block_length * 2),
        dtype=np.float32
    )
    test_labels = np.zeros(config.num_test_samples, dtype=np.int32)
    
    # Generate training data with multiple SNR values
    # This is the key feature: training across different noise levels
    for class_idx in range(all_classes):
        for snr_idx in range(len(config.snr_db_vec)):
            for sample_idx in range(config.samples_per_snr):
                # Calculate global sample index
                global_idx = (
                    config.samples_per_class_train * class_idx +
                    config.samples_per_snr * snr_idx +
                    sample_idx
                )
                
                # Add noise at specific SNR
                training_set[global_idx, :] = modulate_awgn(
                    full_encoded_set[class_idx, :],
                    config.sigma_vec[snr_idx]
                )
        
        # Set labels for all samples of this class
        start_idx = class_idx * config.samples_per_class_train
        end_idx = (class_idx + 1) * config.samples_per_class_train
        training_labels[start_idx:end_idx] = class_idx
    
    # Generate test data with single SNR value
    test_sigma = config.sigma_vec[config.test_snr_index]
    for class_idx in range(all_classes):
        for sample_idx in range(config.samples_per_class_test):
            global_idx = config.samples_per_class_test * class_idx + sample_idx
            test_set[global_idx, :] = modulate_awgn(
                full_encoded_set[class_idx, :],
                test_sigma
            )
            test_labels[global_idx] = class_idx
    
    # Generate random test set (if enabled)
    # This provides additional evaluation on unseen sequences
    random_test_bits = np.random.randint(
        2,
        size=config.random_test_length
    )
    random_test_encoded = encode_133171(random_test_bits)
    random_test_encoded = modulate_awgn(random_test_encoded, test_sigma)
    
    # Reshape random test set
    random_test_set = random_test_encoded.reshape(
        config.random_test_set_size,
        config.block_length * 2
    )
    
    # Convert random test bits to decimal labels
    random_test_bits_reshaped = random_test_bits.reshape(
        config.random_test_set_size,
        config.block_length
    )
    random_test_labels_decimal = np.zeros(
        config.random_test_set_size,
        dtype=np.uint16
    )
    for i in range(config.random_test_set_size):
        random_test_labels_decimal[i] = binary_to_decimal(
            random_test_bits_reshaped[i, :]
        )
    
    # Shuffle test set for better evaluation
    shuffle_indices = np.random.permutation(len(test_labels))
    test_set = test_set[shuffle_indices, :]
    test_labels = test_labels[shuffle_indices]
    
    # Create one-hot encodings
    training_labels_onehot = create_one_hot_labels(
        training_labels,
        config.n_classes
    )
    random_test_labels_onehot = create_one_hot_labels(
        random_test_labels_decimal,
        config.n_classes
    )
    
    return (
        training_set,
        test_set,
        training_labels_onehot,
        random_test_labels_onehot,
        random_test_set,
        class_bits
    )


# ==============================================================================
# NEURAL NETWORK MODEL
# ==============================================================================

def initialize_weights(config: MixedSNRDecoderConfig) -> Tuple[Dict, Dict]:
    """
    Initialize weights and biases for the neural network.
    
    Uses Xavier initialization for better gradient flow.
    
    Args:
        config: Decoder configuration
        
    Returns:
        Tuple of (weights_dict, biases_dict)
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
    Build feedforward neural network without dropout.
    
    Args:
        x: Input tensor
        weights: Weight variables
        biases: Bias variables
        
    Returns:
        Output tensor (logits)
    """
    layer_1 = tf.nn.tanh(tf.matmul(x, weights['h1']) + biases['b1'])
    layer_2 = tf.nn.tanh(tf.matmul(layer_1, weights['h2']) + biases['b2'])
    layer_3 = tf.nn.tanh(tf.matmul(layer_2, weights['h3']) + biases['b3'])
    out_layer = tf.nn.tanh(tf.matmul(layer_3, weights['out']) + biases['out'])
    return out_layer


def multilayer_perceptron_with_dropout(
    x: tf.Tensor,
    weights: Dict[str, tf.Variable],
    biases: Dict[str, tf.Variable],
    keep_prob: tf.Tensor
) -> tf.Tensor:
    """
    Build feedforward neural network with dropout regularization.
    
    Dropout helps prevent overfitting by randomly dropping neurons
    during training.
    
    Args:
        x: Input tensor
        weights: Weight variables
        biases: Bias variables
        keep_prob: Probability of keeping a neuron active
        
    Returns:
        Output tensor (logits)
    """
    layer_1 = tf.nn.tanh(tf.matmul(x, weights['h1']) + biases['b1'])
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    
    layer_2 = tf.nn.tanh(tf.matmul(layer_1, weights['h2']) + biases['b2'])
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    
    layer_3 = tf.nn.tanh(tf.matmul(layer_2, weights['h3']) + biases['b3'])
    layer_3 = tf.nn.dropout(layer_3, keep_prob)
    
    out_layer = tf.nn.tanh(tf.matmul(layer_3, weights['out']) + biases['out'])
    return out_layer


# ==============================================================================
# TRAINING AND EVALUATION
# ==============================================================================

def calculate_ber(
    predictions: np.ndarray,
    labels: np.ndarray,
    block_length: int
) -> float:
    """
    Calculate Bit Error Rate (BER).
    
    Args:
        predictions: Predicted class labels
        labels: True class labels
        block_length: Number of bits per block
        
    Returns:
        Bit error rate (0 to 1)
    """
    total_bit_errors = sum(
        calculate_hamming_distance_decimal(pred, label, block_length)
        for pred, label in zip(predictions, labels)
    )
    total_bits = len(predictions) * block_length
    return total_bit_errors / total_bits


def calculate_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Calculate frame-level accuracy.
    
    Args:
        predictions: Predicted class labels
        labels: True class labels
        
    Returns:
        Accuracy (0 to 1)
    """
    correct = np.sum(predictions == labels.astype(int))
    return correct / len(predictions)


def create_model_graph(
    config: MixedSNRDecoderConfig,
    training_set: np.ndarray,
    training_labels: np.ndarray,
    test_set: np.ndarray,
    test_labels: np.ndarray,
    random_test_set: Optional[np.ndarray] = None,
    random_test_labels: Optional[np.ndarray] = None,
    train_mode: bool = True,
    model_save_path: Optional[str] = None,
    model_load_path: Optional[str] = None
) -> None:
    """
    Create, train, and evaluate the neural network model.
    
    Args:
        config: Decoder configuration
        training_set: Training input data
        training_labels: Training labels (one-hot)
        test_set: Test input data
        test_labels: Test labels (one-hot)
        random_test_set: Random test set (optional)
        random_test_labels: Random test labels (optional)
        train_mode: If True, train model; if False, load existing model
        model_save_path: Path to save trained model
        model_load_path: Path to load existing model
    """
    # Reset TensorFlow graph
    tf.reset_default_graph()
    
    # Create placeholders
    x = tf.placeholder("float", [None, config.n_input], name='input')
    y = tf.placeholder("float", [None, config.n_classes], name='labels')
    keep_prob = tf.placeholder("float", name='keep_prob')
    
    # Initialize weights
    weights, biases = initialize_weights(config)
    
    # Build model (with or without dropout)
    if config.use_dropout:
        pred = multilayer_perceptron_with_dropout(x, weights, biases, keep_prob)
        test_model = multilayer_perceptron(x, weights, biases)
    else:
        pred = multilayer_perceptron(x, weights, biases)
        test_model = pred
    
    # Define loss function
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
    )
    
    # Add L2 regularization
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
    saver = tf.train.Saver()
    
    if train_mode:
        # Training mode
        with tf.Session() as sess:
            sess.run(init)
            
            print("Starting training...")
            print(f"Training samples: {config.num_training_samples}")
            print(f"SNR values: {config.snr_db_vec} dB")
            print(f"Test SNR: {config.snr_db_vec[config.test_snr_index]} dB")
            
            # Training loop
            for epoch in range(config.training_epochs):
                avg_cost = 0.0
                total_batch = int(config.num_training_samples / config.batch_size)
                
                # Mini-batch training
                for i in range(total_batch):
                    batch_x = training_set[
                        i * config.batch_size:(i + 1) * config.batch_size, :
                    ]
                    batch_y = training_labels[
                        i * config.batch_size:(i + 1) * config.batch_size
                    ]
                    
                    feed_dict = {x: batch_x, y: batch_y}
                    if config.use_dropout:
                        feed_dict[keep_prob] = config.dropout_keep_prob
                    
                    _, c = sess.run([optimizer, cost], feed_dict=feed_dict)
                    avg_cost += c / total_batch
                
                # Display progress
                if epoch % config.display_step == 0:
                    print(f"Epoch: {epoch + 1:04d} | cost = {avg_cost:.9f}")
            
            print("Optimization Finished!")
            
            # Save model if path provided
            if model_save_path:
                saver.save(sess, model_save_path)
                print(f"Model saved to {model_save_path}")
            
            # Test model
            correct_prediction = tf.equal(tf.argmax(test_model, 1), tf.argmax(y, 1))
            predictions_tensor = tf.argmax(test_model, 1)
            
            # Evaluate on test set
            predictions = predictions_tensor.eval(feed_dict={x: test_set})
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            test_accuracy = accuracy.eval({x: test_set, y: test_labels})
            
            print(f"\nTest Set Accuracy: {test_accuracy:.4f}")
            
            # Calculate BER
            test_labels_decimal = np.argmax(test_labels, axis=1)
            ber = calculate_ber(
                predictions,
                test_labels_decimal,
                config.block_length
            )
            print
