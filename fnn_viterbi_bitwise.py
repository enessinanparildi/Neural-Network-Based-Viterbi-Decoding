#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bit-wise Neural Viterbi Decoder using Binary Classification.

This module implements a neural network decoder that predicts bits individually
using binary classification (0 or 1). Unlike the multiclass approach, this
decoder can handle longer sequences by predicting one bit at a time.

The decoder can start from any encoder state, making it suitable for
sequential decoding scenarios.

Author: Sinan
Date: July 17, 2017
Improved: December 2024
"""

from typing import Tuple, Optional, Dict
import math
import tensorflow as tf
import numpy as np
from random import randint


# ==============================================================================
# CONFIGURATION AND GLOBAL CONSTANTS
# ==============================================================================

class BitwiseDecoderConfig:
    """Configuration container for bitwise neural decoder."""
    
    def __init__(
        self,
        constraint_length: int = 6,
        block_length: int = 512,
        n_hidden_1: int = 30,
        n_hidden_2: int = 100,
        n_hidden_3: int = 100,
        n_hidden_4: int = 100,
        learning_rate: float = 0.0002,
        training_epochs: int = 300,
        batch_size: int = 1000,
        beta_regularization: float = 0.0001,
        training_size: int = 300000,
        test_size: int = 300000,
        snr_db_range: Tuple[int, int] = (-2, 9),
        test_snr_index: int = 0,
        bit_position: int = 0,
        starting_state: int = 0,
    ):
        """
        Initialize bitwise decoder configuration.
        
        Args:
            constraint_length: Constraint length of convolutional encoder
            block_length: Number of information bits to decode
            n_hidden_1: Neurons in first hidden layer
            n_hidden_2: Neurons in second hidden layer
            n_hidden_3: Neurons in third hidden layer
            n_hidden_4: Neurons in fourth hidden layer
            learning_rate: Adam optimizer learning rate
            training_epochs: Number of training epochs
            batch_size: Mini-batch size
            beta_regularization: L2 regularization coefficient
            training_size: Total training samples
            test_size: Total test samples
            snr_db_range: Range of SNR values for training (min, max)
            test_snr_index: Index of SNR value for testing
            bit_position: Position of bit to predict (0 to block_length-1)
            starting_state: Initial encoder state (0 to 2^constraint_length-1)
        """
        # Encoder parameters
        self.constraint_length = constraint_length
        self.block_length = block_length
        self.total_length = constraint_length + block_length
        
        # Network architecture
        # Input includes constraint_length + block_length encoded bits (doubled)
        self.n_input = 2 * self.total_length
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.n_hidden_3 = n_hidden_3
        self.n_hidden_4 = n_hidden_4
        self.n_output = 2  # Binary classification: [0, 1]
        
        # Training parameters
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.beta_regularization = beta_regularization
        self.display_step = 1
        
        # Dataset parameters
        self.training_size = training_size
        self.test_size = test_size
        
        # Channel parameters
        snr_db_vec = np.arange(snr_db_range[0], snr_db_range[1])
        self.snr_db_vec = snr_db_vec
        self.snr_vec = 10 ** (snr_db_vec / 10.0)
        self.n0_vec = 1.0 / self.snr_vec
        self.sigma_vec = np.sqrt(self.n0_vec * 0.5)
        
        # Test configuration
        self.test_snr_index = test_snr_index
        self.bit_position = bit_position
        self.starting_state = starting_state


# ==============================================================================
# ENCODING FUNCTIONS
# ==============================================================================

def extract_state_bits(state_number: int, constraint_length: int = 6) -> np.ndarray:
    """
    Convert encoder state number to binary representation.
    
    The encoder state is represented by the last (constraint_length - 1) bits
    in the shift register.
    
    Args:
        state_number: State number (0 to 2^constraint_length - 1)
        constraint_length: Constraint length of encoder
        
    Returns:
        Binary array of shape (constraint_length,) representing the state
        
    Example:
        >>> extract_state_bits(5, 6)  # State 5 = 000101
        array([0, 0, 0, 1, 0, 1])
    """
    bits = np.zeros(constraint_length, dtype=np.int32)
    # Convert state number to binary string with fixed width
    binary_str = format(state_number, f'0{constraint_length}b')
    # Convert string to array, fill from right
    binary_list = [int(b) for b in binary_str]
    bits[len(bits) - len(binary_list):] = binary_list
    return bits


def encode_133171_with_state(
    bits: np.ndarray,
    state_num: int
) -> np.ndarray:
    """
    Encode bits using (133,171) encoder starting from a specific state.
    
    This allows encoding to start from any encoder state, which is useful
    for decoding in the middle of a sequence.
    
    Args:
        bits: Information bits to encode
        state_num: Initial encoder state (0 to 63 for constraint length 6)
        
    Returns:
        Encoded sequence with length 2 * (len(bits) + constraint_length)
        
    Note:
        The state bits are prepended to the information bits before encoding
    """
    # Get state bits and prepend to information bits
    state_bits = extract_state_bits(state_num)
    bits_with_state = np.concatenate([state_bits, bits])
    
    length = len(bits_with_state)
    encoded_bits = np.zeros(length * 2, dtype=np.int32)
    
    # Encode first 6 bits (initialization phase)
    encoded_bits[0] = bits_with_state[0] % 2
    encoded_bits[1] = bits_with_state[0] % 2
    
    encoded_bits[2] = bits_with_state[1] % 2
    encoded_bits[3] = (bits_with_state[1] + bits_with_state[0]) % 2
    
    encoded_bits[4] = (bits_with_state[2] + bits_with_state[0]) % 2
    encoded_bits[5] = (bits_with_state[2] + bits_with_state[1] + bits_with_state[0]) % 2
    
    encoded_bits[6] = (bits_with_state[3] + bits_with_state[1] + bits_with_state[0]) % 2
    encoded_bits[7] = (
        bits_with_state[3] + bits_with_state[2] + 
        bits_with_state[1] + bits_with_state[0]
    ) % 2
    
    encoded_bits[8] = (bits_with_state[4] + bits_with_state[2] + bits_with_state[1]) % 2
    encoded_bits[9] = (
        bits_with_state[4] + bits_with_state[3] + 
        bits_with_state[2] + bits_with_state[1]
    ) % 2
    
    encoded_bits[10] = (
        bits_with_state[5] + bits_with_state[3] + 
        bits_with_state[2] + bits_with_state[0]
    ) % 2
    encoded_bits[11] = (
        bits_with_state[5] + bits_with_state[4] + 
        bits_with_state[3] + bits_with_state[2]
    ) % 2
    
    # Encode remaining bits (steady state)
    for i in range(6, length):
        # G1 = 133 (octal) = 1011011 (binary)
        encoded_bits[2 * i] = (
            bits_with_state[i] + bits_with_state[i - 2] + 
            bits_with_state[i - 3] + bits_with_state[i - 5] + 
            bits_with_state[i - 6]
        ) % 2
        
        # G2 = 171 (octal) = 1111001 (binary)
        encoded_bits[2 * i + 1] = (
            bits_with_state[i] + bits_with_state[i - 1] + 
            bits_with_state[i - 2] + bits_with_state[i - 3] + 
            bits_with_state[i - 6]
        ) % 2
    
    return encoded_bits


# ==============================================================================
# CHANNEL AND DATA GENERATION
# ==============================================================================

def modulate_awgn(encoded_seq: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply BPSK modulation and add AWGN to encoded sequence.
    
    Args:
        encoded_seq: Binary encoded sequence
        sigma: Noise standard deviation
        
    Returns:
        Noisy modulated sequence
    """
    modulated = encoded_seq.copy().astype(np.float32)
    modulated[np.where(modulated == 0)] = -1
    noise = np.random.normal(0, sigma, len(modulated))
    return modulated + noise


def generate_input_vectors(length: int, number: int) -> np.ndarray:
    """
    Generate random binary vectors.
    
    Args:
        length: Length of each vector
        number: Number of vectors to generate
        
    Returns:
        Array of shape (number, length) with random binary values
    """
    return np.random.randint(2, size=(number, length))


def generate_training_data(
    config: BitwiseDecoderConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training dataset with mixed SNR values.
    
    Args:
        config: Decoder configuration
        
    Returns:
        Tuple of (info_sequences, encoded_sequences)
        - info_sequences: Shape (training_size, block_length)
        - encoded_sequences: Shape (training_size, 2 * total_length)
    """
    # Generate random information sequences
    info_seq = generate_input_vectors(
        config.block_length,
        config.training_size
    ).astype(np.float32)
    
    # Encode all sequences starting from specified state
    encoded_seq = np.array([
        encode_133171_with_state(seq, config.starting_state)
        for seq in info_seq
    ])
    
    # Add noise with random SNR from available range
    for i in range(config.training_size):
        snr_idx = np.random.randint(len(config.sigma_vec))
        encoded_seq[i, :] = modulate_awgn(
            encoded_seq[i, :],
            config.sigma_vec[snr_idx]
        )
    
    return info_seq, encoded_seq


def generate_test_data(
    config: BitwiseDecoderConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate test dataset with fixed SNR value.
    
    Args:
        config: Decoder configuration
        
    Returns:
        Tuple of (info_sequences, encoded_sequences)
    """
    # Generate random information sequences
    info_seq = generate_input_vectors(
        config.block_length,
        config.test_size
    )
    
    # Encode all sequences
    encoded_seq = np.array([
        encode_133171_with_state(seq, config.starting_state)
        for seq in info_seq
    ])
    
    # Add noise with fixed test SNR
    test_sigma = config.sigma_vec[config.test_snr_index]
    for i in range(config.test_size):
        encoded_seq[i, :] = modulate_awgn(encoded_seq[i, :], test_sigma)
    
    return info_seq, encoded_seq


# ==============================================================================
# NEURAL NETWORK MODEL
# ==============================================================================

def build_network(
    x: tf.Tensor,
    config: BitwiseDecoderConfig
) -> Tuple[tf.Tensor, Dict[str, tf.Variable], Dict[str, tf.Variable]]:
    """
    Build feedforward neural network for binary classification.
    
    Args:
        x: Input placeholder tensor
        config: Decoder configuration
        
    Returns:
        Tuple of (output_layer, weights_dict, biases_dict)
    """
    # Initialize weights with Xavier initialization
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
        'h4': tf.Variable(
            tf.truncated_normal(
                [config.n_hidden_3, config.n_hidden_4],
                stddev=math.sqrt(2.0 / config.n_hidden_3)
            ),
            name='weights_h4'
        ),
        'out': tf.Variable(
            tf.truncated_normal(
                [config.n_hidden_1, config.n_output],
                stddev=math.sqrt(2.0 / config.n_hidden_4)
            ),
            name='weights_out'
        ),
    }
    
    biases = {
        'b1': tf.Variable(tf.zeros([config.n_hidden_1]), name='bias_h1'),
        'b2': tf.Variable(tf.zeros([config.n_hidden_2]), name='bias_h2'),
        'b3': tf.Variable(tf.zeros([config.n_hidden_3]), name='bias_h3'),
        'b4': tf.Variable(tf.zeros([config.n_hidden_4]), name='bias_h4'),
        'out': tf.Variable(tf.zeros([config.n_output]), name='bias_out'),
    }
    
    # Build network layers with ReLU activation
    layer_1 = tf.nn.relu(tf.matmul(x, weights['h1']) + biases['b1'])
    
    # Note: Layers 2-4 are commented out to reduce complexity
    # Uncomment if needed for better performance
    # layer_2 = tf.nn.relu(tf.matmul(layer_1, weights['h2']) + biases['b2'])
    # layer_3 = tf.nn.relu(tf.matmul(layer_2, weights['h3']) + biases['b3'])
    # layer_4 = tf.nn.relu(tf.matmul(layer_3, weights['h4']) + biases['b4'])
    
    # Output layer (logits for binary classification)
    output = tf.nn.relu(tf.matmul(layer_1, weights['out']) + biases['out'])
    
    return output, weights, biases


# ==============================================================================
# TRAINING AND EVALUATION
# ==============================================================================

def train_and_evaluate(
    config: BitwiseDecoderConfig,
    train_info: np.ndarray,
    train_encoded: np.ndarray,
    test_info: np.ndarray,
    test_encoded: np.ndarray,
    model_save_path: Optional[str] = None
) -> Tuple[np.ndarray, float]:
    """
    Train the neural network and evaluate on test set.
    
    Args:
        config: Decoder configuration
        train_info: Training information sequences
        train_encoded: Training encoded sequences (noisy)
        test_info: Test information sequences
        test_encoded: Test encoded sequences (noisy)
        model_save_path: Path to save trained model (optional)
        
    Returns:
        Tuple of (predictions, error_rate)
    """
    # Reset TensorFlow graph
    tf.reset_default_graph()
    
    # Create placeholders
    x = tf.placeholder("float", [None, config.n_input], name='input')
    y = tf.placeholder("float", [None, config.n_output], name='labels')
    
    # Build network
    pred, weights, biases = build_network(x, config)
    
    # Define loss function (binary cross-entropy)
    cost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y)
    )
    
    # Add L2 regularization
    regularizer = (
        tf.nn.l2_loss(weights['h1']) + 
        tf.nn.l2_loss(weights['h2'])
    )
    cost = tf.reduce_mean(cost + config.beta_regularization * regularizer)
    
    # Define optimizer
    optimizer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate
    ).minimize(cost)
    
    # Prepare labels for specified bit position
    train_labels = train_info[:, config.bit_position]
    test_labels = test_info[:, config.bit_position]
    
    # Convert to one-hot encoding
    train_labels_onehot = np.zeros((config.training_size, 2), dtype=np.float16)
    test_labels_onehot = np.zeros((config.test_size, 2), dtype=np.float16)
    
    for i in range(config.training_size):
        train_labels_onehot[i, int(train_labels[i])] = 1
    for i in range(config.test_size):
        test_labels_onehot[i, int(test_labels[i])] = 1
    
    # Initialize session
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init)
        
        # Training loop
        for epoch in range(config.training_epochs):
            avg_cost = 0.0
            total_batch = int(config.training_size / config.batch_size)
            
            # Mini-batch training
            for i in range(total_batch):
                batch_x = train_encoded[
                    i * config.batch_size:(i + 1) * config.batch_size, :
                ]
                batch_y = train_labels_onehot[
                    i * config.batch_size:(i + 1) * config.batch_size
                ]
                
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            
            # Display progress
            if epoch % config.display_step == 0:
                print(f"Epoch: {epoch + 1:04d} | cost = {avg_cost:.9f}")
        
        print("Optimization Finished!")
        
        # Save model if path provided
        if model_save_path:
            saver.save(sess, model_save_path)
            print(f"Model saved to {model_save_path}")
        
        # Evaluate on test set
        predictions_tensor = tf.argmax(pred, 1)
        predictions = predictions_tensor.eval(feed_dict={x: test_encoded})
        
        # Calculate error rate
        errors = np.sum(predictions != test_labels.astype(int))
        error_rate = errors / len(predictions)
        
        print(f"Test Error Rate: {error_rate:.6f}")
        print(f"Test Accuracy: {1 - error_rate:.6f}")
        
        return predictions, error_rate


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """
    Main function for bitwise neural Viterbi decoder.
    """
    print("=" * 70)
    print("Bitwise Neural Viterbi Decoder")
    print("=" * 70)
    
    # Initialize configuration
    config = BitwiseDecoderConfig(
        constraint_length=6,
        block_length=512,
        n_hidden_1=30,
        learning_rate=0.0002,
        training_epochs=300,
        batch_size=1000,
        training_size=300000,
        test_size=300000,
        test_snr_index=0,  # Test at lowest SNR
        bit_position=0,    # Predict first bit
    )
    
    print("\nConfiguration:")
    print(f"  Block Length: {config.block_length}")
    print(f"  Constraint Length: {config.constraint_length}")
    print(f"  Predicting Bit Position: {config.bit_position}")
    print(f"  Test SNR: {config.snr_db_vec[config.test_snr_index]} dB")
    print(f"  Training Samples: {config.training_size}")
    print(f"  Test Samples: {config.test_size}")
    
    # Generate data
    print("\nGenerating training data...")
    train_info, train_encoded = generate_training_data(config)
    
    print("Generating test data...")
    test_info, test_encoded = generate_test_data(config)
    
    # Train and evaluate
    print("\nTraining neural network...")
    predictions, error_rate = train_and_evaluate(
        config,
        train_info,
        train_encoded,
        test_info,
        test_encoded
    )
    
    print("\n" + "=" * 70)
    print("Decoding complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
