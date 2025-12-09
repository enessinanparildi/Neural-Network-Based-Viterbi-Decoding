#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sequential Prediction Neural Viterbi Decoder.

This module implements a sliding window approach for neural decoding where
the network predicts one bit at a time based on a window of noisy encoded bits.
This approach is inspired by "ANN_decoder_Sujan" paper and attempts to replicate
its methodology for convolutional code decoding.

The key idea is to use a feedforward network that looks at a fixed window
of received symbols and predicts the information bit at the center of that window.

Author: Enes
Date: July 20, 2017
Improved: December 2024

Note:
    This is an experimental approach that showed mixed results. More work
    is needed to optimize the window size, network architecture, and training
    procedure. Simpler encoders may yield better results.
"""

from typing import Tuple, Optional, Dict
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# ==============================================================================
# CONFIGURATION AND GLOBAL CONSTANTS
# ==============================================================================

class SequentialDecoderConfig:
    """Configuration container for sequential prediction decoder."""
    
    def __init__(
        self,
        total_train_length_info: int = 1000100,
        total_test_length_info: int = 1000100,
        window_length: int = 200,
        n_hidden_1: int = 500,
        n_hidden_2: int = 500,
        n_hidden_3: int = 500,
        n_hidden_4: int = 500,
        learning_rate: float = 0.0004,
        training_epochs: int = 50,
        batch_size: int = 100,
        beta_regularization: float = 0.00000,
        snr_db_range: Tuple[int, int] = (-2, 9),
        test_snr_index: int = 1,
        model_save_path: str = "./models/seq_predict_model",
        model_load_path: Optional[str] = None,
        test_old_model: bool = False,
    ):
        """
        Initialize sequential decoder configuration.
        
        Args:
            total_train_length_info: Total information bits in training stream
            total_test_length_info: Total information bits in test stream
            window_length: Number of encoded bits in sliding window
            n_hidden_1: Neurons in first hidden layer
            n_hidden_2: Neurons in second hidden layer
            n_hidden_3: Neurons in third hidden layer
            n_hidden_4: Neurons in fourth hidden layer
            learning_rate: Adam optimizer learning rate
            training_epochs: Number of training epochs
            batch_size: Mini-batch size
            beta_regularization: L2 regularization coefficient
            snr_db_range: Range of SNR values (min, max)
            test_snr_index: Index of SNR value for testing
            model_save_path: Path to save trained model
            model_load_path: Path to load existing model
            test_old_model: If True, load and test existing model
        """
        # Stream lengths
        self.total_train_length_info = total_train_length_info
        self.total_train_length_coded = 2 * total_train_length_info
        self.total_test_length_info = total_test_length_info
        self.total_test_length_coded = 2 * total_test_length_info
        
        # Window parameters
        # The window slides over the encoded stream
        # and predicts the information bit at the center
        self.window_length = window_length
        self.training_size = 1  # Single long training sequence
        self.test_size = 1      # Single long test sequence
        
        # Calculate total training and test instances
        # Each instance is one window position
        self.total_training_instances = int(
            (self.total_train_length_coded - window_length) * 0.5
        )
        self.total_test_instances = int(
            (self.total_test_length_coded - window_length) * 0.5
        )
        
        # Network architecture
        self.n_input = window_length
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.n_hidden_3 = n_hidden_3
        self.n_hidden_4 = n_hidden_4
        self.n_output = 2  # Binary classification
        
        # Training parameters
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.beta_regularization = beta_regularization
        self.display_step = 1
        
        # SNR parameters
        snr_db_vec = np.arange(snr_db_range[0], snr_db_range[1])
        self.snr_db_vec = snr_db_vec
        self.snr_vec = 10 ** (snr_db_vec / 10.0)
        self.n0_vec = 1.0 / self.snr_vec
        self.sigma_vec = np.sqrt(self.n0_vec * 0.5)
        self.test_snr_index = test_snr_index
        
        # Model paths
        self.model_save_path = model_save_path
        self.model_load_path = model_load_path
        self.test_old_model = test_old_model


# ==============================================================================
# DATA GENERATION AND ENCODING
# ==============================================================================

def generate_info_sequence(block_length: int) -> np.ndarray:
    """
    Generate a random binary information sequence.
    
    Args:
        block_length: Length of the information sequence
        
    Returns:
        Binary array with random 0s and 1s
    """
    return np.random.randint(2, size=block_length)


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
    
    # Initialize encoding
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
        encoded_bits[2 * i] = (
            bits[i] + bits[i - 2] + bits[i - 3] + 
            bits[i - 5] + bits[i - 6]
        ) % 2
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
    Apply BPSK modulation and add AWGN.
    
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


def generate_training_data(
    total_length_info: int,
    training_size: int,
    sigma_vec: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate continuous training stream with mixed SNR.
    
    This creates a single long sequence of information bits,
    encodes them, and adds noise. The decoder will then slide
    a window over this noisy sequence.
    
    Args:
        total_length_info: Total information bits
        training_size: Number of sequences (usually 1 for streaming)
        sigma_vec: Array of noise standard deviations
        
    Returns:
        Tuple of (info_sequences, encoded_sequences)
    """
    # Generate random information sequences
    info_seq = np.random.randint(
        2,
        size=(training_size, total_length_info)
    ).astype(np.float32)
    
    # Encode sequences
    encoded_seq = np.array([
        encode_133171(seq) for seq in info_seq
    ])
    
    # Add noise with random SNR
    for i in range(training_size):
        snr_idx = np.random.randint(len(sigma_vec))
        encoded_seq[i, :] = modulate_awgn(encoded_seq[i, :], sigma_vec[snr_idx])
    
    return info_seq, encoded_seq


def generate_test_data(
    total_length_info: int,
    test_size: int,
    sigma_vec: np.ndarray,
    noise_index: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate continuous test stream with fixed SNR.
    
    Args:
        total_length_info: Total information bits
        test_size: Number of sequences
        sigma_vec: Array of noise standard deviations
        noise_index: Index of SNR to use
        
    Returns:
        Tuple of (info_sequences, encoded_sequences)
    """
    # Generate random information sequences
    info_seq = np.random.randint(2, size=(test_size, total_length_info))
    
    # Encode sequences
    encoded_seq = np.array([
        encode_133171(seq) for seq in info_seq
    ])
    
    # Add noise with fixed SNR
    test_sigma = sigma_vec[noise_index]
    for i in range(test_size):
        encoded_seq[i, :] = modulate_awgn(encoded_seq[i, :], test_sigma)
    
    return info_seq, encoded_seq


# ==============================================================================
# WINDOW EXTRACTION
# ==============================================================================

def get_single_instance(
    window_length: int,
    coded_seq: np.ndarray,
    info_seq: np.ndarray,
    offset: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a single training instance (window) from the sequence.
    
    This function extracts a window of encoded bits and the corresponding
    information bit that should be predicted.
    
    Args:
        window_length: Size of the sliding window
        coded_seq: Noisy encoded sequence (shape: [length, 1])
        info_seq: Information sequence (shape: [length, 1])
        offset: Position in the sequence to extract window from
        
    Returns:
        Tuple of (window_data, label)
        - window_data: Array of shape (window_length,)
        - label: One-hot encoded label of shape (2,)
        
    Note:
        The information bit at position offset/2 corresponds to
        the encoded bits starting at position offset
    """
    # Transpose to get column vectors
    coded_seq_t = np.transpose(coded_seq)
    info_seq_t = np.transpose(info_seq)
    
    # Extract window of encoded bits
    batch_x = coded_seq_t[offset:offset + window_length, 0]
    
    # Get corresponding information bit
    # The information bit index is offset/2 because
    # each information bit produces 2 encoded bits
    info_bit_value = info_seq_t[int(offset * 0.5), 0]
    
    # Create one-hot label
    batch_y = np.zeros(2, dtype=np.float32)
    batch_y[int(info_bit_value)] = 1.0
    
    return batch_x, batch_y


# ==============================================================================
# NEURAL NETWORK MODEL
# ==============================================================================

def build_network(
    x: tf.Tensor,
    config: SequentialDecoderConfig
) -> Tuple[tf.Tensor, Dict[str, tf.Variable], Dict[str, tf.Variable]]:
    """
    Build deep feedforward neural network.
    
    Uses 4 hidden layers with tanh activation for learning
    complex patterns in the noisy window.
    
    Args:
        x: Input placeholder
        config: Decoder configuration
        
    Returns:
        Tuple of (output, weights, biases)
    """
    # Initialize weights with Xavier scaling
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
                [config.n_hidden_4, config.n_output],
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
    
    # Build network layers
    layer_1 = tf.nn.tanh(tf.matmul(x, weights['h1']) + biases['b1'])
    layer_2 = tf.nn.tanh(tf.matmul(layer_1, weights['h2']) + biases['b2'])
    layer_3 = tf.nn.tanh(tf.matmul(layer_2, weights['h3']) + biases['b3'])
    layer_4 = tf.nn.tanh(tf.matmul(layer_3, weights['h4']) + biases['b4'])
    output = tf.nn.tanh(tf.matmul(layer_4, weights['out']) + biases['out'])
    
    return output, weights, biases


# ==============================================================================
# TRAINING AND EVALUATION
# ==============================================================================

def train_and_evaluate(config: SequentialDecoderConfig) -> None:
    """
    Train and evaluate the sequential prediction decoder.
    
    This function implements the complete training pipeline:
    1. Generate continuous training and test streams
    2. Build the neural network
    3. Train by sliding window over the stream
    4. Evaluate on test stream
    
    Args:
        config: Sequential decoder configuration
    """
    # Reset TensorFlow graph
    tf.reset_default_graph()
    session = tf.InteractiveSession()
    
    print("=" * 70)
    print("Sequential Prediction Neural Decoder")
    print("=" * 70)
    
    print("\nConfiguration:")
    print(f"  Training stream length: {config.total_train_length_info} bits")
    print(f"  Test stream length: {config.total_test_length_info} bits")
    print(f"  Window length: {config.window_length} encoded bits")
    print(f"  Training instances: {config.total_training_instances}")
    print(f"  Test instances: {config.total_test_instances}")
    print(f"  Test SNR: {config.snr_db_vec[config.test_snr_index]} dB")
    
    # Generate data streams
    print("\nGenerating data streams...")
    train_info, train_coded = generate_training_data(
        config.total_train_length_info,
        config.training_size,
        config.sigma_vec
    )
    
    test_info, test_coded = generate_test_data(
        config.total_test_length_info,
        config.test_size,
        config.sigma_vec,
        config.test_snr_index
    )
    
    print(f"Training stream shape: {train_coded.shape}")
    print(f"Test stream shape: {test_coded.shape}")
    
    # Create placeholders
    x = tf.placeholder("float", [None, config.n_input], name='input')
    y = tf.placeholder("float", [None, config.n_output], name='labels')
    
    # Build network
    pred, weights, biases = build_network(x, config)
    
    # Define loss function
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
    
    # Initialize variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    # Prediction operations
    predictions_tensor = tf.argmax(pred, 1)
    
    if config.test_old_model:
        # Load and evaluate existing model
        print("\nLoading existing model...")
        if config.model_load_path:
            saver.restore(session, config.model_load_path)
            print(f"Model loaded from {config.model_load_path}")
        
        # Prepare test data
        print("\nPreparing test data...")
        test_data = np.zeros(
            (config.total_test_instances, config.window_length),
            dtype=np.float32
        )
        test_labels = np.zeros(
            (config.total_test_instances, 2),
            dtype=np.float32
        )
        
        for i in range(config.total_test_instances):
            test_data[i, :], test_labels[i, :] = get_single_instance(
                config.window_length,
                test_coded,
                test_info,
                2 * i
            )
        
        # Evaluate
        print("Evaluating...")
        predictions = predictions_tensor.eval(feed_dict={x: test_data})
        test_labels_decimal = np.argmax(test_labels, axis=1)
        
        # Calculate error rate
        errors = np.sum(predictions != test_labels_decimal)
        error_rate = errors / len(predictions)
        
        print(f"\nTest Error Rate: {error_rate:.6f}")
        print(f"Test Accuracy: {1 - error_rate:.6f}")
        
    else:
        # Training mode
        session.run(init)
        
        print("\nStarting training...")
        offset = 0
        
        for epoch in range(config.training_epochs):
            avg_cost = 0.0
            total_batch = int(config.total_training_instances / config.batch_size)
            
            # Loop over all batches
            for i in range(total_batch):
                # Prepare batch by extracting windows
                batch_x = np.zeros(
                    (config.batch_size, config.window_length),
                    dtype=np.float32
                )
                batch_y = np.zeros(
                    (config.batch_size, 2),
                    dtype=np.float32
                )
                
                for j in range(config.batch_size):
                    batch_x[j, :], batch_y[j, :] = get_single_instance(
                        config.window_length,
                        train_coded,
                        train_info,
                        offset
                    )
                    offset += 2  # Move to next information bit
                
                # Train step
                _, c = session.run(
                    [optimizer, cost],
                    feed_dict={x: batch_x, y: batch_y}
                )
                
                avg_cost += c / total_batch
            
            # Reset offset for next epoch
            offset = 0
            
            # Display progress
            if epoch % config.display_step == 0:
                print(f"Epoch: {epoch + 1:04d} | cost = {avg_cost:.9f}")
        
        print("Optimization Finished!")
        
        # Save model
        saver.save(session, config.model_save_path)
        print(f"Model saved to {config.model_save_path}")
        
        # Prepare test data
        print("\nPreparing test data...")
        test_data = np.zeros(
            (config.total_test_instances, config.window_length),
            dtype=np.float32
        )
        test_labels = np.zeros(
            (config.total_test_instances, 2),
            dtype=np.float32
        )
        
        for i in range(config.total_test_instances):
            test_data[i, :], test_labels[i, :] = get_single_instance(
                config.window_length,
                test_coded,
                test_info,
                2 * i
            )
        
        # Evaluate
        print("Evaluating on test set...")
        predictions = predictions_tensor.eval(feed_dict={x: test_data})
        test_labels_decimal = np.argmax(test_labels, axis=1)
        
        # Calculate error rate
        errors = np.sum(predictions != test_labels_decimal)
        error_rate = errors / len(predictions)
        
        print(f"\nTest Error Rate: {error_rate:.6f}")
        print(f"Test Accuracy: {1 - error_rate:.6f}")
        print(f"SNR: {config.snr_db_vec[config.test_snr_index]} dB")
    
    session.close()
    
    print("\n" + "=" * 70)
    print("Decoding complete!")
    print("=" * 70)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """
    Main function for sequential prediction decoder.
    """
    # Initialize configuration
    config = SequentialDecoderConfig(
        total_train_length_info=1000100,
        total_test_length_info=1000100,
        window_length=200,
        n_hidden_1=500,
        n_hidden_2=500,
        n_hidden_3=500,
        n_hidden_4=500,
        learning_rate=0.0004,
        training_epochs=50,
        batch_size=100,
        beta_regularization=0.00000,
        snr_db_range=(-2, 9),
        test_snr_index=1,
        test_old_model=False,
    )
    
    # Run training and evaluation
    train_and_evaluate(config)


if __name__ == "__main__":
    main()
