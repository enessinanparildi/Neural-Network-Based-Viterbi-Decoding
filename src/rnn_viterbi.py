#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Recurrent Neural Network-Based Viterbi Decoder.

This module implements a sequence-to-sequence (seq2seq) neural decoder using
LSTM (Long Short-Term Memory) networks. The encoder-decoder architecture
maps noisy encoded sequences back to information sequences.

This approach is inspired by neural machine translation and can handle
variable-length sequences.

Author: Fahrettin
Date: July 17, 2017
Improved: December 2024
"""

from typing import Tuple, Dict
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# ==============================================================================
# CONFIGURATION AND GLOBAL CONSTANTS
# ==============================================================================

class RNNDecoderConfig:
    """Configuration container for RNN-based decoder."""
    
    def __init__(
        self,
        block_length: int = 100,
        encoder_hidden_units: int = 20,
        decoder_hidden_units: Optional[int] = None,
        learning_rate: float = 0.01,
        training_epochs: int = 6,
        batch_size: int = 100,
        beta_regularization: float = 0.001,
        training_size: int = 100000,
        test_size: int = 1000,
        snr_db_range: Tuple[int, int] = (0, 9),
        test_snr_index: int = 2,
    ):
        """
        Initialize RNN decoder configuration.
        
        Args:
            block_length: Number of information bits per block
            encoder_hidden_units: Number of LSTM units in encoder
            decoder_hidden_units: Number of LSTM units in decoder (defaults to encoder_hidden_units)
            learning_rate: Adam optimizer learning rate
            training_epochs: Number of training epochs
            batch_size: Mini-batch size for training
            beta_regularization: L2 regularization coefficient
            training_size: Total training samples
            test_size: Total test samples
            snr_db_range: Range of SNR values (min, max)
            test_snr_index: Index of SNR value for testing
        """
        # Sequence parameters
        self.block_length = block_length
        self.encoded_length = 2 * block_length
        
        # RNN architecture
        self.encoder_hidden_units = encoder_hidden_units
        self.decoder_hidden_units = (
            decoder_hidden_units if decoder_hidden_units 
            else encoder_hidden_units
        )
        self.vocab_size = 1  # Binary vocabulary
        
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
        self.test_snr_index = test_snr_index
        
        # Special tokens for sequence-to-sequence
        self.PAD = 0  # Padding token
        self.EOS = 1  # End-of-sequence token


# ==============================================================================
# ENCODING FUNCTIONS
# ==============================================================================

def encode_133171(bits: np.ndarray) -> np.ndarray:
    """
    Encode information sequence using (133,171) convolutional encoder.
    
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


# ==============================================================================
# DATA GENERATION
# ==============================================================================

def generate_input_vectors(length: int, number: int) -> np.ndarray:
    """
    Generate random binary sequences.
    
    Args:
        length: Length of each sequence
        number: Number of sequences to generate
        
    Returns:
        Array of shape (number, length) with random binary values
    """
    return np.random.randint(2, size=(number, length))


def generate_datasets(
    config: RNNDecoderConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate training and test datasets for seq2seq model.
    
    Args:
        config: RNN decoder configuration
        
    Returns:
        Tuple of (train_info, train_encoded, test_info, test_encoded)
        All sequences are mapped to {-1, +1} for BPSK
    """
    # Generate training data
    train_info = generate_input_vectors(config.block_length, config.training_size)
    train_encoded = np.array([
        encode_133171(seq) for seq in train_info
    ])
    
    # Add noise to training data
    test_sigma = config.sigma_vec[config.test_snr_index]
    for i in range(config.training_size):
        train_encoded[i, :] = modulate_awgn(train_encoded[i, :], test_sigma)
    
    # Map {0, 1} to {-1, +1} for information sequences
    train_info = train_info.astype(np.float32)
    train_info[np.where(train_info == 0)] = -1
    
    # Generate test data
    test_info = generate_input_vectors(config.block_length, config.test_size)
    test_encoded = np.array([
        encode_133171(seq) for seq in test_info
    ])
    
    # Add noise to test data
    for i in range(config.test_size):
        test_encoded[i, :] = modulate_awgn(test_encoded[i, :], test_sigma)
    
    # Map {0, 1} to {-1, +1} for information sequences
    test_info = test_info.astype(np.float32)
    test_info[np.where(test_info == 0)] = -1
    
    return train_info, train_encoded, test_info, test_encoded


# ==============================================================================
# SEQUENCE-TO-SEQUENCE MODEL
# ==============================================================================

def build_seq2seq_model(
    config: RNNDecoderConfig
) -> Dict[str, tf.Tensor]:
    """
    Build sequence-to-sequence model with LSTM encoder and decoder.
    
    Architecture:
        Encoder: LSTM processes the noisy encoded sequence
        Decoder: LSTM generates the information sequence
    
    Args:
        config: RNN decoder configuration
        
    Returns:
        Dictionary containing model tensors and operations
    """
    # Create placeholders
    # Shape: (sequence_length, batch_size)
    encoder_inputs = tf.placeholder(
        shape=(config.encoded_length, None),
        dtype=tf.float32,
        name='encoder_inputs'
    )
    
    decoder_targets = tf.placeholder(
        shape=(config.encoded_length, None),
        dtype=tf.float32,
        name='decoder_targets'
    )
    
    decoder_inputs = tf.placeholder(
        shape=(config.encoded_length, None),
        dtype=tf.float32,
        name='decoder_inputs'
    )
    
    # Expand dimensions for RNN processing
    # Shape: (sequence_length, batch_size, 1)
    encoder_inputs_expanded = tf.expand_dims(encoder_inputs, 2)
    decoder_targets_expanded = tf.expand_dims(decoder_targets, 2)
    decoder_inputs_expanded = tf.expand_dims(decoder_inputs, 2)
    
    # Build encoder LSTM
    encoder_cell = tf.contrib.rnn.LSTMCell(config.encoder_hidden_units)
    
    # Run encoder
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
        encoder_cell,
        encoder_inputs_expanded,
        dtype=tf.float32,
        time_major=True,
    )
    
    # Build decoder LSTM
    decoder_cell = tf.contrib.rnn.LSTMCell(config.decoder_hidden_units)
    
    # Run decoder with encoder's final state as initial state
    decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
        decoder_cell,
        decoder_inputs_expanded,
        initial_state=encoder_final_state,
        dtype=tf.float32,
        time_major=True,
        scope='plain_decoder',
    )
    
    # Linear projection to vocabulary size
    decoder_logits = tf.contrib.layers.linear(
        decoder_outputs,
        config.vocab_size
    )
    
    # Predictions (argmax over vocabulary)
    decoder_prediction = tf.argmax(decoder_logits, 2)
    
    # Loss function (mean squared error for continuous values)
    loss = tf.reduce_mean(
        tf.squared_difference(decoder_targets_expanded, decoder_logits)
    )
    
    # Optimizer
    train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(loss)
    
    return {
        'encoder_inputs': encoder_inputs,
        'decoder_inputs': decoder_inputs,
        'decoder_targets': decoder_targets,
        'decoder_prediction': decoder_prediction,
        'loss': loss,
        'train_op': train_op,
    }


def prepare_batch(
    info_batch: np.ndarray,
    encoded_batch: np.ndarray,
    config: RNNDecoderConfig
) -> Dict[str, np.ndarray]:
    """
    Prepare a batch for seq2seq training/inference.
    
    Args:
        info_batch: Information sequences batch
        encoded_batch: Encoded sequences batch
        config: RNN decoder configuration
        
    Returns:
        Dictionary of feed_dict values
    """
    batch_size = len(info_batch)
    
    # Encoder inputs: noisy encoded sequences
    encoder_inputs = np.transpose(encoded_batch)
    encoder_inputs = np.expand_dims(encoder_inputs, axis=2)
    
    # Decoder targets: information sequences + EOS token + padding
    decoder_targets = np.concatenate([
        info_batch,
        np.tile(
            np.concatenate([[config.EOS], np.zeros(config.block_length - 1)]),
            (batch_size, 1)
        )
    ], axis=1)
    decoder_targets = np.transpose(decoder_targets)
    decoder_targets = np.expand_dims(decoder_targets, axis=2)
    
    # Decoder inputs: EOS token + information sequences + padding
    decoder_inputs = np.concatenate([
        np.tile([config.EOS], (batch_size, 1)),
        info_batch
    ], axis=1)
    decoder_inputs = np.concatenate([
        decoder_inputs,
        np.zeros((batch_size, config.block_length - 1))
    ], axis=1)
    decoder_inputs = np.transpose(decoder_inputs)
    decoder_inputs = np.expand_dims(decoder_inputs, axis=2)
    
    return {
        'encoder_inputs': encoder_inputs,
        'decoder_inputs': decoder_inputs,
        'decoder_targets': decoder_targets,
    }


# ==============================================================================
# TRAINING AND EVALUATION
# ==============================================================================

def train_seq2seq(
    config: RNNDecoderConfig,
    train_info: np.ndarray,
    train_encoded: np.ndarray,
    test_info: np.ndarray,
    test_encoded: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Train and evaluate seq2seq decoder.
    
    Args:
        config: RNN decoder configuration
        train_info: Training information sequences
        train_encoded: Training encoded sequences
        test_info: Test information sequences
        test_encoded: Test encoded sequences
        
    Returns:
        Tuple of (predictions, ber)
    """
    # Build model
    model = build_seq2seq_model(config)
    
    # Initialize session
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        # Training loop
        loss_history = []
        batches_per_epoch = int(config.training_size / config.batch_size)
        max_batches = batches_per_epoch * config.training_epochs
        
        print(f"Training for {config.training_epochs} epochs "
              f"({max_batches} batches)...")
        
        for batch_idx in range(max_batches):
            # Get batch
            batch_start = (batch_idx % batches_per_epoch) * config.batch_size
            batch_end = batch_start + config.batch_size
            
            info_batch = train_info[batch_start:batch_end]
            encoded_batch = train_encoded[batch_start:batch_end]
            
            # Prepare feed dict
            feed_dict_values = prepare_batch(info_batch, encoded_batch, config)
            feed_dict = {
                model['encoder_inputs']: feed_dict_values['encoder_inputs'],
                model['decoder_inputs']: feed_dict_values['decoder_inputs'],
                model['decoder_targets']: feed_dict_values['decoder_targets'],
            }
            
            # Train step
            _, loss_val = sess.run(
                [model['train_op'], model['loss']],
                feed_dict=feed_dict
            )
            loss_history.append(loss_val)
            
            # Log progress
            if batch_idx % batches_per_epoch == 0:
                epoch = batch_idx // batches_per_epoch
                print(f"Epoch {epoch + 1}/{config.training_epochs} | "
                      f"Loss: {loss_val:.6f}")
                
                # Show sample predictions
                if epoch == 0 or (epoch + 1) % 2 == 0:
                    predictions = sess.run(
                        model['decoder_prediction'],
                        feed_dict=feed_dict
                    )
                    print(f"  Sample input:  {encoded_batch[0, :10]}")
                    print(f"  Sample target: {info_batch[0, :10]}")
                    print(f"  Sample pred:   {predictions[:10, 0]}")
        
        print("Training finished!")
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_feed_values = prepare_batch(test_info, test_encoded, config)
        test_feed_dict = {
            model['encoder_inputs']: test_feed_values['encoder_inputs'],
            model['decoder_inputs']: test_feed_values['decoder_inputs'],
            model['decoder_targets']: test_feed_values['decoder_targets'],
        }
        
        predictions = sess.run(
            model['decoder_prediction'],
            feed_dict=test_feed_dict
        )
        
        # Calculate BER
        predictions = np.transpose(predictions)[:, :config.block_length]
        
        # Map back from {-1, +1} to {0, 1}
        test_info_binary = test_info.copy()
        test_info_binary[test_info_binary == -1] = 0
        
        # Count bit errors
        bit_errors = 0
        for i in range(config.test_size):
            bit_errors += np.sum(
                np.logical_xor(predictions[i, :], test_info_binary[i, :])
            )
        
        total_bits = config.test_size * config.block_length
        ber = bit_errors / total_bits
        
        print(f"Bit Error Rate: {ber:.6f}")
        print(f"Bit Errors: {bit_errors}/{total_bits}")
        
        # Plot loss history
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.grid(True)
        plt.savefig('rnn_training_loss.png')
        print("Loss plot saved to rnn_training_loss.png")
        
        return predictions, ber


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """
    Main function for RNN-based Viterbi decoder.
    """
    print("=" * 70)
    print("RNN-Based Viterbi Decoder (Sequence-to-Sequence)")
    print("=" * 70)
    
    # Initialize configuration
    config = RNNDecoderConfig(
        block_length=100,
        encoder_hidden_units=20,
        learning_rate=0.01,
        training_epochs=6,
        batch_size=100,
        training_size=100000,
        test_size=1000,
        test_snr_index=2,
    )
    
    print("\nConfiguration:")
    print(f"  Block Length: {config.block_length}")
    print(f"  Encoder Hidden Units: {config.encoder_hidden_units}")
    print(f"  Decoder Hidden Units: {config.decoder_hidden_units}")
    print(f"  Test SNR: {config.snr_db_vec[config.test_snr_index]} dB")
    print(f"  Training Samples: {config.training_size}")
    print(f"  Test Samples: {config.test_size}")
    
    # Generate datasets
    print("\nGenerating datasets...")
    train_info, train_encoded, test_info, test_encoded = generate_datasets(config)
    
    # Train and evaluate
    print("\nTraining RNN decoder...")
    predictions, ber = train_seq2seq(
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
