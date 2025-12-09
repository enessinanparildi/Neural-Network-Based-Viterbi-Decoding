#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Production-ready training wrapper for Neural Viterbi Decoder.

This script wraps the original fnn_viterbi.py with production features:
- Configuration management
- Comprehensive logging
- Metrics tracking
- Checkpoint management
- Early stopping

Usage:
    python train_production.py --config config.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import tensorflow as tf
import numpy as np

# Import production utilities
from utils.config import ConfigManager
from utils.logger import setup_logger
from utils.metrics import MetricsTracker, EarlyStopping
from utils.checkpoint import CheckpointManager, create_model_saver

# Import original functions from fnn_viterbi
from fnn_viterbi import (
    encode133171,
    modulateAwgn,
    train_test_dataset,
    multilayer_perceptron,
    get_weights,
)


class ProductionTrainer:
    """Production-ready trainer wrapper for Neural Viterbi Decoder."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize trainer with configuration."""
        # Load configuration
        self.config_mgr = ConfigManager(config_path)
        self.config = self.config_mgr.get()
        
        # Setup logger
        self.logger = setup_logger(
            name="neural_viterbi",
            log_level=self.config.logging_level,
            log_dir=self.config.paths.logs_dir,
            use_json=(self.config.logging_format == "json"),
            log_to_console=self.config.log_to_console,
            log_to_file=self.config.log_to_file
        )
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(
            block_length=self.config.model.block_length
        )
        
        # Initialize checkpoint manager
        self.checkpoint_mgr = CheckpointManager(
            checkpoint_dir=self.config.paths.checkpoints_dir,
            max_to_keep=5
        )
        
        # Initialize early stopping
        if self.config.training.early_stopping_enabled:
            self.early_stopping = EarlyStopping(
                patience=self.config.training.early_stopping_patience,
                min_delta=self.config.training.early_stopping_min_delta,
                mode='min'
            )
        else:
            self.early_stopping = None
        
        self.logger.info("Trainer initialized with configuration:")
        self.logger.info(f"  Block length: {self.config.model.block_length}")
        self.logger.info(f"  Learning rate: {self.config.training.learning_rate}")
        self.logger.info(f"  Epochs: {self.config.training.epochs}")
        self.logger.info(f"  Batch size: {self.config.training.batch_size}")
    
    def prepare_data(self):
        """Prepare training and test datasets."""
        self.logger.info("Preparing datasets...")
        
        # Use original data preparation function
        (
            self.train_x,
            self.train_y,
            self.test_x,
            self.test_y,
            self.class_bits
        ) = train_test_dataset()
        
        self.logger.info(f"Training samples: {len(self.train_x)}")
        self.logger.info(f"Test samples: {len(self.test_x)}")
        
        return self
    
    def build_model(self):
        """Build TensorFlow computation graph."""
        self.logger.info("Building model...")
        
        tf.reset_default_graph()
        
        # Get network parameters from config
        n_input = 2 * self.config.model.block_length
        n_classes = 2 ** self.config.model.block_length
        
        # Placeholders
        self.x = tf.placeholder("float", [None, n_input])
        self.y = tf.placeholder("float", [None, n_classes])
        
        # Get weights and biases
        self.weights, self.biases = get_weights()
        
        # Build model
        self.pred = multilayer_perceptron(self.x, self.weights, self.biases)
        
        # Define loss
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y)
        )
        
        # Regularization
        regularizer = (
            tf.nn.l2_loss(self.weights["h1"]) +
            tf.nn.l2_loss(self.weights["h2"]) +
            tf.nn.l2_loss(self.weights["h3"])
        )
        self.cost = tf.reduce_mean(
            cost + self.config.training.beta_regularization * regularizer
        )
        
        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.config.training.learning_rate
        ).minimize(self.cost)
        
        # Predictions
        self.predictions = tf.argmax(self.pred, 1)
        self.correct_prediction = tf.equal(self.predictions, tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        
        self.logger.info("Model built successfully")
        
        return self
    
    def train(self):
        """Train the model with production features."""
        self.logger.info("Starting training...")
        
        # Create saver
        saver = create_model_saver(max_to_keep=5)
        
        # Training parameters
        batch_size = self.config.training.batch_size
        epochs = self.config.training.epochs
        display_step = self.config.training.display_step
        
        num_samples = len(self.train_x)
        total_batch = int(num_samples / batch_size)
        
        # Initialize session
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            
            # Training loop
            for epoch in range(epochs):
                epoch_start = time.time()
                avg_cost = 0.0
                
                # Mini-batch training
                for i in range(total_batch):
                    batch_x = self.train_x[i * batch_size:(i + 1) * batch_size]
                    batch_y = self.train_y[i * batch_size:(i + 1) * batch_size]
                    
                    _, c = sess.run(
                        [self.optimizer, self.cost],
                        feed_dict={self.x: batch_x, self.y: batch_y}
                    )
                    
                    avg_cost += c / total_batch
                
                # Calculate metrics
                if epoch % display_step == 0:
                    # Training metrics
                    train_predictions = sess.run(
                        self.predictions,
                        feed_dict={self.x: self.train_x}
                    )
                    train_labels = np.argmax(self.train_y, axis=1)
                    train_metrics = self.metrics_tracker.calculate_metrics(
                        train_predictions,
                        train_labels,
                        self.class_bits
                    )
                    
                    # Test metrics
                    test_predictions = sess.run(
                        self.predictions,
                        feed_dict={self.x: self.test_x}
                    )
                    test_labels = np.argmax(self.test_y, axis=1)
                    test_metrics = self.metrics_tracker.calculate_metrics(
                        test_predictions,
                        test_labels,
                        self.class_bits
                    )
                    
                    # Log progress
                    epoch_time = time.time() - epoch_start
                    self.logger.info(
                        f"Epoch {epoch + 1}/{epochs} | "
                        f"Time: {epoch_time:.2f}s | "
                        f"Loss: {avg_cost:.6f}"
                    )
                    self.logger.info(f"  Train: {train_metrics}")
                    self.logger.info(f"  Test:  {test_metrics}")
                    
                    # Track metrics
                    self.metrics_tracker.add_epoch_metrics(
                        epoch=epoch + 1,
                        train_metrics=train_metrics,
                        val_metrics=test_metrics,
                        train_loss=avg_cost
                    )
                    
                    # Save checkpoint
                    if epoch % 10 == 0 or epoch == epochs - 1:
                        checkpoint_path = self.checkpoint_mgr.save_checkpoint(
                            saver=saver,
                            session=sess,
                            epoch=epoch + 1,
                            metrics={
                                'ber': test_metrics.ber,
                                'fer': test_metrics.fer,
                                'accuracy': test_metrics.accuracy,
                                'loss': avg_cost
                            },
                            config=self.config_mgr._config_to_dict(self.config)
                        )
                        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
                    
                    # Early stopping check
                    if self.early_stopping:
                        if self.early_stopping(test_metrics.ber):
                            self.logger.info(
                                f"Early stopping triggered at epoch {epoch + 1}"
                            )
                            break
            
            # Final evaluation
            self.logger.info("Training completed!")
            self.logger.info("Final evaluation on test set:")
            
            final_predictions = sess.run(
                self.predictions,
                feed_dict={self.x: self.test_x}
            )
            final_labels = np.argmax(self.test_y, axis=1)
            final_metrics = self.metrics_tracker.calculate_metrics(
                final_predictions,
                final_labels,
                self.class_bits
            )
            
            self.logger.info(f"Final metrics: {final_metrics}")
            
            # Save final model
            final_model_path = self.config.paths.models_dir / "final_model"
            saver.save(sess, str(final_model_path))
            self.logger.info(f"Final model saved: {final_model_path}")
        
        # Save metrics history
        history_path = self.config.paths.results_dir / "training_history.json"
        self.metrics_tracker.save_history(history_path)
        self.logger.info(f"Metrics history saved: {history_path}")
        
        # Get best checkpoint
        best_checkpoint = self.checkpoint_mgr.get_best_checkpoint(metric='ber')
        if best_checkpoint:
            self.logger.info(
                f"Best checkpoint: Epoch {best_checkpoint['epoch']} "
                f"with BER {best_checkpoint['metrics']['ber']:.6f}"
            )
        
        return self
    
    def run(self):
        """Run complete training pipeline."""
        try:
            self.prepare_data()
            self.build_model()
            self.train()
            self.logger.info("Training pipeline completed successfully!")
            return 0
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}", exc_info=True)
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Neural Viterbi Decoder with production features"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Run training
    trainer = ProductionTrainer(config_path=args.config)
    exit_code = trainer.run()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
