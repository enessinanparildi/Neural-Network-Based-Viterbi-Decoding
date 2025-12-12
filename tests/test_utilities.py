"""
Unit tests for utility functions.

Tests helper functions like bit conversions, distance calculations,
and data manipulation utilities.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fnn_viterbi_improved import (
    generate_class_bits,
    create_one_hot_labels,
)
from fnn_viterbi_mixed_snr_improved import (
    binary_to_decimal,
    extract_bits_single,
    calculate_hamming_distance_decimal,
)


class TestGenerateClassBits:
    """Test generation of all possible bit combinations."""
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        block_length = 5
        class_bits = generate_class_bits(block_length)
        
        expected_rows = 2 ** block_length
        assert class_bits.shape == (expected_rows, block_length)
    
    def test_all_unique(self):
        """Test that all bit combinations are unique."""
        class_bits = generate_class_bits(5)
        
        # Convert rows to tuples for unique check
        unique_rows = set(tuple(row) for row in class_bits)
        assert len(unique_rows) == len(class_bits)
    
    def test_contains_all_zeros(self):
        """Test that all-zeros combination exists."""
        class_bits = generate_class_bits(5)
        
        all_zeros = np.all(class_bits == 0, axis=1)
        assert np.any(all_zeros)
    
    def test_contains_all_ones(self):
        """Test that all-ones combination exists."""
        class_bits = generate_class_bits(5)
        
        all_ones = np.all(class_bits == 1, axis=1)
        assert np.any(all_ones)
    
    def test_first_row_is_zeros(self):
        """Test that first row (index 0) is all zeros."""
        class_bits = generate_class_bits(5)
        
        assert np.all(class_bits[0] == 0)
    
    def test_last_row_is_ones(self):
        """Test that last row is all ones."""
        class_bits = generate_class_bits(5)
        
        assert np.all(class_bits[-1] == 1)
    
    def test_binary_order(self):
        """Test that rows are in binary counting order."""
        class_bits = generate_class_bits(3)
        
        expected = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ])
        
        assert np.array_equal(class_bits, expected)
    
    def test_different_lengths(self):
        """Test with different block lengths."""
        for length in [1, 3, 5, 8, 10]:
            class_bits = generate_class_bits(length)
            
            assert class_bits.shape == (2 ** length, length)
            assert np.all((class_bits == 0) | (class_bits == 1))


class TestOneHotEncoding:
    """Test one-hot label encoding."""
    
    def test_simple_encoding(self):
        """Test basic one-hot encoding."""
        labels = np.array([0, 1, 2])
        one_hot = create_one_hot_labels(labels, num_classes=3)
        
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        
        assert np.array_equal(one_hot, expected)
    
    def test_output_shape(self):
        """Test that output shape is correct."""
        labels = np.array([0, 1, 2, 3, 4])
        one_hot = create_one_hot_labels(labels, num_classes=10)
        
        assert one_hot.shape == (5, 10)
    
    def test_single_one_per_row(self):
        """Test that each row has exactly one 1."""
        labels = np.array([1, 3, 5, 7])
        one_hot = create_one_hot_labels(labels, num_classes=10)
        
        row_sums = np.sum(one_hot, axis=1)
        assert np.all(row_sums == 1)
    
    def test_correct_position(self):
        """Test that 1 is in correct position."""
        labels = np.array([0, 5, 9])
        one_hot = create_one_hot_labels(labels, num_classes=10)
        
        assert one_hot[0, 0] == 1
        assert one_hot[1, 5] == 1
        assert one_hot[2, 9] == 1
    
    def test_dtype(self):
        """Test that output has correct dtype."""
        labels = np.array([0, 1, 2])
        one_hot = create_one_hot_labels(labels, num_classes=3)
        
        assert one_hot.dtype in [np.float16, np.float32, np.float64]


class TestBinaryDecimalConversion:
    """Test binary to decimal conversion."""
    
    def test_simple_conversions(self):
        """Test simple binary to decimal conversions."""
        assert binary_to_decimal(np.array([0, 0, 0])) == 0
        assert binary_to_decimal(np.array([0, 0, 1])) == 1
        assert binary_to_decimal(np.array([0, 1, 0])) == 2
        assert binary_to_decimal(np.array([1, 0, 0])) == 4
    
    def test_all_ones(self):
        """Test conversion of all ones."""
        bits = np.ones(5, dtype=int)
        result = binary_to_decimal(bits)
        
        assert result == 31  # 2^5 - 1
    
    def test_all_zeros(self):
        """Test conversion of all zeros."""
        bits = np.zeros(8, dtype=int)
        result = binary_to_decimal(bits)
        
        assert result == 0
    
    def test_single_bit(self):
        """Test conversion of single bit."""
        assert binary_to_decimal(np.array([0])) == 0
        assert binary_to_decimal(np.array([1])) == 1
    
    def test_longer_sequences(self):
        """Test conversion of longer sequences."""
        # 10101010 = 170
        bits = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        result = binary_to_decimal(bits)
        assert result == 170


class TestExtractBitsSingle:
    """Test decimal to binary conversion."""
    
    def test_zero_conversion(self):
        """Test conversion of 0."""
        bits = extract_bits_single(0, block_length=5)
        
        assert np.all(bits == 0)
        assert len(bits) == 5
    
    def test_simple_conversions(self):
        """Test simple decimal to binary conversions."""
        # 5 = 00101
        bits = extract_bits_single(5, block_length=5)
        expected = np.array([0, 0, 1, 0, 1])
        
        assert np.array_equal(bits, expected)
    
    def test_max_value(self):
        """Test conversion of maximum value for given length."""
        block_length = 7
        max_val = 2 ** block_length - 1
        bits = extract_bits_single(max_val, block_length)
        
        assert np.all(bits == 1)
    
    def test_round_trip(self):
        """Test that decimal->binary->decimal is identity."""
        for value in [0, 5, 15, 31, 63]:
            bits = extract_bits_single(value, block_length=7)
            result = binary_to_decimal(bits)
            
            assert result == value
    
    def test_leading_zeros(self):
        """Test that leading zeros are preserved."""
        # 1 should be 0000001 with block_length=7
        bits = extract_bits_single(1, block_length=7)
        
        assert len(bits) == 7
        assert bits[0] == 0
        assert bits[-1] == 1


class TestHammingDistance:
    """Test Hamming distance calculation."""
    
    def test_identical_values(self):
        """Test that identical values have distance 0."""
        for value in [0, 5, 15, 31]:
            distance = calculate_hamming_distance_decimal(value, value, block_length=7)
            assert distance == 0
    
    def test_single_bit_difference(self):
        """Test values differing by single bit."""
        # 0 (000) vs 1 (001) -> distance 1
        distance = calculate_hamming_distance_decimal(0, 1, block_length=3)
        assert distance == 1
    
    def test_all_bits_different(self):
        """Test when all bits are different."""
        # 0 (0000) vs 15 (1111) -> distance 4
        distance = calculate_hamming_distance_decimal(0, 15, block_length=4)
        assert distance == 4
    
    def test_symmetry(self):
        """Test that distance is symmetric."""
        d1 = calculate_hamming_distance_decimal(5, 3, block_length=7)
        d2 = calculate_hamming_distance_decimal(3, 5, block_length=7)
        
        assert d1 == d2
    
    def test_known_distances(self):
        """Test with known Hamming distances."""
        # 5 (101) vs 3 (011) -> differ in positions 0,1 -> distance 2
        distance = calculate_hamming_distance_decimal(5, 3, block_length=3)
        assert distance == 2
    
    def test_triangle_inequality(self):
        """Test that Hamming distance obeys triangle inequality."""
        a, b, c = 0, 5, 7
        block_length = 4
        
        d_ab = calculate_hamming_distance_decimal(a, b, block_length)
        d_bc = calculate_hamming_distance_decimal(b, c, block_length)
        d_ac = calculate_hamming_distance_decimal(a, c, block_length)
        
        # Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        assert d_ac <= d_ab + d_bc


class TestUtilityConsistency:
    """Test consistency between utility functions."""
    
    def test_class_bits_one_hot_consistency(self):
        """Test consistency between class bits and one-hot encoding."""
        block_length = 5
        class_bits = generate_class_bits(block_length)
        
        # Create labels for each class
        labels = np.arange(2 ** block_length)
        one_hot = create_one_hot_labels(labels, num_classes=2 ** block_length)
        
        # Each one-hot row should correspond to a class
        assert len(class_bits) == len(one_hot)
    
    def test_binary_decimal_round_trip(self):
        """Test round-trip conversion."""
        block_length = 7
        
        for decimal_value in range(2 ** block_length):
            # Decimal -> Binary
            bits = extract_bits_single(decimal_value, block_length)
            
            # Binary -> Decimal
            result = binary_to_decimal(bits)
            
            assert result == decimal_value
    
    def test_hamming_via_conversion(self):
        """Test Hamming distance using explicit conversion."""
        val1, val2 = 5, 3
        block_length = 7
        
        # Method 1: Direct calculation
        dist1 = calculate_hamming_distance_decimal(val1, val2, block_length)
        
        # Method 2: Via binary conversion
        bits1 = extract_bits_single(val1, block_length)
        bits2 = extract_bits_single(val2, block_length)
        dist2 = np.sum(bits1 != bits2)
        
        assert dist1 == dist2


class TestEdgeCases:
    """Test edge cases for utility functions."""
    
    def test_generate_class_bits_single(self):
        """Test with single bit."""
        class_bits = generate_class_bits(1)
        
        assert class_bits.shape == (2, 1)
        assert np.array_equal(class_bits, [[0], [1]])
    
    def test_one_hot_single_class(self):
        """Test one-hot with single sample."""
        labels = np.array([3])
        one_hot = create_one_hot_labels(labels, num_classes=5)
        
        assert one_hot.shape == (1, 5)
        assert one_hot[0, 3] == 1
        assert np.sum(one_hot) == 1
    
    def test_hamming_zero_length(self):
        """Test Hamming distance with zero block length."""
        distance = calculate_hamming_distance_decimal(0, 0, block_length=0)
        assert distance == 0
    
    def test_large_block_length(self):
        """Test with large block length."""
        block_length = 15
        class_bits = generate_class_bits(block_length)
        
        assert class_bits.shape == (2 ** block_length, block_length)
        assert len(class_bits) == 32768


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
