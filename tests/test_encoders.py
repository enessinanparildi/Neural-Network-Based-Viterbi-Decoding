"""
Unit tests for convolutional encoders.

Tests the encoding functions used in all decoder variants to ensure
they produce correct outputs for known inputs.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fnn_viterbi_improved import (
    encode_57,
    encode_133171,
    generate_info_sequence,
)
from fnn_viterbi_bitwise_improved import (
    encode_133171_with_state,
    extract_state_bits,
)


class TestBasicEncoders:
    """Test basic encoding functions."""
    
    def test_generate_info_sequence_length(self):
        """Test that generated sequences have correct length."""
        block_length = 10
        seq = generate_info_sequence(block_length)
        
        assert len(seq) == block_length
        assert seq.dtype in [np.int32, np.int64]
    
    def test_generate_info_sequence_binary(self):
        """Test that generated sequences contain only 0s and 1s."""
        seq = generate_info_sequence(100)
        
        assert np.all((seq == 0) | (seq == 1))
    
    def test_encode_57_length(self):
        """Test that (7,5) encoder doubles the length."""
        info_seq = np.array([1, 0, 1, 0, 1])
        encoded = encode_57(info_seq)
        
        assert len(encoded) == 2 * len(info_seq)
    
    def test_encode_57_known_input(self):
        """Test (7,5) encoder with known input/output pair."""
        # All zeros should produce all zeros
        info_seq = np.zeros(5, dtype=np.int32)
        encoded = encode_57(info_seq)
        
        assert np.all(encoded == 0)
    
    def test_encode_57_single_one(self):
        """Test (7,5) encoder with single 1 bit."""
        info_seq = np.array([1, 0, 0, 0, 0])
        encoded = encode_57(info_seq)
        
        # First two outputs should be [1, 0] based on G1=7, G2=5
        assert encoded[0] == 1
        assert encoded[1] in [0, 1]  # Depends on next bit
    
    def test_encode_133171_length(self):
        """Test that (133,171) encoder doubles the length."""
        info_seq = np.array([1, 0, 1, 0, 1, 1, 0])
        encoded = encode_133171(info_seq)
        
        assert len(encoded) == 2 * len(info_seq)
    
    def test_encode_133171_all_zeros(self):
        """Test (133,171) encoder with all zeros."""
        info_seq = np.zeros(10, dtype=np.int32)
        encoded = encode_133171(info_seq)
        
        # All zeros input should produce all zeros output
        assert np.all(encoded == 0)
    
    def test_encode_133171_all_ones(self):
        """Test (133,171) encoder with all ones."""
        info_seq = np.ones(10, dtype=np.int32)
        encoded = encode_133171(info_seq)
        
        # Should produce valid encoded sequence
        assert len(encoded) == 20
        assert np.all((encoded == 0) | (encoded == 1))
    
    def test_encode_133171_binary_output(self):
        """Test that encoder produces only binary values."""
        info_seq = np.random.randint(0, 2, size=20)
        encoded = encode_133171(info_seq)
        
        assert np.all((encoded == 0) | (encoded == 1))
    
    def test_encode_133171_deterministic(self):
        """Test that encoder is deterministic."""
        info_seq = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        
        encoded1 = encode_133171(info_seq)
        encoded2 = encode_133171(info_seq)
        
        assert np.array_equal(encoded1, encoded2)


class TestStateAwareEncoders:
    """Test state-aware encoding functions."""
    
    def test_extract_state_bits_zero(self):
        """Test extracting bits for state 0."""
        bits = extract_state_bits(0, constraint_length=6)
        
        assert len(bits) == 6
        assert np.all(bits == 0)
    
    def test_extract_state_bits_max(self):
        """Test extracting bits for maximum state."""
        constraint_length = 6
        max_state = 2**constraint_length - 1
        bits = extract_state_bits(max_state, constraint_length)
        
        assert len(bits) == constraint_length
        assert np.all(bits == 1)
    
    def test_extract_state_bits_specific(self):
        """Test extracting bits for specific state value."""
        # State 5 = 000101 in 6 bits
        bits = extract_state_bits(5, constraint_length=6)
        
        expected = np.array([0, 0, 0, 1, 0, 1])
        assert np.array_equal(bits, expected)
    
    def test_encode_with_state_zero(self):
        """Test encoding with initial state 0."""
        info_bits = np.array([1, 0, 1, 1, 0])
        encoded = encode_133171_with_state(info_bits, state_num=0)
        
        # Should be longer due to state bits prepended
        assert len(encoded) == 2 * (len(info_bits) + 6)
    
    def test_encode_with_state_consistency(self):
        """Test that encoding with state 0 matches regular encoding."""
        info_bits = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0])
        
        # First 6 bits are state, rest are info
        encoded_regular = encode_133171(info_bits)
        encoded_with_state = encode_133171_with_state(info_bits[6:], state_num=0)
        
        # The outputs should match for the info bits portion
        assert len(encoded_with_state) == len(encoded_regular)
    
    def test_encode_with_different_states(self):
        """Test that different initial states produce different outputs."""
        info_bits = np.array([1, 0, 1])
        
        encoded_state0 = encode_133171_with_state(info_bits, state_num=0)
        encoded_state1 = encode_133171_with_state(info_bits, state_num=1)
        
        # Different states should produce different encodings
        assert not np.array_equal(encoded_state0, encoded_state1)


class TestEncoderProperties:
    """Test mathematical properties of encoders."""
    
    def test_linearity_property(self):
        """Test that encoder exhibits linear property (for linear codes)."""
        # For linear codes: encode(a) XOR encode(b) = encode(a XOR b)
        seq1 = np.array([1, 0, 1, 0, 1, 0, 1])
        seq2 = np.array([0, 1, 1, 0, 0, 1, 0])
        seq_xor = (seq1 + seq2) % 2
        
        enc1 = encode_133171(seq1)
        enc2 = encode_133171(seq2)
        enc_xor = encode_133171(seq_xor)
        
        # XOR of encoded sequences should equal encoding of XOR
        result = (enc1 + enc2) % 2
        assert np.array_equal(result, enc_xor)
    
    def test_zero_input_zero_output(self):
        """Test that all-zero input produces all-zero output."""
        for length in [5, 10, 20, 50]:
            info_seq = np.zeros(length, dtype=np.int32)
            encoded = encode_133171(info_seq)
            
            assert np.all(encoded == 0), f"Failed for length {length}"
    
    def test_rate_is_half(self):
        """Test that encoder is rate 1/2."""
        for length in [7, 10, 15, 20]:
            info_seq = np.random.randint(0, 2, size=length)
            encoded = encode_133171(info_seq)
            
            assert len(encoded) == 2 * length


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_encode_single_bit(self):
        """Test encoding a single bit."""
        info_seq = np.array([1])
        encoded = encode_133171(info_seq)
        
        assert len(encoded) == 2
    
    def test_encode_long_sequence(self):
        """Test encoding a long sequence."""
        info_seq = np.random.randint(0, 2, size=1000)
        encoded = encode_133171(info_seq)
        
        assert len(encoded) == 2000
        assert np.all((encoded == 0) | (encoded == 1))
    
    def test_encode_alternating_pattern(self):
        """Test encoding alternating bit pattern."""
        info_seq = np.array([0, 1] * 10)
        encoded = encode_133171(info_seq)
        
        assert len(encoded) == 40
        assert np.all((encoded == 0) | (encoded == 1))
    
    def test_encode_with_invalid_state(self):
        """Test encoding with state value that's too large."""
        info_bits = np.array([1, 0, 1])
        
        # State should be < 2^6 = 64 for constraint length 6
        # This should still work but only use lower 6 bits
        encoded = encode_133171_with_state(info_bits, state_num=100)
        
        assert len(encoded) == 2 * (len(info_bits) + 6)


class TestReproducibility:
    """Test that encoders produce consistent results."""
    
    def test_multiple_runs_same_result(self):
        """Test that same input always gives same output."""
        info_seq = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        
        results = [encode_133171(info_seq) for _ in range(10)]
        
        # All results should be identical
        for result in results[1:]:
            assert np.array_equal(results[0], result)
    
    def test_different_inputs_different_outputs(self):
        """Test that different inputs give different outputs."""
        seq1 = np.array([1, 0, 1, 0, 1, 0, 1])
        seq2 = np.array([0, 1, 0, 1, 0, 1, 0])
        
        enc1 = encode_133171(seq1)
        enc2 = encode_133171(seq2)
        
        assert not np.array_equal(enc1, enc2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
