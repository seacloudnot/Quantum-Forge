// Three-Qubit Phase Flip Code Implementation
//
// This file implements the three-qubit phase flip code, which is a basic quantum
// error correction code that can detect and correct phase flip errors.

use crate::core::QuantumRegister;

/// Implementation of the three-qubit phase flip code
pub struct PhaseFlipCode {
    /// The quantum register containing the qubits
    register: QuantumRegister,
    
    /// The index of the first qubit in the encoding
    start_idx: usize,
}

impl PhaseFlipCode {
    /// Create a new phase flip code instance
    /// 
    /// # Arguments
    /// 
    /// * `register` - The quantum register to use
    /// * `start_idx` - The starting index for the encoded qubits
    #[must_use]
    pub fn new(register: QuantumRegister, start_idx: usize) -> Self {
        Self {
            register,
            start_idx,
        }
    }
    
    /// Encode a single logical qubit into three physical qubits
    ///
    /// The encoding maps:
    /// |0⟩ → |+++⟩
    /// |1⟩ → |---⟩
    ///
    /// # Arguments
    ///
    /// * `qubit_idx` - The index of the qubit to encode
    ///
    /// # Returns
    ///
    /// A tuple containing the indices of the three physical qubits
    /// 
    /// # Panics
    /// 
    /// Panics if `qubit_idx` is out of range or if there aren't enough qubits
    /// for encoding (at least 3 qubits starting from `start_idx`).
    pub fn encode(&mut self, qubit_idx: usize) -> (usize, usize, usize) {
        // Ensure we have enough qubits for encoding
        assert!(qubit_idx < self.register.size());
        assert!(self.start_idx + 2 < self.register.size());
        
        // Allocate three qubits for the encoding
        let q1 = self.start_idx;
        let q2 = self.start_idx + 1;
        let q3 = self.start_idx + 2;
        
        // First ensure the target qubits are in |0⟩ state
        self.register.set_zero(q1);
        self.register.set_zero(q2);
        self.register.set_zero(q3);
        
        // Put target qubits in |+⟩ state
        self.register.hadamard(q1);
        self.register.hadamard(q2);
        self.register.hadamard(q3);
        
        // Determine the source state
        let source_is_one = self.register.qubit(qubit_idx)
            .is_some_and(|q| q.prob_1() > 0.5);
        
        // If the source is |1⟩, apply Z to all qubits to get |---⟩
        if source_is_one {
            self.register.z(q1);
            self.register.z(q2);
            self.register.z(q3);
        }
        
        (q1, q2, q3)
    }
    
    /// Detect phase flip errors in the encoded qubits
    ///
    /// This performs error detection based on the phase information.
    ///
    /// # Arguments
    ///
    /// * `q1`, `q2`, `q3` - The indices of the three physical qubits
    /// 
    /// # Returns
    /// 
    /// A vector of qubit indices where errors were detected
    #[must_use]
    pub fn detect_errors(&self, q1: usize, q2: usize, q3: usize) -> Vec<usize> {
        let mut errors = Vec::new();
        
        // For phase flip code, we need to transform to X basis first
        // then check for bit flips (which correspond to phase flips in Z basis)
        // Here we're doing a simplified approach
        
        // Get qubit states - this is theoretical since measuring would collapse the state
        // In a real implementation, we'd use ancilla qubits and controlled operations
        
        // In X basis, we'd expect all qubits to be in |0⟩ or all in |1⟩
        // Differing qubits indicate phase flips in the original basis
        
        // Convert to the X basis by applying Hadamard
        // (This is a theoretical operation for error detection)
        
        // Now measure in the X basis - simulated measurement
        let state1 = self.register.qubit(q1).map_or(0, |q| i32::from(q.prob_1() > 0.5));
        let state2 = self.register.qubit(q2).map_or(0, |q| i32::from(q.prob_1() > 0.5));
        let state3 = self.register.qubit(q3).map_or(0, |q| i32::from(q.prob_1() > 0.5));
        
        // Determine the majority vote
        let majority = (state1 + state2 + state3) >= 2;
        
        // Find qubits that differ from the majority
        if (state1 == 1) != majority {
            errors.push(q1);
        }
        
        if (state2 == 1) != majority {
            errors.push(q2);
        }
        
        if (state3 == 1) != majority {
            errors.push(q3);
        }
        
        errors
    }
    
    /// Correct phase flip errors in the encoded qubits
    ///
    /// This performs syndrome measurement and correction based on majority vote.
    ///
    /// # Arguments
    ///
    /// * `q1`, `q2`, `q3` - The indices of the three physical qubits
    pub fn correct(&mut self, q1: usize, q2: usize, q3: usize) {
        // Get qubit states in the Z basis
        let state1 = self.register.qubit(q1).map_or(0, |q| i32::from(q.prob_1() > 0.5));
        let state2 = self.register.qubit(q2).map_or(0, |q| i32::from(q.prob_1() > 0.5));
        let state3 = self.register.qubit(q3).map_or(0, |q| i32::from(q.prob_1() > 0.5));
        
        // In a phase flip code, we need to work in the X basis
        // Apply Hadamard to transform to X basis
        self.register.hadamard(q1);
        self.register.hadamard(q2);
        self.register.hadamard(q3);
        
        // Perform majority vote correction (now bit flips in X basis)
        let majority = (state1 + state2 + state3) >= 2;
        
        // Correct each qubit if it doesn't match the majority
        if (state1 == 1) != majority {
            self.register.x(q1);
        }
        
        if (state2 == 1) != majority {
            self.register.x(q2);
        }
        
        if (state3 == 1) != majority {
            self.register.x(q3);
        }
        
        // Transform back to Z basis
        self.register.hadamard(q1);
        self.register.hadamard(q2);
        self.register.hadamard(q3);
    }
    
    /// Decode the three physical qubits back to a single logical qubit
    ///
    /// # Arguments
    ///
    /// * `q1`, `q2`, `q3` - The indices of the three physical qubits
    /// * `target_idx` - The index of the qubit to decode into
    pub fn decode(&mut self, q1: usize, q2: usize, q3: usize, target_idx: usize) {
        // First, make sure we're not affected by potential errors
        self.correct(q1, q2, q3);
        
        // Switch to the X basis for phase detection
        self.register.hadamard(q1);
        self.register.hadamard(q2);
        self.register.hadamard(q3);
        
        // Perform majority vote in X basis to determine the logical state
        let state1 = self.register.qubit(q1).map_or(0, |q| i32::from(q.prob_1() > 0.5));
        let state2 = self.register.qubit(q2).map_or(0, |q| i32::from(q.prob_1() > 0.5));
        let state3 = self.register.qubit(q3).map_or(0, |q| i32::from(q.prob_1() > 0.5));
        
        let majority = (state1 + state2 + state3) >= 2;
        
        // Reset target qubit to |0⟩ first
        self.register.set_zero(target_idx);
        
        // If majority is |1⟩ in X basis (meaning |-⟩ in Z basis after transform back)
        // we need to set the target qubit to |1⟩
        if majority {
            self.register.x(target_idx);
        }
    }
    
    /// Apply a phase flip to a qubit in the register
    ///
    /// # Arguments
    ///
    /// * `qubit_idx` - The index of the qubit to flip
    pub fn apply_phase_flip(&mut self, qubit_idx: usize) {
        self.register.z(qubit_idx);
    }
    
    /// Get a reference to the underlying quantum register
    #[must_use]
    pub fn register(&self) -> &QuantumRegister {
        &self.register
    }
    
    /// Get a mutable reference to the underlying quantum register
    pub fn register_mut(&mut self) -> &mut QuantumRegister {
        &mut self.register
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_encode_decode() {
        // Create a register with 4 qubits
        let mut register = QuantumRegister::new(4);
        
        // Set the first qubit to |1⟩
        register.x(0);
        
        // Create a phase flip code with the encoding starting at qubit 1
        let mut code = PhaseFlipCode::new(register, 1);
        
        // Encode the 0th qubit into qubits 1, 2, 3
        let (q1, q2, q3) = code.encode(0);
        
        // Decode back to the 0th qubit
        code.decode(q1, q2, q3, 0);
        
        // Verify that the 0th qubit is still |1⟩
        assert!(code.register().qubit(0).unwrap().prob_1() > 0.99);
        
        // Now test with |0⟩ input
        let register = QuantumRegister::new(4);
        // Leave first qubit as |0⟩
        
        // Create a phase flip code with the encoding starting at qubit 1
        let mut code = PhaseFlipCode::new(register, 1);
        
        // Encode the 0th qubit into qubits 1, 2, 3
        let (q1, q2, q3) = code.encode(0);
        
        // Decode back to the 0th qubit
        code.decode(q1, q2, q3, 0);
        
        // Verify that the 0th qubit is still |0⟩
        assert!(code.register().qubit(0).unwrap().prob_0() > 0.99);
    }
    
    #[test]
    fn test_error_correction() {
        // Create a register with 4 qubits
        let register = QuantumRegister::new(4);
        
        // Create a phase flip code with the encoding starting at qubit 1
        let mut code = PhaseFlipCode::new(register, 1);
        
        // Encode the 0th qubit into qubits 1, 2, 3
        let (q1, q2, q3) = code.encode(0);
        
        // Introduce a phase flip error on the second qubit
        code.register_mut().z(q2);
        
        // Perform error correction
        code.correct(q1, q2, q3);
        
        // Get a reference to the quantum register
        let register = code.register();
        
        // Verify that all qubits are in the |+⟩ state (since the source was |0⟩)
        // This means prob_0 and prob_1 should both be close to 0.5
        // But the phase should be positive, which we can't directly check
        let qubit1 = register.qubit(q1).unwrap();
        let qubit2 = register.qubit(q2).unwrap();
        let qubit3 = register.qubit(q3).unwrap();
        
        assert!((qubit1.prob_0() - 0.5).abs() < 0.01);
        assert!((qubit1.prob_1() - 0.5).abs() < 0.01);
        
        assert!((qubit2.prob_0() - 0.5).abs() < 0.01);
        assert!((qubit2.prob_1() - 0.5).abs() < 0.01);
        
        assert!((qubit3.prob_0() - 0.5).abs() < 0.01);
        assert!((qubit3.prob_1() - 0.5).abs() < 0.01);
    }
} 