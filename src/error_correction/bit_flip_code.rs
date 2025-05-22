// Three-Qubit Bit Flip Code Implementation
//
// This file implements the three-qubit bit flip code, which is a basic quantum
// error correction code that can detect and correct bit flip errors.

use crate::core::QuantumRegister;

/// Implements the three-qubit bit flip code for quantum error correction
pub struct BitFlipCode {
    /// The quantum register containing the encoded qubits
    register: QuantumRegister,
    
    /// The index of the first qubit in the logical encoding
    start_idx: usize,
}

impl BitFlipCode {
    /// Create a new bit flip code instance
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
    /// |0⟩ → |000⟩
    /// |1⟩ → |111⟩
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
        
        // Determine the source state
        let source_is_one = self.register.qubit(qubit_idx)
            .is_some_and(|q| q.prob_1() > 0.5);
        
        // If the source is |1⟩, apply X to all qubits
        if source_is_one {
            self.register.x(q1);
            self.register.x(q2);
            self.register.x(q3);
        }
        
        (q1, q2, q3)
    }
    
    /// Decode the three physical qubits back to a single logical qubit
    ///
    /// # Arguments
    ///
    /// * `q1`, `q2`, `q3` - The indices of the three physical qubits
    /// * `target_idx` - The index of the qubit to decode into
    pub fn decode(&mut self, q1: usize, q2: usize, q3: usize, target_idx: usize) {
        // Perform majority vote to determine the logical state
        let state1 = self.register.qubit(q1).map_or(0, |q| i32::from(q.prob_1() > 0.5));
        let state2 = self.register.qubit(q2).map_or(0, |q| i32::from(q.prob_1() > 0.5));
        let state3 = self.register.qubit(q3).map_or(0, |q| i32::from(q.prob_1() > 0.5));
        
        let majority = (state1 + state2 + state3) >= 2;
        
        // Reset target qubit to |0⟩ first
        if let Some(target) = self.register.qubit(target_idx) {
            if target.prob_1() > 0.5 {
                self.register.x(target_idx);
            }
        }
        
        // If majority is |1⟩, apply X to target
        if majority {
            self.register.x(target_idx);
        }
    }
    
    /// Correct errors using majority voting
    ///
    /// This function detects and corrects bit flip errors by majority voting.
    ///
    /// # Arguments
    ///
    /// * `q1`, `q2`, `q3` - The indices of the three physical qubits
    pub fn correct(&mut self, q1: usize, q2: usize, q3: usize) {
        // Perform majority vote to determine the correct state
        let state1 = self.register.qubit(q1).map_or(0, |q| i32::from(q.prob_1() > 0.5));
        let state2 = self.register.qubit(q2).map_or(0, |q| i32::from(q.prob_1() > 0.5));
        let state3 = self.register.qubit(q3).map_or(0, |q| i32::from(q.prob_1() > 0.5));
        
        let sum = state1 + state2 + state3;
        let majority = sum >= 2;
        
        // Correct any qubits that differ from the majority
        if (state1 == 1) != majority {
            self.register.x(q1);
        }
        
        if (state2 == 1) != majority {
            self.register.x(q2);
        }
        
        if (state3 == 1) != majority {
            self.register.x(q3);
        }
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
        
        // Create a bit flip code with the encoding starting at qubit 1
        let mut code = BitFlipCode::new(register, 1);
        
        // Encode the 0th qubit into qubits 1, 2, 3
        let (q1, q2, q3) = code.encode(0);
        
        // Verify that all qubits are |1⟩ (since the source was |1⟩)
        assert!(code.register().qubit(q1).unwrap().prob_1() > 0.99);
        assert!(code.register().qubit(q2).unwrap().prob_1() > 0.99);
        assert!(code.register().qubit(q3).unwrap().prob_1() > 0.99);
        
        // Decode back to the 0th qubit
        code.decode(q1, q2, q3, 0);
        
        // Verify that the 0th qubit is still |1⟩
        assert!(code.register().qubit(0).unwrap().prob_1() > 0.99);
    }
    
    #[test]
    fn test_error_correction() {
        // Create a register with 4 qubits
        let mut register = QuantumRegister::new(4);
        
        // Create a bit flip code with the encoding starting at qubit 1
        let mut code = BitFlipCode::new(register, 1);
        
        // Encode the 0th qubit into qubits 1, 2, 3
        let (q1, q2, q3) = code.encode(0);
        
        // Introduce a bit flip error on the second qubit
        code.register_mut().x(q2);
        
        // Perform error correction
        code.correct(q1, q2, q3);
        
        // Verify that all qubits are restored to |0⟩
        assert!(code.register().qubit(q1).unwrap().prob_0() > 0.99);
        assert!(code.register().qubit(q2).unwrap().prob_0() > 0.99);
        assert!(code.register().qubit(q3).unwrap().prob_0() > 0.99);
    }
} 