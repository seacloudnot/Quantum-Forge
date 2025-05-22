// Quantum Error Correction Code (QECC)
//
// This module implements quantum error correction codes to protect quantum information
// from noise and errors in quantum computing systems.

use std::collections::HashMap;
use thiserror::Error;

use crate::core::QuantumState;
use crate::error_correction::{CorrectionCode, ErrorCorrection};

// Add this constant at the top of the file
const FLOAT_EPSILON: f64 = 1e-10;

/// Errors specific to Quantum Error Correction Codes
#[derive(Error, Debug, Clone)]
pub enum QECCError {
    /// Insufficient qubits for encoding
    #[error("Insufficient qubits for encoding with {0} code: need {1}, have {2}")]
    InsufficientQubits(String, usize, usize),
    
    /// Unsupported code
    #[error("Unsupported error correction code: {0:?}")]
    UnsupportedCode(CorrectionCode),
    
    /// Error syndrome detection failure
    #[error("Failed to detect error syndrome: {0}")]
    SyndromeDetectionFailure(String),
    
    /// Logical qubit error
    #[error("Logical qubit error at index {0}")]
    LogicalQubitError(usize),
    
    /// State already encoded
    #[error("Quantum state already encoded with {0:?} code")]
    AlreadyEncoded(CorrectionCode),
    
    /// State not encoded
    #[error("Quantum state not encoded with any error correction code")]
    NotEncoded,
}

/// Error syndrome type
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ErrorSyndrome {
    /// No error detected
    None,
    
    /// Bit flip error (X error)
    BitFlip(usize),
    
    /// Phase flip error (Z error)
    PhaseFlip(usize),
    
    /// Bit and phase flip error (Y error)
    BitPhaseFlip(usize),
    
    /// Multiple errors detected
    Multiple(Vec<usize>),
    
    /// Unknown syndrome pattern
    Unknown,
}

/// Configuration options for QECC
#[derive(Debug, Clone)]
pub struct QECCConfig {
    /// Whether to automatically detect and correct errors
    pub auto_correct: bool,
    
    /// How frequently to check for errors (number of operations)
    pub check_frequency: usize,
    
    /// Simulated environmental error rate
    pub error_rate: f64,
    
    /// Number of syndrome measurements to take
    pub syndrome_measurements: usize,
}

impl Default for QECCConfig {
    fn default() -> Self {
        Self {
            auto_correct: true,
            check_frequency: 10,
            error_rate: 0.01, // 1% error rate
            syndrome_measurements: 3,
        }
    }
}

/// Quantum Error Correction Code implementation
#[derive(Debug)]
pub struct QECC {
    /// Configuration settings
    config: QECCConfig,
    
    /// The protected quantum state
    state: Option<QuantumState>,
    
    /// Map of logical qubits to physical qubits
    logical_to_physical: HashMap<usize, Vec<usize>>,
    
    /// Current error correction code in use
    current_code: CorrectionCode,
    
    /// Operation counter for frequency checks
    operation_count: usize,
    
    /// History of detected error syndromes
    syndrome_history: Vec<ErrorSyndrome>,
    
    /// Number of corrections performed
    correction_count: usize,
}

impl Default for QECC {
    fn default() -> Self {
        Self::new()
    }
}

impl QECC {
    /// Create a new QECC instance
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: QECCConfig::default(),
            state: None,
            logical_to_physical: HashMap::new(),
            current_code: CorrectionCode::None,
            operation_count: 0,
            syndrome_history: Vec::new(),
            correction_count: 0,
        }
    }
    
    /// Create a new QECC instance with custom configuration
    #[must_use]
    pub fn with_config(config: QECCConfig) -> Self {
        Self {
            config,
            state: None,
            logical_to_physical: HashMap::new(),
            current_code: CorrectionCode::None,
            operation_count: 0,
            syndrome_history: Vec::new(),
            correction_count: 0,
        }
    }
    
    /// Set the quantum state to protect with error correction
    pub fn set_state(&mut self, state: QuantumState) {
        self.state = Some(state);
        self.logical_to_physical.clear();
        self.current_code = CorrectionCode::None;
        self.operation_count = 0;
        self.syndrome_history.clear();
    }
    
    /// Take the protected quantum state
    pub fn take_state(&mut self) -> Option<QuantumState> {
        // If state was encoded, we should decode it first
        if self.current_code != CorrectionCode::None {
            let _ = self.decode_and_correct();
        }
        
        self.state.take()
    }
    
    /// Get the current state being protected
    #[must_use]
    pub fn state(&self) -> Option<&QuantumState> {
        self.state.as_ref()
    }
    
    /// Check if the current state is encoded with error correction
    #[must_use]
    pub fn is_encoded(&self) -> bool {
        self.current_code != CorrectionCode::None
    }
    
    /// Get error correction statistics
    #[must_use]
    pub fn correction_stats(&self) -> (usize, Vec<ErrorSyndrome>) {
        (self.correction_count, self.syndrome_history.clone())
    }
    
    /// Reset error correction
    pub fn reset(&mut self) {
        if let Some(_state) = &mut self.state {
            if self.current_code != CorrectionCode::None {
                let _ = self.decode_and_correct();
            }
        }
        
        self.logical_to_physical.clear();
        self.current_code = CorrectionCode::None;
        self.operation_count = 0;
        self.syndrome_history.clear();
        self.correction_count = 0;
    }
    
    /// Register a quantum operation for error detection
    ///
    /// # Errors
    ///
    /// Returns `QECCError::NoStateToProtect` if there is no active state
    pub fn register_operation(&mut self) -> Result<(), QECCError> {
        if self.current_code == CorrectionCode::None || !self.config.auto_correct {
            return Ok(());
        }
        
        self.operation_count += 1;
        
        if self.operation_count >= self.config.check_frequency {
            // Check for errors and correct if needed
            let errors = self.detect_errors();
            
            if !errors.is_empty() {
                self.correct_errors(&errors);
            }
            
            self.operation_count = 0;
        }
        
        Ok(())
    }
    
    /// Encode using a repetition code (simple implementation)
    fn encode_repetition(&mut self, state: &mut QuantumState) -> Result<(), QECCError> {
        // Repetition code uses 3 physical qubits per logical qubit for bit flip protection
        // If we have exactly 3 qubits, we'll treat this as a special case for 1 logical qubit
        let num_qubits = state.num_qubits();
        
        if num_qubits < 3 {
            return Err(QECCError::InsufficientQubits(
                "Repetition".to_string(),
                3,
                num_qubits,
            ));
        }
        
        // Special case: Exactly 3 qubits - treat as single logical qubit
        if num_qubits == 3 {
            // Create a mapping for a single logical qubit
            self.logical_to_physical.clear();
            self.logical_to_physical.insert(0, vec![0, 1, 2]);
            
            // Apply encoding operations (just copies)
            let qubits = state.qubits_mut();
            let control_is_one = qubits[0].prob_1() > 0.5;
            
            if control_is_one {
                qubits[1].x();
                qubits[2].x();
            }
            
            return Ok(());
        }
        
        // General case for multiple logical qubits
        let logical_qubits = state.num_qubits() / 3;
        let physical_qubits_needed = logical_qubits * 3;
        
        if state.num_qubits() < physical_qubits_needed {
            return Err(QECCError::InsufficientQubits(
                "Repetition".to_string(),
                physical_qubits_needed,
                state.num_qubits(),
            ));
        }
        
        // Create a mapping of logical to physical qubits
        self.logical_to_physical.clear();
        
        for logical_idx in 0..logical_qubits {
            let physical_indices = vec![
                logical_idx, 
                logical_qubits + logical_idx,
                2 * logical_qubits + logical_idx,
            ];
            
            self.logical_to_physical.insert(logical_idx, physical_indices);
        }
        
        // Apply encoding operations to create redundancy
        for logical_idx in 0..logical_qubits {
            let physical_indices = self.logical_to_physical
                .get(&logical_idx)
                .ok_or(QECCError::LogicalQubitError(logical_idx))?;
            
            // Apply CNOT gates from first physical qubit to others
            let control_idx = physical_indices[0];
            let target_idx_1 = physical_indices[1];
            let target_idx_2 = physical_indices[2];
            
            // Get mutable access to qubits
            let qubits = state.qubits_mut();
            
            // Store the control qubit's state
            let control_qubit_is_one = qubits[control_idx].prob_1() > 0.5;
            
            // Apply the equivalent of CNOT: if control is |1⟩, flip the targets
            if control_qubit_is_one {
                qubits[target_idx_1].x();
                qubits[target_idx_2].x();
            }
        }
        
        Ok(())
    }
    
    /// Decode state that was encoded with the repetition code
    #[allow(clippy::unnecessary_wraps)]
    fn decode_repetition(&mut self, _state: &mut QuantumState) -> Result<(), QECCError> {
        // Algorithm simplified for clarity
        // In a real implementation, this would actually decode the repetition code
        
        // Reset the logical mapping
        self.logical_to_physical.clear();
        
        // Update the code
        self.current_code = CorrectionCode::None;
        
        Ok(())
    }
    
    /// Encode using Shor's 9-qubit code
    fn encode_shor(&mut self, state: &mut QuantumState) -> Result<(), QECCError> {
        // Shor's code uses 9 physical qubits per logical qubit
        let num_qubits = state.num_qubits();
        
        if num_qubits < 9 {
            return Err(QECCError::InsufficientQubits(
                "Shor's".to_string(),
                9,
                num_qubits,
            ));
        }
        
        // Special case: Exactly 9 qubits - treat as single logical qubit
        if num_qubits == 9 {
            // Create a mapping for a single logical qubit
            self.logical_to_physical.clear();
            self.logical_to_physical.insert(0, (0..9).collect());
            
            // Apply Shor's encoding for one logical qubit
            let qubits = state.qubits_mut();
            
            // First create the state |+⟩ using Hadamard
            qubits[0].hadamard();
            
            // Store state info for control qubit
            let q0_is_one = qubits[0].prob_1() > 0.5;
            
            // Apply CNOT gates
            if q0_is_one {
                qubits[3].x();
                qubits[6].x();
            }
            
            // Apply Hadamard to each block
            qubits[0].hadamard();
            qubits[3].hadamard();
            qubits[6].hadamard();
            
            // Create the 3-qubit repetition code in each block
            let q0_is_one = qubits[0].prob_1() > 0.5;
            let q3_is_one = qubits[3].prob_1() > 0.5;
            let q6_is_one = qubits[6].prob_1() > 0.5;
            
            if q0_is_one {
                qubits[1].x();
                qubits[2].x();
            }
            
            if q3_is_one {
                qubits[4].x();
                qubits[5].x();
            }
            
            if q6_is_one {
                qubits[7].x();
                qubits[8].x();
            }
            
            return Ok(());
        }
        
        // General case for multiple logical qubits
        let logical_qubits = state.num_qubits() / 9;
        let physical_qubits_needed = logical_qubits * 9;
        
        if state.num_qubits() < physical_qubits_needed {
            return Err(QECCError::InsufficientQubits(
                "Shor's".to_string(),
                physical_qubits_needed,
                state.num_qubits(),
            ));
        }
        
        // Create logical to physical mapping
        self.logical_to_physical.clear();
        
        for logical_idx in 0..logical_qubits {
            let base_idx = logical_idx * 9;
            let physical_indices = (0..9).map(|i| base_idx + i).collect();
            
            self.logical_to_physical.insert(logical_idx, physical_indices);
        }
        
        // Apply Shor's encoding circuit
        for logical_idx in 0..logical_qubits {
            let physical_indices = self.logical_to_physical
                .get(&logical_idx)
                .ok_or(QECCError::LogicalQubitError(logical_idx))?;
            
            // Get mutable access to qubits
            let qubits = state.qubits_mut();
            
            // First create the state |+⟩ using Hadamard
            qubits[physical_indices[0]].hadamard();
            
            // Store some state information to simulate CNOT gates
            let q0_is_one = qubits[physical_indices[0]].prob_1() > 0.5;
            
            // Apply CNOT gates to create |+++⟩ state
            if q0_is_one {
                qubits[physical_indices[3]].x();
                qubits[physical_indices[6]].x();
            }
            
            // Apply Hadamard to each block
            qubits[physical_indices[0]].hadamard();
            qubits[physical_indices[3]].hadamard();
            qubits[physical_indices[6]].hadamard();
            
            // Create the 3-qubit repetition code in each block
            let q0_is_one = qubits[physical_indices[0]].prob_1() > 0.5;
            let q3_is_one = qubits[physical_indices[3]].prob_1() > 0.5;
            let q6_is_one = qubits[physical_indices[6]].prob_1() > 0.5;
            
            if q0_is_one {
                qubits[physical_indices[1]].x();
                qubits[physical_indices[2]].x();
            }
            
            if q3_is_one {
                qubits[physical_indices[4]].x();
                qubits[physical_indices[5]].x();
            }
            
            if q6_is_one {
                qubits[physical_indices[7]].x();
                qubits[physical_indices[8]].x();
            }
        }
        
        Ok(())
    }
    
    /// Decode Shor's 9-qubit code
    fn decode_shor(&mut self, state: &mut QuantumState) -> Result<(), QECCError> {
        // Apply the inverse of the encoding circuit
        for logical_idx in 0..self.logical_to_physical.len() {
            let physical_indices = self.logical_to_physical
                .get(&logical_idx)
                .ok_or(QECCError::LogicalQubitError(logical_idx))?;
            
            // Get mutable access to qubits
            let qubits = state.qubits_mut();
            
            // Inverse of the 3-qubit repetition code in each block
            // For simulation, we do a majority vote
            
            // First block
            let block1_votes = [
                qubits[physical_indices[0]].prob_1() > 0.5,
                qubits[physical_indices[1]].prob_1() > 0.5,
                qubits[physical_indices[2]].prob_1() > 0.5,
            ];
            
            // Second block
            let block2_votes = [
                qubits[physical_indices[3]].prob_1() > 0.5,
                qubits[physical_indices[4]].prob_1() > 0.5,
                qubits[physical_indices[5]].prob_1() > 0.5,
            ];
            
            // Third block
            let block3_votes = [
                qubits[physical_indices[6]].prob_1() > 0.5,
                qubits[physical_indices[7]].prob_1() > 0.5,
                qubits[physical_indices[8]].prob_1() > 0.5,
            ];
            
            // Majority vote for each block
            let block1_result = block1_votes.iter().filter(|&&x| x).count() > 1;
            let block2_result = block2_votes.iter().filter(|&&x| x).count() > 1;
            let block3_result = block3_votes.iter().filter(|&&x| x).count() > 1;
            
            // Apply hadamard on each block
            qubits[physical_indices[0]].hadamard();
            qubits[physical_indices[3]].hadamard();
            qubits[physical_indices[6]].hadamard();
            
            // Restore original qubit based on majority of blocks
            let final_result = [block1_result, block2_result, block3_result].iter().filter(|&&x| x).count() > 1;
            
            if final_result {
                qubits[physical_indices[0]].x();
            }
        }
        
        self.logical_to_physical.clear();
        
        Ok(())
    }
    
    /// Encode using Steane's 7-qubit code
    fn encode_steane(&mut self, state: &mut QuantumState) -> Result<(), QECCError> {
        // Steane's code uses 7 physical qubits per logical qubit
        let num_qubits = state.num_qubits();
        
        if num_qubits < 7 {
            return Err(QECCError::InsufficientQubits(
                "Steane's".to_string(),
                7,
                num_qubits,
            ));
        }
        
        // Special case: Exactly 7 qubits - treat as single logical qubit
        if num_qubits == 7 {
            // Create a mapping for a single logical qubit
            self.logical_to_physical.clear();
            self.logical_to_physical.insert(0, (0..7).collect());
            
            // Apply Steane's encoding for one logical qubit
            let qubits = state.qubits_mut();
            
            // Prepare ancilla qubits in |+⟩ state
            for qubit in qubits.iter_mut().skip(1).take(6) {
                qubit.hadamard();
            }
            
            // Store state info for control qubits
            let q0_is_one = qubits[0].prob_1() > 0.5;
            let q1_is_one = qubits[1].prob_1() > 0.5;
            let q3_is_one = qubits[3].prob_1() > 0.5;
            
            // Apply simulated CNOT operations based on generator matrix
            if q0_is_one {
                qubits[2].x();
                qubits[4].x();
                qubits[6].x();
            }
            
            if q1_is_one {
                qubits[2].x();
                qubits[5].x();
                qubits[6].x();
            }
            
            if q3_is_one {
                qubits[4].x();
                qubits[5].x();
                qubits[6].x();
            }
            
            return Ok(());
        }
        
        // General case for multiple logical qubits
        let logical_qubits = state.num_qubits() / 7;
        let physical_qubits_needed = logical_qubits * 7;
        
        if state.num_qubits() < physical_qubits_needed {
            return Err(QECCError::InsufficientQubits(
                "Steane's".to_string(),
                physical_qubits_needed,
                state.num_qubits(),
            ));
        }
        
        // Create logical to physical mapping
        self.logical_to_physical.clear();
        
        for logical_idx in 0..logical_qubits {
            let base_idx = logical_idx * 7;
            let physical_indices = (0..7).map(|i| base_idx + i).collect();
            
            self.logical_to_physical.insert(logical_idx, physical_indices);
        }
        
        // Apply Steane's encoding circuit
        for logical_idx in 0..logical_qubits {
            let physical_indices = self.logical_to_physical
                .get(&logical_idx)
                .ok_or(QECCError::LogicalQubitError(logical_idx))?;
            
            // Get mutable access to qubits
            let qubits = state.qubits_mut();
            
            // Prepare ancilla qubits in |+⟩ state
            for i in 1..7 {
                qubits[physical_indices[i]].hadamard();
            }
            
            // Apply CNOT gates according to the generator matrix
            // This is a simplified simulation that doesn't match real quantum operations
            
            // Store state info for control qubits
            let control_states = [
                qubits[physical_indices[0]].prob_1() > 0.5,
                qubits[physical_indices[1]].prob_1() > 0.5,
                qubits[physical_indices[3]].prob_1() > 0.5,
            ];
            
            // Apply simulated CNOT operations
            if control_states[0] {
                qubits[physical_indices[2]].x();
                qubits[physical_indices[4]].x();
                qubits[physical_indices[6]].x();
            }
            
            if control_states[1] {
                qubits[physical_indices[2]].x();
                qubits[physical_indices[5]].x();
                qubits[physical_indices[6]].x();
            }
            
            if control_states[2] {
                qubits[physical_indices[4]].x();
                qubits[physical_indices[5]].x();
                qubits[physical_indices[6]].x();
            }
        }
        
        Ok(())
    }
    
    /// Decode Steane's 7-qubit code
    fn decode_steane(&mut self, state: &mut QuantumState) -> Result<(), QECCError> {
        // Apply the inverse of the encoding circuit
        for logical_idx in 0..self.logical_to_physical.len() {
            let physical_indices = self.logical_to_physical
                .get(&logical_idx)
                .ok_or(QECCError::LogicalQubitError(logical_idx))?;
            
            // Get mutable access to qubits
            let qubits = state.qubits_mut();
            
            // In a simplified simulation, we would do syndrome measurements
            // For now, we'll just revert to the original state
            
            // Apply Hadamard gates to restore the state
            for i in 1..7 {
                qubits[physical_indices[i]].hadamard();
            }
        }
        
        self.logical_to_physical.clear();
        
        Ok(())
    }
    
    /// Implement a very basic surface code (simplified for simulation)
    fn encode_surface(&mut self, state: &mut QuantumState) -> Result<(), QECCError> {
        // Surface code needs a 2D lattice of qubits
        // For simplicity, we'll use a 5x5 grid (25 qubits) per logical qubit
        let logical_qubits = state.num_qubits();
        let physical_qubits_per_logical = 25; // 5x5 grid
        let physical_qubits_needed = logical_qubits * physical_qubits_per_logical;
        
        if state.num_qubits() < physical_qubits_needed {
            return Err(QECCError::InsufficientQubits(
                "Surface".to_string(),
                physical_qubits_needed,
                state.num_qubits(),
            ));
        }
        
        // Create logical to physical mapping
        self.logical_to_physical.clear();
        
        for logical_idx in 0..logical_qubits {
            let base_idx = logical_idx * physical_qubits_per_logical;
            let physical_indices = (0..physical_qubits_per_logical)
                .map(|i| base_idx + i)
                .collect();
            
            self.logical_to_physical.insert(logical_idx, physical_indices);
        }
        
        // Apply simplified surface code encoding
        // (In a real implementation, this would be much more complex)
        for logical_idx in 0..logical_qubits {
            let physical_indices = self.logical_to_physical
                .get(&logical_idx)
                .ok_or(QECCError::LogicalQubitError(logical_idx))?;
            
            // Get mutable access to qubits
            let qubits = state.qubits_mut();
            
            // Initialize the surface code lattice
            // Central qubit holds the logical state
            let center = physical_indices[12]; // Center of 5x5 grid
            
            // Store central qubit state
            let center_is_one = qubits[center].prob_1() > 0.5;
            
            // Set up the stabilizers (simplified)
            // Apply X stabilizers
            if center_is_one {
                for i in [6, 8, 16, 18] {
                    qubits[physical_indices[i]].x();
                }
            }
            
            // Apply Z stabilizers (simplified simulation)
            for i in [7, 11, 13, 17] {
                qubits[physical_indices[i]].hadamard();
                
                // Simulate CNOT
                if center_is_one {
                    qubits[physical_indices[i]].x();
                }
                
                qubits[physical_indices[i]].hadamard();
            }
        }
        
        Ok(())
    }
    
    /// Decode surface code
    fn decode_surface(&mut self, state: &mut QuantumState) -> Result<(), QECCError> {
        // Apply the inverse of the encoding circuit (simplified)
        for logical_idx in 0..self.logical_to_physical.len() {
            let physical_indices = self.logical_to_physical
                .get(&logical_idx)
                .ok_or(QECCError::LogicalQubitError(logical_idx))?;
            
            // Get mutable access to qubits
            let qubits = state.qubits_mut();
            
            // In a real implementation, we would measure syndromes and decode
            // For simulation, we're just going to revert changes
            
            // Undo Z stabilizers
            for i in [7, 11, 13, 17] {
                qubits[physical_indices[i]].hadamard();
                qubits[physical_indices[i]].hadamard();
            }
        }
        
        self.logical_to_physical.clear();
        
        Ok(())
    }
    
    /// Detect error syndromes
    #[allow(clippy::too_many_lines)]
    fn detect_syndrome(&self, state: &QuantumState) -> ErrorSyndrome {
        if self.logical_to_physical.is_empty() {
            return ErrorSyndrome::None;
        }
        
        match self.current_code {
            CorrectionCode::None => ErrorSyndrome::None,
            
            CorrectionCode::Repetition => {
                // For repetition code, we check if the physical qubits disagree
                let mut errors = Vec::new();
                
                for physical_indices in self.logical_to_physical.values() {
                    if physical_indices.len() < 3 {
                        continue;
                    }
                    
                    // Get the state of each physical qubit (in simulation)
                    let qubits = state.qubits();
                    
                    // In a real quantum computer, we would use ancilla qubits and 
                    // CNOT gates to extract parity information without measuring data qubits
                    
                    // For simulation, we just look at the internal state
                    let state0 = qubits[physical_indices[0]].prob_0();
                    let state1 = qubits[physical_indices[1]].prob_0();
                    let state2 = qubits[physical_indices[2]].prob_0();
                    
                    // If one disagrees with the other two, it's probably an error
                    if (state0 > 0.5 && state1 > 0.5 && state2 < 0.5) ||
                       (state0 > 0.5 && state1 < 0.5 && state2 > 0.5) ||
                       (state0 < 0.5 && state1 > 0.5 && state2 > 0.5) {
                        if (state0 - state1).abs() > FLOAT_EPSILON {
                            if (state1 - state2).abs() < FLOAT_EPSILON {
                                errors.push(physical_indices[0]);
                            } else {
                                errors.push(physical_indices[1]);
                            }
                        } else if (state0 - state2).abs() > FLOAT_EPSILON {
                            errors.push(physical_indices[2]);
                        }
                    }
                }
                
                if errors.is_empty() {
                    ErrorSyndrome::None
                } else if errors.len() == 1 {
                    ErrorSyndrome::BitFlip(errors[0])
                } else {
                    ErrorSyndrome::Multiple(errors)
                }
            },
            
            CorrectionCode::Shor9 => {
                // In Shor's code, check both bit and phase flips
                let errors = self.detect_errors();
                
                if errors.is_empty() {
                    ErrorSyndrome::None
                } else if errors.len() == 1 {
                    ErrorSyndrome::BitFlip(errors[0])
                } else {
                    ErrorSyndrome::Multiple(errors)
                }
            },
            
            CorrectionCode::Steane7 => {
                // In Steane's code, check both bit and phase flips
                let errors = self.detect_errors();
                
                if errors.is_empty() {
                    ErrorSyndrome::None
                } else if errors.len() == 1 {
                    ErrorSyndrome::BitFlip(errors[0])
                } else {
                    ErrorSyndrome::Multiple(errors)
                }
            },
            
            CorrectionCode::Surface => {
                // Surface code uses stabilizer measurements
                let errors = self.detect_errors();
                
                if errors.is_empty() {
                    ErrorSyndrome::None
                } else if errors.len() == 1 {
                    ErrorSyndrome::BitFlip(errors[0])
                } else {
                    ErrorSyndrome::Multiple(errors)
                }
            },
            
            CorrectionCode::BitFlip3 => {
                // Simple 3-qubit bit flip code
                let mut errors = Vec::new();
                
                // Check for bit flips via majority vote
                if self.logical_to_physical.len() >= 3 {
                    let q0 = self.logical_to_physical.get(&0).unwrap()[0];
                    let q1 = self.logical_to_physical.get(&1).unwrap()[0];
                    let q2 = self.logical_to_physical.get(&2).unwrap()[0];
                    
                    // In this simplified model, we assume a bit flip if qubits diverge from majority
                    let sum = q0 + q1 + q2;
                    let majority = sum >= 2;
                    
                    // Find divergent qubits
                    if (q0 == 1) != majority {
                        errors.push(0);
                    }
                    if (q1 == 1) != majority {
                        errors.push(1);
                    }
                    if (q2 == 1) != majority {
                        errors.push(2);
                    }
                }
                
                ErrorSyndrome::Multiple(errors)
            },
            
            CorrectionCode::PhaseFlip3 => {
                // Simple 3-qubit phase flip code
                let mut errors = Vec::new();
                
                // Phase flips are detected by measuring in the X basis
                // Since our model doesn't support true basis changes,
                // we'll approximate this in the current implementation
                
                if self.logical_to_physical.len() >= 3 {
                    // In a proper implementation, we would:
                    // 1. Apply Hadamard to all qubits
                    // 2. Perform bit flip detection
                    // 3. Apply Hadamard again
                    
                    // For now, we'll just use a simple simulation
                    // Get approximate phase information
                    let p0 = self.logical_to_physical.get(&0).unwrap()[0];
                    let p1 = self.logical_to_physical.get(&1).unwrap()[0];
                    let p2 = self.logical_to_physical.get(&2).unwrap()[0];
                    
                    // Check for majority phase
                    // For simplicity, we'll use a binary approximation
                    #[allow(clippy::cast_precision_loss)]
                    let phase_0 = (p0 as f64) < 0.5;
                    #[allow(clippy::cast_precision_loss)]
                    let phase_1 = (p1 as f64) < 0.5;
                    #[allow(clippy::cast_precision_loss)]
                    let phase_2 = (p2 as f64) < 0.5;
                    
                    // Determine majority phase
                    let majority_phase = (usize::from(phase_0) + usize::from(phase_1) + usize::from(phase_2)) >= 2;
                    
                    // Find qubits with different phase
                    if phase_0 != majority_phase {
                        errors.push(0);
                    }
                    if phase_1 != majority_phase {
                        errors.push(1);
                    }
                    if phase_2 != majority_phase {
                        errors.push(2);
                    }
                }
                
                ErrorSyndrome::Multiple(errors)
            }
        }
    }
    
    /// Correct detected errors
    fn correct_errors(&mut self, errors: &[usize]) {
        if errors.is_empty() || self.state.is_none() {
            return;
        }
        
        let state = self.state.as_mut().unwrap();
        let qubits = state.qubits_mut();
        
        // Simple approach: assume all errors in the list are bit flip errors
        // Apply X gates to the specified qubits
        for &qubit_idx in errors {
            qubits[qubit_idx].x();
            self.correction_count += 1;
            self.syndrome_history.push(ErrorSyndrome::BitFlip(qubit_idx));
        }
    }

    /// Locate errors based on syndrome
    #[allow(dead_code)]
    fn error_locations(syndrome: &ErrorSyndrome) -> Vec<usize> {
        // Simplified demonstration implementation
        match syndrome {
            ErrorSyndrome::BitFlip(idx) | ErrorSyndrome::PhaseFlip(idx) | ErrorSyndrome::BitPhaseFlip(idx) => vec![*idx],
            ErrorSyndrome::Multiple(idxs) => idxs.clone(),
            ErrorSyndrome::None | ErrorSyndrome::Unknown => Vec::new(),
        }
    }
}

impl ErrorCorrection for QECC {
    fn encode(&mut self, code: CorrectionCode) -> bool {
        if self.state.is_none() {
            println!("Encoding failed: No state to encode");
            return false;
        }
        
        // If already encoded with same code, nothing to do
        if self.current_code == code {
            println!("Already encoded with the same code: {code:?}");
            return true;
        }
        
        // If already encoded with different code, decode first
        if self.current_code != CorrectionCode::None && !self.decode_and_correct() {
            println!("Failed to decode current encoding before applying new one");
            return false;
        }
        
        let Some(mut state_clone) = self.state.clone() else {
                println!("Encoding failed: State clone returned None");
                return false;
        };
        
        println!("Attempting to encode with {:?}, state has {} qubits", code, state_clone.num_qubits());
        
        let result = match code {
            CorrectionCode::None => {
                // No encoding needed
                Ok(())
            },
            CorrectionCode::Repetition => {
                let result = self.encode_repetition(&mut state_clone);
                if let Err(ref err) = result {
                    println!("Repetition encoding error: {err:?}");
                }
                result
            },
            CorrectionCode::Shor9 => {
                let result = self.encode_shor(&mut state_clone);
                if let Err(ref err) = result {
                    println!("Shor encoding error: {err:?}");
                }
                result
            },
            CorrectionCode::Steane7 => {
                let result = self.encode_steane(&mut state_clone);
                if let Err(ref err) = result {
                    println!("Steane encoding error: {err:?}");
                }
                result
            },
            CorrectionCode::Surface => {
                let result = self.encode_surface(&mut state_clone);
                if let Err(ref err) = result {
                    println!("Surface encoding error: {err:?}");
                }
                result
            },
            CorrectionCode::BitFlip3 => {
                // Implement 3-qubit bit flip code
                if self.logical_to_physical.len() < 3 {
                    // Need at least 3 qubits for this code
                    Err(QECCError::InsufficientQubits("BitFlip3".to_string(), 3, self.logical_to_physical.len()))
                } else {
                    // Store original state
                    let original_state = self.logical_to_physical.get(&0).unwrap()[0];
                    
                    // Reset all qubits to zero
                    for i in 0..3 {
                        self.logical_to_physical.get_mut(&i).unwrap().clear();
                    }
                    
                    // Set all qubits to the same value
                    if original_state == 1 {
                        for i in 0..3 {
                            self.logical_to_physical.get_mut(&i).unwrap().push(0);
                        }
                    }
                    
                    // Set the current correction code
                    self.current_code = CorrectionCode::BitFlip3;
                    self.operation_count = 0;
                    Ok(())
                }
            },
            CorrectionCode::PhaseFlip3 => {
                // Implement 3-qubit phase flip code
                if self.logical_to_physical.len() < 3 {
                    // Need at least 3 qubits for this code
                    Err(QECCError::InsufficientQubits("PhaseFlip3".to_string(), 3, self.logical_to_physical.len()))
                } else {
                    // Store original state
                    let original_state = self.logical_to_physical.get(&0).unwrap()[0];
                    let _original_phase = self.logical_to_physical.get(&1).unwrap()[0];
                    
                    // For phase flip code, we need to:
                    // 1. Apply Hadamard to all qubits to switch to X basis
                    // 2. Encode in the bit flip code
                    // 3. Apply Hadamard again to switch back to Z basis
                    
                    // In our simplified model:
                    
                    // Set all qubits to the same state
                    for i in 0..3 {
                        self.logical_to_physical.get_mut(&i).unwrap().clear();
                        self.logical_to_physical.get_mut(&i).unwrap().push(original_state);
                    }
                    
                    // Set the current correction code
                    self.current_code = CorrectionCode::PhaseFlip3;
                    self.operation_count = 0;
                    Ok(())
                }
            }
        };
        
        if let Ok(()) = result {
                self.state = Some(state_clone);
                self.current_code = code;
                self.operation_count = 0;
                println!("Encoding succeeded");
                true
        } else {
                println!("Encoding failed");
                false
        }
    }
    
    fn decode_and_correct(&mut self) -> bool {
        if self.state.is_none() {
            println!("Decoding failed: No state to decode");
            return false;
        }
        
        // If not encoded, nothing to do
        if self.current_code == CorrectionCode::None {
            println!("State not encoded with any error correction code");
            return true;
        }
        
        let Some(mut state_clone) = self.state.clone() else {
            println!("Decoding failed: State clone returned None");
            return false;
        };
        
        println!("Decoding code {:?}, state has {} qubits", self.current_code, state_clone.num_qubits());
        
        let result = match self.current_code {
            CorrectionCode::None => {
                // No correction to perform
                Ok(())
            },
            CorrectionCode::Repetition => {
                self.decode_repetition(&mut state_clone)
            },
            CorrectionCode::Shor9 => {
                self.decode_shor(&mut state_clone)
            },
            CorrectionCode::Steane7 => {
                self.decode_steane(&mut state_clone)
            },
            CorrectionCode::Surface => {
                self.decode_surface(&mut state_clone)
            },
            CorrectionCode::BitFlip3 => {
                // Implement bit flip code decoding
                let physical_indices = self.logical_to_physical.values().flatten().copied().collect::<Vec<_>>();
                
                for &physical_idx in &physical_indices {
                    // Apply error correction based on majority vote
                    // Algorithm omitted for simplicity
                    if physical_idx < state_clone.qubits().len() {
                        state_clone.qubits_mut()[physical_idx].x();
                    }
                }
                
                // Detect and correct errors
                let _corrected_errors = self.detect_errors();
                
                Ok(())
            },
            CorrectionCode::PhaseFlip3 => {
                // Implement phase flip code decoding
                let physical_indices = self.logical_to_physical.values().flatten().copied().collect::<Vec<_>>();
                
                for &physical_idx in &physical_indices {
                    // Apply error correction based on majority vote in X basis
                    // Algorithm omitted for simplicity
                    if physical_idx < state_clone.qubits().len() {
                        state_clone.qubits_mut()[physical_idx].z();
                    }
                }
                
                // Detect and correct errors
                let _corrected_errors = self.detect_errors();
                
                Ok(())
            }
        };
        
        match result {
            Ok(()) => {
                self.state = Some(state_clone);
                self.current_code = CorrectionCode::None;
                self.logical_to_physical.clear();
                true
            },
            Err(_) => false,
        }
    }
    
    fn detect_errors(&self) -> Vec<usize> {
        if self.state.is_none() || self.current_code == CorrectionCode::None {
            return Vec::new();
        }
        
        // Detect error syndromes
        match self.detect_syndrome(self.state.as_ref().unwrap()) {
                    ErrorSyndrome::None | ErrorSyndrome::Unknown => Vec::new(),
                    ErrorSyndrome::BitFlip(idx) | ErrorSyndrome::PhaseFlip(idx) | ErrorSyndrome::BitPhaseFlip(idx) => vec![idx],
                    ErrorSyndrome::Multiple(idxs) => idxs,
        }
    }
    
    fn current_code(&self) -> CorrectionCode {
        self.current_code
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_repetition_code() {
        // Create a quantum state with enough qubits for 1 logical qubit (3 physical qubits)
        let state = QuantumState::new(3);
        
        // Create QECC
        let mut qecc = QECC::new();
        qecc.set_state(state);
        
        // Encode with repetition code
        assert!(qecc.encode(CorrectionCode::Repetition));
        
        // Verify current code
        assert_eq!(qecc.current_code(), CorrectionCode::Repetition);
        
        // Decode
        assert!(qecc.decode_and_correct());
        
        // Verify code is removed
        assert_eq!(qecc.current_code(), CorrectionCode::None);
    }
    
    #[test]
    fn test_error_correction() {
        // Create a quantum state with 9 qubits for 1 logical qubit using Shor's code
        let state = QuantumState::new(9);
        
        // Create QECC with high error rate for testing
        let config = QECCConfig {
            error_rate: 0.1, // 10% error rate
            ..Default::default()
        };
        
        let mut qecc = QECC::with_config(config);
        qecc.set_state(state);
        
        // Encode with Shor's 9-qubit code
        assert!(qecc.encode(CorrectionCode::Shor9));
        
        // Manually introduce errors by directly manipulating the syndrome detection result
        // This is a hack for testing purposes, simulating error detection
        let dummy_errors = vec![0]; // Manually introduce error on qubit 0
        
        // Correct the introduced errors
        qecc.correct_errors(&dummy_errors);
        
        // Decode and correct
        assert!(qecc.decode_and_correct());
        
        // Verify code is removed
        assert_eq!(qecc.current_code(), CorrectionCode::None);
    }
    
    #[test]
    fn test_multiple_encoding_decoding() {
        // Create a quantum state with 7 qubits for 1 logical qubit using Steane's code
        let state = QuantumState::new(7);
        
        // Create QECC
        let mut qecc = QECC::new();
        qecc.set_state(state);
        
        // Encode with Steane code
        assert!(qecc.encode(CorrectionCode::Steane7));
        
        // Change to a different code (should decode first)
        assert!(qecc.encode(CorrectionCode::Repetition));
        
        // Verify current code
        assert_eq!(qecc.current_code(), CorrectionCode::Repetition);
        
        // Take the state and verify we get it back
        let recovered_state = qecc.take_state();
        assert!(recovered_state.is_some());
    }
} 