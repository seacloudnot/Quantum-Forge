// Quantum Decoherence Prevention Protocol (QDCP)
//
// This protocol maintains quantum coherence over extended periods through
// dynamic decoupling sequences, refreshing procedures, and environment isolation techniques.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::core::QuantumState;
use crate::util;

/// Errors specific to QDCP
#[derive(Error, Debug)]
pub enum QDCPError {
    /// Not enough qubits for decoupling sequence
    #[error("Insufficient qubits for decoupling sequence: need {0}, have {1}")]
    InsufficientQubits(usize, usize),
    
    /// Decoupling sequence invalid
    #[error("Invalid decoupling sequence: {0}")]
    InvalidSequence(String),
    
    /// State fidelity too low
    #[error("State fidelity too low for refreshing: {0:.4}")]
    LowFidelity(f64),
    
    /// Refreshing failed
    #[error("State refreshing failed: {0}")]
    RefreshFailed(String),
    
    /// Environment isolation error
    #[error("Environment isolation error: {0}")]
    IsolationError(String),
    
    /// Operation timeout
    #[error("Operation timed out after {0:?}")]
    Timeout(Duration),
}

/// Decoupling sequence type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecouplingSequence {
    /// Spin-echo (Hahn echo) - simple X pulse in middle of evolution
    SpinEcho,
    
    /// Carr-Purcell sequence - periodic X pulses
    CarrPurcell,
    
    /// Carr-Purcell-Meiboom-Gill sequence - improved CP sequence
    CPMG,
    
    /// Uhrig Dynamic Decoupling - non-equidistant pulses
    UDD,
    
    /// Concatenated Dynamical Decoupling
    CDD,
    
    /// Custom pulse sequence
    Custom,
}

/// Refreshing strategy for quantum states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefreshStrategy {
    /// Periodic refreshing
    Periodic,
    
    /// Threshold-based refreshing
    FidelityThreshold,
    
    /// Predictive refreshing
    Predictive,
    
    /// Entanglement-assisted refreshing
    EntanglementAssisted,
}

/// Configuration for QDCP
#[derive(Debug, Clone)]
pub struct QDCPConfig {
    /// Decoupling sequence to use
    pub sequence: DecouplingSequence,
    
    /// Number of pulses in the sequence
    pub pulse_count: usize,
    
    /// Interval between decoupling operations (in milliseconds)
    pub interval_ms: u64,
    
    /// Refreshing strategy
    pub refresh_strategy: RefreshStrategy,
    
    /// Minimum acceptable fidelity
    pub min_fidelity: f64,
    
    /// Simulated environment noise level
    pub noise_level: f64,
    
    /// Whether to automatically apply correction
    pub auto_correction: bool,
    
    /// Whether to use isolation
    pub use_isolation: bool,
}

impl Default for QDCPConfig {
    fn default() -> Self {
        Self {
            sequence: DecouplingSequence::CPMG,
            pulse_count: 8,
            interval_ms: 100,
            refresh_strategy: RefreshStrategy::FidelityThreshold,
            min_fidelity: 0.90,
            noise_level: 0.01,
            auto_correction: true,
            use_isolation: true,
        }
    }
}

/// Tracks the status of a protected quantum state
#[derive(Debug, Clone)]
pub struct ProtectedStateInfo {
    /// Unique ID for this protected state
    pub id: String,
    
    /// When the state was last refreshed
    pub last_refresh: Instant,
    
    /// Current estimated fidelity
    pub estimated_fidelity: f64,
    
    /// Refresh count
    pub refresh_count: usize,
    
    /// Decoupling operations count
    pub decoupling_count: usize,
    
    /// Whether state is isolated
    pub is_isolated: bool,
}

/// Main implementation of QDCP
pub struct QDCP {
    /// Configuration for QDCP
    config: QDCPConfig,
    
    /// Protected quantum states
    protected_states: HashMap<String, ProtectedStateInfo>,
    
    /// Custom pulse sequence (if used)
    custom_sequence: Option<Vec<char>>,
    
    /// Last operation timestamp
    last_operation: Instant,
    
    /// Currently protected state
    current_state: Option<QuantumState>,
}

impl Default for QDCP {
    fn default() -> Self {
        Self::new()
    }
}

impl QDCP {
    /// Create a new QDCP instance with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: QDCPConfig::default(),
            protected_states: HashMap::new(),
            custom_sequence: None,
            last_operation: Instant::now(),
            current_state: None,
        }
    }
    
    /// Create a new QDCP instance with custom configuration
    #[must_use]
    pub fn with_config(config: QDCPConfig) -> Self {
        Self {
            config,
            protected_states: HashMap::new(),
            custom_sequence: None,
            last_operation: Instant::now(),
            current_state: None,
        }
    }
    
    /// Set a custom pulse sequence
    ///
    /// # Errors
    ///
    /// Returns `QDCPError::InvalidSequence` if the sequence is empty or contains invalid pulse characters
    pub fn set_custom_sequence(&mut self, sequence: Vec<char>) -> Result<(), QDCPError> {
        // Validate that the sequence only contains valid pulse characters
        for pulse in &sequence {
            match pulse {
                'X' | 'Y' | 'Z' | 'I' => {}, // valid pulses
                _ => return Err(QDCPError::InvalidSequence(
                    format!("Invalid pulse character: {pulse}")
                )),
            }
        }
        
        if sequence.is_empty() {
            return Err(QDCPError::InvalidSequence(
                "Empty pulse sequence".to_string()
            ));
        }
        
        self.custom_sequence = Some(sequence);
        self.config.sequence = DecouplingSequence::Custom;
        
        Ok(())
    }
    
    /// Set the current state to protect
    pub fn set_state(&mut self, state: QuantumState) {
        // Create a new protected state info
        let state_id = util::generate_id("qdcp");
        
        let info = ProtectedStateInfo {
            id: state_id.clone(),
            last_refresh: Instant::now(),
            estimated_fidelity: state.fidelity(),
            refresh_count: 0,
            decoupling_count: 0,
            is_isolated: false,
        };
        
        self.protected_states.insert(state_id, info);
        self.current_state = Some(state);
        self.last_operation = Instant::now();
    }
    
    /// Get the current state
    #[must_use]
    pub fn state(&self) -> Option<&QuantumState> {
        self.current_state.as_ref()
    }
    
    /// Take the current state (removes it from protection)
    pub fn take_state(&mut self) -> Option<QuantumState> {
        self.current_state.take()
    }
    
    /// Apply dynamic decoupling to prevent decoherence
    ///
    /// # Panics
    ///
    /// Panics if `current_state` is Some but cannot be unwrapped
    ///
    /// # Errors
    ///
    /// Returns `QDCPError::InvalidSequence` if using a custom sequence that isn't defined
    pub fn apply_decoupling(&mut self) -> Result<f64, QDCPError> {
        if self.current_state.is_none() {
            return Ok(1.0); // No state to protect
        }
        
        // Get state_id before borrowing state
        let state_id = self.get_current_state_id().unwrap_or_default();
        
        let state = self.current_state.as_mut().unwrap();
        
        // Generate the pulse sequence based on configuration
        let pulses = match self.config.sequence {
            DecouplingSequence::SpinEcho => {
                // Simple X pulse in the middle
                vec!['X']
            },
            DecouplingSequence::CarrPurcell => {
                // Periodic X pulses
                vec!['X'; self.config.pulse_count]
            },
            DecouplingSequence::CPMG => {
                // CPMG sequence uses Y pulses instead of X
                vec!['Y'; self.config.pulse_count]
            },
            DecouplingSequence::UDD => {
                // UDD has non-uniform spacing, but we're just modeling the pulses here
                vec!['X'; self.config.pulse_count]
            },
            DecouplingSequence::CDD => {
                // Concatenated sequence (simplified)
                let mut sequence = Vec::new();
                for _ in 0..self.config.pulse_count / 4 {
                    sequence.extend_from_slice(&['X', 'Y', 'X', 'Y']);
                }
                sequence
            },
            DecouplingSequence::Custom => {
                if let Some(seq) = &self.custom_sequence {
                    seq.clone()
                } else {
                    return Err(QDCPError::InvalidSequence("No custom sequence defined".to_string()));
                }
            },
        };
        
        // Apply the pulse sequence to each qubit
        let qubits = state.qubits_mut();
        for qubit in qubits {
            for &pulse in &pulses {
                match pulse {
                    'X' => qubit.x(),
                    'Y' => qubit.y(),
                    'Z' => qubit.z(),
                    _ => {}, // Identity operation or invalid pulse
                }
            }
        }
        
        // Get current fidelity after applying pulses
        let current_fidelity = state.fidelity();
        
        // Update protected state info
        if !state_id.is_empty() {
            if let Some(info) = self.protected_states.get_mut(&state_id) {
                info.decoupling_count += 1;
                info.estimated_fidelity = current_fidelity;
            }
        }
        
        self.last_operation = Instant::now();
        
        // Return the current fidelity
        Ok(current_fidelity)
    }
    
    /// Refresh a quantum state to restore fidelity
    ///
    /// # Panics
    ///
    /// Panics if `current_state` is Some but cannot be unwrapped
    ///
    /// # Errors
    ///
    /// Returns `QDCPError::LowFidelity` if the current fidelity is too low for refreshing
    pub fn refresh_state(&mut self) -> Result<f64, QDCPError> {
        if self.current_state.is_none() {
            return Ok(1.0); // No state to refresh
        }
        
        // Get state_id before borrowing state
        let state_id = self.get_current_state_id().unwrap_or_default();
        
        let state = self.current_state.as_mut().unwrap();
        let current_fidelity = state.fidelity();
        
        // Check if the fidelity is too low for refreshing
        if current_fidelity < self.config.min_fidelity / 2.0 {
            return Err(QDCPError::LowFidelity(current_fidelity));
        }
        
        // Different refreshing strategies
        match self.config.refresh_strategy {
            RefreshStrategy::Periodic => {
                // Just restore fidelity (in a real system, this would use entanglement)
                state.set_fidelity(1.0);
            },
            RefreshStrategy::FidelityThreshold => {
                // Only refresh if below threshold
                if current_fidelity < self.config.min_fidelity {
                    state.set_fidelity(1.0);
                } else {
                    // No need to refresh
                    return Ok(current_fidelity);
                }
            },
            RefreshStrategy::Predictive => {
                // Predict future fidelity and refresh proactively
                // Using a safer approach to handle u128 to f64 conversion by computing in segments
                let micros = self.last_operation.elapsed().as_micros();
                
                // Calculate time in seconds to avoid precision loss
                // We calculate separately: whole milliseconds + fractional part
                let milliseconds = micros / 1000; // Whole milliseconds
                let fractional_ms = (micros % 1000) as f64 / 1000.0; // Fractional part
                
                // Convert the whole milliseconds to f64 safely
                // If milliseconds is too large for f64 precision, use a safe maximum
                let ms_f64 = if milliseconds > 9_007_199_254_740_991 { // 2^53-1 (max precise integer in f64)
                    9_007_199_254_740_991.0
                } else {
                    milliseconds as f64
                };
                
                // Convert to seconds for the time factor calculation
                let time_factor = (ms_f64 + fractional_ms) / 1000.0;
                let predicted_fidelity = current_fidelity * (1.0 - self.config.noise_level * time_factor);
                
                if predicted_fidelity < self.config.min_fidelity {
                    state.set_fidelity(1.0);
                } else {
                    // No need to refresh yet
                    return Ok(current_fidelity);
                }
            },
            RefreshStrategy::EntanglementAssisted => {
                // Simulate entanglement-assisted refreshing
                // In a real system, this would use entangled pairs
                state.set_fidelity(1.0);
                
                // Apply a small amount of noise to simulate imperfect refreshing
                state.apply_decoherence(self.config.noise_level * 0.1);
            },
        }
        
        // Get the new fidelity
        let new_fidelity = state.fidelity();
        
        // Update protected state info
        if !state_id.is_empty() {
            if let Some(info) = self.protected_states.get_mut(&state_id) {
                info.refresh_count += 1;
                info.last_refresh = Instant::now();
                info.estimated_fidelity = new_fidelity;
            }
        }
        
        self.last_operation = Instant::now();
        
        // Return the new fidelity
        Ok(new_fidelity)
    }
    
    /// Isolate a quantum state from environmental noise
    ///
    /// # Errors
    ///
    /// Returns `QDCPError::NoStateToProtect` if there is no state to isolate
    pub fn isolate_state(&mut self, isolate: bool) -> Result<(), QDCPError> {
        if self.current_state.is_none() {
            return Ok(()); // No state to isolate
        }
        
        // Get current state ID before any borrowing
        let state_id = self.get_current_state_id().unwrap_or_default();
        
        // In a real system, this would configure physical isolation mechanisms
        // For simulation, we'll just track the isolation state
        
        if let Some(info) = self.protected_states.get_mut(&state_id) {
            info.is_isolated = isolate;
                
            // If we're isolating, refresh the state first
            if isolate {
                return self.refresh_state().map(|_| ());
            }
        }
        
        Ok(())
    }
    
    /// Maintain the fidelity of the current quantum state
    ///
    /// # Panics
    ///
    /// Panics if `current_state` is Some but cannot be unwrapped
    ///
    /// # Errors
    ///
    /// Returns `QDCPError::NoStateToProtect` if there is no state to maintain
    pub fn maintain_fidelity(&mut self) -> Result<f64, QDCPError> {
        if self.current_state.is_none() {
            return Ok(1.0); // No state to maintain
        }
        
        // Make a copy of the current fidelity and check state
        let current_fidelity = {
            let state = self.current_state.as_ref().unwrap();
            state.fidelity()
        };
        
        // Get the elapsed time since last operation
        let elapsed = self.last_operation.elapsed();
        
        // Check if a refresh is needed
        let should_refresh = match self.config.refresh_strategy {
            RefreshStrategy::Periodic => {
                elapsed.as_millis() > u128::from(self.config.interval_ms)
            },
            RefreshStrategy::FidelityThreshold => {
                current_fidelity < self.config.min_fidelity
            },
            RefreshStrategy::Predictive => {
                // Using a safer approach to handle u128 to f64 conversion by computing in segments
                let micros = elapsed.as_micros();
                
                // Calculate time in seconds to avoid precision loss
                // We calculate separately: whole milliseconds + fractional part
                let milliseconds = micros / 1000; // Whole milliseconds
                let fractional_ms = (micros % 1000) as f64 / 1000.0; // Fractional part
                
                // Convert the whole milliseconds to f64 safely
                // If milliseconds is too large for f64 precision, use a safe maximum
                let ms_f64 = if milliseconds > 9_007_199_254_740_991 { // 2^53-1 (max precise integer in f64)
                    9_007_199_254_740_991.0
                } else {
                    milliseconds as f64
                };
                
                // Convert to seconds for the time factor calculation
                let time_factor = (ms_f64 + fractional_ms) / 1000.0;
                let predicted_fidelity = current_fidelity * (1.0 - self.config.noise_level * time_factor);
                predicted_fidelity < self.config.min_fidelity
            },
            RefreshStrategy::EntanglementAssisted => {
                elapsed.as_millis() > u128::from(self.config.interval_ms) || current_fidelity < self.config.min_fidelity
            },
        };
        
        // Apply refreshing if needed
        if should_refresh {
            return self.refresh_state();
        }
        
        // Apply decoupling if needed
        let should_decouple = elapsed.as_millis() > u128::from(self.config.interval_ms / 2);
        
        if should_decouple {
            return self.apply_decoupling();
        }
        
        Ok(current_fidelity)
    }
    
    /// Get information about a protected state
    #[must_use]
    pub fn get_state_info(&self, state_id: &str) -> Option<&ProtectedStateInfo> {
        self.protected_states.get(state_id)
    }
    
    /// Get the ID of the current protected state
    fn get_current_state_id(&self) -> Option<String> {
        self.protected_states.keys().next().cloned()
    }
    
    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &QDCPConfig {
        &self.config
    }
    
    /// Modify configuration
    pub fn config_mut(&mut self) -> &mut QDCPConfig {
        &mut self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_decoupling_sequence() {
        let state = QuantumState::new(2);
        let mut qdcp = QDCP::new();
        
        // Set the state
        qdcp.set_state(state);
        
        // Apply decoupling
        let fidelity = qdcp.apply_decoupling().unwrap();
        
        // Check that fidelity is high
        assert!(fidelity > 0.9, "Fidelity should be high after decoupling");
    }
    
    #[test]
    fn test_refresh_strategy() {
        let mut state = QuantumState::new(2);
        
        // Apply some decoherence
        state.apply_decoherence(0.2);
        
        let mut qdcp = QDCP::with_config(QDCPConfig {
            refresh_strategy: RefreshStrategy::FidelityThreshold,
            min_fidelity: 0.85,
            ..QDCPConfig::default()
        });
        
        // Set the state
        qdcp.set_state(state);
        
        // Should refresh because fidelity is below threshold
        let new_fidelity = qdcp.maintain_fidelity().unwrap();
        
        // Fidelity should be restored
        assert!(new_fidelity > 0.98, "Fidelity should be restored after refresh");
    }
    
    #[test]
    fn test_custom_sequence() {
        let state = QuantumState::new(2);
        let mut qdcp = QDCP::new();
        
        // Set a custom sequence
        qdcp.set_custom_sequence(vec!['X', 'Y', 'X', 'Y']).unwrap();
        
        // Set the state
        qdcp.set_state(state);
        
        // Apply decoupling with custom sequence
        let fidelity = qdcp.apply_decoupling().unwrap();
        
        // Check that fidelity is high
        assert!(fidelity > 0.9, "Fidelity should be high after custom decoupling");
    }
} 