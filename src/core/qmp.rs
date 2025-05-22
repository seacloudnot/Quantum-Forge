// Quantum Measurement Protocol (QMP)
//
// This module implements the Quantum Measurement Protocol, which standardizes
// measurement procedures and coordination across quantum network nodes.

use std::collections::HashMap;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use crate::core::{Qubit, QuantumRegister};
use crate::error::Result;

/// Represents different measurement bases
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MeasurementBasis {
    /// Standard computational basis (Z-basis: |0⟩, |1⟩)
    Computational,
    /// Hadamard basis (X-basis: |+⟩, |-⟩)
    Hadamard,
    /// Y-basis (|+i⟩, |-i⟩)
    YBasis,
    /// Custom basis with rotation angles (theta, phi)
    Custom(f64, f64),
}

/// Configuration for the QMP protocol
#[derive(Debug, Clone)]
pub struct QMPConfig {
    /// Default measurement basis to use
    pub default_basis: MeasurementBasis,
    /// Whether to use synchronized measurement timing
    pub synchronized_timing: bool,
    /// Timeout for synchronized measurements (in milliseconds)
    pub sync_timeout_ms: u64,
    /// Random seed for deterministic testing (None for true randomness)
    pub random_seed: Option<u64>,
}

impl Default for QMPConfig {
    fn default() -> Self {
        Self {
            default_basis: MeasurementBasis::Computational,
            synchronized_timing: false,
            sync_timeout_ms: 1000,
            random_seed: None,
        }
    }
}

/// Result of a measurement operation
#[derive(Debug, Clone)]
pub struct MeasurementResult {
    /// The basis used for measurement
    pub basis: MeasurementBasis,
    /// The measurement outcome (typically 0 or 1 for a single qubit)
    pub outcome: Vec<u8>,
    /// Confidence/fidelity of the measurement (0.0-1.0)
    pub fidelity: f64,
    /// Timestamp when measurement occurred
    pub timestamp: std::time::Instant,
}

/// Main implementation of the Quantum Measurement Protocol
pub struct QMP {
    /// Configuration for this QMP instance
    config: QMPConfig,
    /// Random number generator for measurement decisions
    rng: StdRng,
    /// History of measurements performed
    measurement_history: Vec<MeasurementResult>,
    /// Map of node IDs to their selected measurement bases
    coordinated_bases: HashMap<String, MeasurementBasis>,
}

impl QMP {
    /// Create a new QMP instance with the given configuration
    pub fn new(config: QMPConfig) -> Self {
        let rng = match config.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        
        Self {
            config,
            rng,
            measurement_history: Vec::new(),
            coordinated_bases: HashMap::new(),
        }
    }
    
    /// Measure a single qubit with the specified basis
    pub fn measure_qubit(&mut self, qubit: &mut Qubit, basis: Option<MeasurementBasis>) -> Result<MeasurementResult> {
        let basis = basis.unwrap_or(self.config.default_basis);
        
        // Apply basis transformation before measurement if needed
        match basis {
            MeasurementBasis::Computational => {
                // No transformation needed for computational basis
            },
            MeasurementBasis::Hadamard => {
                qubit.hadamard();
            },
            MeasurementBasis::YBasis => {
                // Apply rotation for Y-basis measurement
                // For S† followed by H, we can use a phase gate and hadamard
                qubit.phase(-std::f64::consts::FRAC_PI_2);
                qubit.hadamard();
            },
            MeasurementBasis::Custom(theta, phi) => {
                // Apply custom rotation - simulate with phase and hadamard
                qubit.phase(phi);
                qubit.hadamard();
                qubit.phase(theta);
            }
        }
        
        // Perform the measurement
        let outcome = vec![qubit.measure()];
        
        // Create measurement result
        let result = MeasurementResult {
            basis,
            outcome,
            fidelity: 1.0, // Assuming perfect measurement for now
            timestamp: std::time::Instant::now(),
        };
        
        // Store in history
        self.measurement_history.push(result.clone());
        
        Ok(result)
    }
    
    /// Measure a quantum register with the specified basis
    pub fn measure_register(&mut self, register: &mut QuantumRegister, basis: Option<MeasurementBasis>) -> Result<MeasurementResult> {
        let basis = basis.unwrap_or(self.config.default_basis);
        let mut outcomes = Vec::with_capacity(register.size());
        
        // Apply basis transformation before measurement if needed
        for i in 0..register.size() {
            if let Some(qubit) = register.qubit_mut(i) {
                match basis {
                    MeasurementBasis::Computational => {
                        // No transformation needed for computational basis
                    },
                    MeasurementBasis::Hadamard => {
                        qubit.hadamard();
                    },
                    MeasurementBasis::YBasis => {
                        // Apply rotation for Y-basis measurement
                        // For S† followed by H, we can use a phase gate and hadamard
                        qubit.phase(-std::f64::consts::FRAC_PI_2);
                        qubit.hadamard();
                    },
                    MeasurementBasis::Custom(theta, phi) => {
                        // Apply custom rotation - simulate with phase and hadamard
                        qubit.phase(phi);
                        qubit.hadamard();
                        qubit.phase(theta);
                    }
                }
            }
        }
        
        // Perform the measurements
        for i in 0..register.size() {
            if let Some(result) = register.measure(i) {
                outcomes.push(result);
            }
        }
        
        // Create measurement result
        let result = MeasurementResult {
            basis,
            outcome: outcomes,
            fidelity: 1.0, // Assuming perfect measurement for now
            timestamp: std::time::Instant::now(),
        };
        
        // Store in history
        self.measurement_history.push(result.clone());
        
        Ok(result)
    }
    
    /// Coordinate measurement basis between multiple nodes
    pub fn coordinate_basis(&mut self, nodes: &[String], suggested_basis: Option<MeasurementBasis>) -> MeasurementBasis {
        let basis = suggested_basis.unwrap_or_else(|| {
            // Randomly select a basis if none suggested
            match self.rng.gen_range(0..3) {
                0 => MeasurementBasis::Computational,
                1 => MeasurementBasis::Hadamard,
                _ => MeasurementBasis::YBasis,
            }
        });
        
        // Record the selected basis for each node
        for node in nodes {
            self.coordinated_bases.insert(node.clone(), basis);
        }
        
        basis
    }
    
    /// Get measurement basis for a specific node
    pub fn get_node_basis(&self, node_id: &str) -> Option<MeasurementBasis> {
        self.coordinated_bases.get(node_id).copied()
    }
    
    /// Get the measurement history
    pub fn history(&self) -> &[MeasurementResult] {
        &self.measurement_history
    }
    
    /// Clear measurement history
    pub fn clear_history(&mut self) {
        self.measurement_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_measure_in_computational_basis() {
        // Create a deterministic QMP instance for testing
        let config = QMPConfig {
            random_seed: Some(42),
            ..Default::default()
        };
        let mut qmp = QMP::new(config);
        
        // Create a qubit in state |0⟩
        let mut qubit = Qubit::new();
        
        // Measure in computational basis
        let result = qmp.measure_qubit(&mut qubit, Some(MeasurementBasis::Computational)).unwrap();
        
        // Check the result
        assert_eq!(result.basis, MeasurementBasis::Computational);
        assert_eq!(result.outcome[0], 0); // |0⟩ state should measure as 0
        
        // Create a qubit in state |1⟩
        let mut qubit = Qubit::new();
        qubit.x();
        
        // Measure in computational basis
        let result = qmp.measure_qubit(&mut qubit, Some(MeasurementBasis::Computational)).unwrap();
        
        // Check the result
        assert_eq!(result.basis, MeasurementBasis::Computational);
        assert_eq!(result.outcome[0], 1); // |1⟩ state should measure as 1
    }
    
    #[test]
    fn test_measure_in_hadamard_basis() {
        // Create a deterministic QMP instance for testing
        let config = QMPConfig {
            random_seed: Some(42),
            ..Default::default()
        };
        let mut qmp = QMP::new(config);
        
        // Create a qubit in state |+⟩
        let mut qubit = Qubit::new();
        qubit.hadamard();
        
        // Measure in Hadamard basis
        let result = qmp.measure_qubit(&mut qubit, Some(MeasurementBasis::Hadamard)).unwrap();
        
        // Check the result
        assert_eq!(result.basis, MeasurementBasis::Hadamard);
        assert_eq!(result.outcome[0], 0); // |+⟩ state should measure as 0 in Hadamard basis
    }
    
    #[test]
    fn test_measurement_coordination() {
        // Create a QMP instance
        let mut qmp = QMP::new(QMPConfig::default());
        
        // Define nodes
        let nodes = vec!["alice".to_string(), "bob".to_string(), "charlie".to_string()];
        
        // Coordinate basis
        let basis = qmp.coordinate_basis(&nodes, Some(MeasurementBasis::Hadamard));
        
        // Check that all nodes have the same basis
        assert_eq!(basis, MeasurementBasis::Hadamard);
        assert_eq!(qmp.get_node_basis("alice"), Some(MeasurementBasis::Hadamard));
        assert_eq!(qmp.get_node_basis("bob"), Some(MeasurementBasis::Hadamard));
        assert_eq!(qmp.get_node_basis("charlie"), Some(MeasurementBasis::Hadamard));
    }
} 