//! # Utility Module
//!
//! This module provides utility functions and helpers for quantum protocols, including:
//!
//! - ID generation
//! - Quantum state utilities (fidelity, entropy calculation)
//! - Simulation utilities (delay, decoherence)
//! - Measurement helpers
//! - Time and serialization utilities
//! - Hashing functions
//!
//! These utilities are used throughout the codebase to simplify common operations
//! and ensure consistent behavior.

use rand::{Rng, thread_rng};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use sha2::{Sha256, Digest};
use serde::{Serialize, Deserialize};

/// Trait for converting objects to and from bytes
pub trait ByteConversion {
    /// Convert object to bytes
    fn to_bytes(&self) -> Vec<u8>;
    
    /// Create object from bytes
    /// 
    /// # Errors
    /// 
    /// Returns an error if bytes cannot be converted to the expected object type
    fn from_bytes(bytes: &[u8]) -> Result<Self, String> where Self: Sized;
}

/// Trait for collecting protocol metrics
pub trait ProtocolMetrics {
    /// Record a successful operation
    fn record_success(&mut self, operation: &str, duration: Duration);
    
    /// Record a failed operation
    fn record_failure(&mut self, operation: &str, reason: &str);
    
    /// Get average operation time
    fn average_time(&self, operation: &str) -> Option<Duration>;
    
    /// Get success rate for an operation
    fn success_rate(&self, operation: &str) -> f64;
}

/// Generate a random ID with a specified prefix
///
/// # Arguments
///
/// * `prefix` - A string prefix to make the ID more readable
///
/// # Returns
///
/// A string with the format "{prefix}-{random_hex}"
///
/// # Example
///
/// ```
/// use quantum_protocols::util::generate_id;
///
/// let id = generate_id("node");
/// assert!(id.starts_with("node-"));
/// ```
#[must_use]
pub fn generate_id(prefix: &str) -> String {
    let random = thread_rng().gen::<u64>();
    format!("{prefix}-{random:x}")
}

/// Generate a random ID with timestamp for uniqueness and traceability
///
/// Creates an ID that includes a timestamp, making it both unique and
/// traceable to a specific point in time.
///
/// # Arguments
///
/// * `prefix` - A string prefix to make the ID more readable
///
/// # Returns
///
/// A string with the format "{prefix}-{timestamp}-{random_hex}"
#[must_use]
pub fn generate_timestamped_id(prefix: &str) -> String {
    let random = thread_rng().gen::<u32>();
    let timestamp = timestamp_now();
    format!("{prefix}-{timestamp}-{random:x}")
}

/// Calculate fidelity between two probability distributions
///
/// # Arguments
///
/// * `p` - First probability distribution
/// * `q` - Second probability distribution
///
/// # Returns
///
/// The fidelity value between 0.0 and 1.0
#[must_use]
pub fn calculate_fidelity(p: &[f64], q: &[f64]) -> f64 {
    if p.len() != q.len() {
        return 0.0;
    }
    
    let mut sum = 0.0;
    for i in 0..p.len() {
        sum += (p[i] * q[i]).sqrt();
    }
    
    sum * sum
}

/// Calculate the von Neumann entropy of a probability distribution
///
/// # Arguments
///
/// * `probabilities` - Probability distribution
///
/// # Returns
///
/// The entropy value
#[must_use]
pub fn calculate_entropy(probabilities: &[f64]) -> f64 {
    probabilities.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum()
}

/// Simulate communication delay between nodes
///
/// # Arguments
///
/// * `distance` - Distance between nodes (arbitrary units)
/// * `speed_factor` - Simulation speed factor
///
/// # Returns
///
/// A Duration representing the simulated delay
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
pub fn simulate_delay(distance: f64, speed_factor: f64) -> Duration {
    // In a real quantum network, communication would be limited by
    // the speed of light and other factors
    let base_delay = distance / speed_factor;
    // Convert to u64 with bound checking to avoid truncation warnings
    let micros = (base_delay * 1000.0).clamp(0.0, u64::MAX as f64) as u64;
    Duration::from_micros(micros)
}

/// Apply decoherence to a fidelity value
///
/// # Arguments
///
/// * `fidelity` - Initial fidelity value
/// * `noise` - Noise parameter
/// * `duration` - Time duration for decoherence
///
/// # Returns
///
/// Reduced fidelity value after decoherence
#[must_use]
pub fn apply_decoherence(fidelity: f64, noise: f64, duration: Duration) -> f64 {
    let time_factor = duration.as_secs_f64();
    let decay = (-noise * time_factor).exp();
    
    // Mix with maximally mixed state
    let mixed_component = 1.0 - decay;
    fidelity * decay + 0.5 * mixed_component
}

/// Calculate success probability of quantum operation based on fidelity
///
/// # Arguments
///
/// * `fidelity` - Quantum state fidelity
/// * `operation_complexity` - Complexity factor of the operation (higher = more error-prone)
///
/// # Returns
///
/// A probability value between 0.0 and 1.0
#[must_use]
pub fn operation_success_probability(fidelity: f64, operation_complexity: f64) -> f64 {
    // Simple model: success probability decreases with operation complexity
    // and increases with fidelity
    let base_probability = fidelity * (1.0 - operation_complexity / 10.0);
    base_probability.clamp(0.0, 1.0)
}

/// Simulated Bell state measurement outcomes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BellMeasurement {
    /// |Φ+⟩ = (|00⟩ + |11⟩)/√2
    PhiPlus,
    
    /// |Φ-⟩ = (|00⟩ - |11⟩)/√2
    PhiMinus,
    
    /// |Ψ+⟩ = (|01⟩ + |10⟩)/√2
    PsiPlus,
    
    /// |Ψ-⟩ = (|01⟩ - |10⟩)/√2
    PsiMinus,
}

impl BellMeasurement {
    /// Convert the Bell measurement to a pair of bits
    #[must_use]
    pub const fn to_bits(&self) -> (bool, bool) {
        match self {
            Self::PhiPlus => (false, false),   // 00
            Self::PhiMinus => (false, true),   // 01
            Self::PsiPlus => (true, false),    // 10
            Self::PsiMinus => (true, true),    // 11
        }
    }
    
    /// Create a Bell measurement from a pair of bits
    #[must_use]
    pub const fn from_bits(bit1: bool, bit2: bool) -> Self {
        match (bit1, bit2) {
            (false, false) => Self::PhiPlus,
            (false, true) => Self::PhiMinus,
            (true, false) => Self::PsiPlus,
            (true, true) => Self::PsiMinus,
        }
    }
}

/// Perform a simulated Bell measurement on a pair of qubits
#[must_use]
pub fn bell_measurement() -> BellMeasurement {
    let outcome = thread_rng().gen_range(0..4);
    match outcome {
        0 => BellMeasurement::PhiPlus,
        1 => BellMeasurement::PhiMinus,
        2 => BellMeasurement::PsiPlus,
        _ => BellMeasurement::PsiMinus,
    }
}

/// Parse a quantum state from string representation
///
/// # Errors
///
/// Returns an error if the state string is invalid or cannot be parsed.
///
/// # Panics
///
/// This function will panic if an invalid binary state is encountered that 
/// cannot be parsed by `from_str_radix`.
pub fn parse_quantum_state(state_str: &str) -> Result<Vec<f64>, String> {
    let mut probabilities = Vec::new();
    
    for part in state_str.split('+') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        
        // Parse coefficient and state
        let parts: Vec<&str> = part.split('|').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid state format: {part}"));
        }
        
        let coeff_str = parts[0].trim();
        let coeff = if coeff_str.is_empty() {
            1.0
        } else {
            coeff_str.parse::<f64>().map_err(|e| format!("Invalid coefficient: {e}"))?
        };
        
        let state = parts[1].trim();
        if state.len() != 2 || !state.chars().all(|c| c == '0' || c == '1') {
            return Err(format!("Invalid state: {state}"));
        }
        
        // This can panic if state is not a valid binary number, but we've already checked that above
        let index = usize::from_str_radix(state, 2).unwrap();
        
        // Ensure vector is large enough
        while probabilities.len() <= index {
            probabilities.push(0.0);
        }
        
        probabilities[index] = coeff * coeff;
    }
    
    Ok(probabilities)
}

/// Format duration in a human-readable way
///
/// Returns a string representation of the duration in the most appropriate unit
/// (seconds, milliseconds or microseconds).
#[must_use]
pub fn format_duration(duration: Duration) -> String {
    if duration.as_secs() > 0 {
        format!("{:.2}s", duration.as_secs_f64())
    } else if duration.as_millis() > 0 {
        format!("{}ms", duration.as_millis())
    } else {
        format!("{}μs", duration.as_micros())
    }
}

/// Get current timestamp in milliseconds since UNIX EPOCH
///
/// # Returns
///
/// A u64 representation of the current time in milliseconds
#[must_use]
pub fn timestamp_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| {
            // Handle potential truncation explicitly
            u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
        })
        .unwrap_or(0)
}

/// Hash bytes using SHA-256
#[must_use]
pub fn hash_bytes(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

/// Convert a string to a hash and return as bytes
#[must_use]
pub fn hash_to_bytes(data: &str) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    hasher.finalize().to_vec()
}

/// Calculate mean of a list of values
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    
    let sum: f64 = values.iter().sum();
    
    let Ok(len) = u32::try_from(values.len()) else {
        #[allow(clippy::cast_precision_loss)]
        return Some(sum / values.len() as f64); // Fallback if length exceeds u32
    };
    
    Some(sum / f64::from(len))
}

/// Calculate standard deviation of a list of values
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn std_deviation(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    
    let mean = mean(values)?;
    
    let Ok(len) = u32::try_from(values.len()) else {
        #[allow(clippy::cast_precision_loss)]
        return Some(values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64); // Fallback if length exceeds u32
    };
    
    let variance = values.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / f64::from(len);
    
    Some(variance.sqrt())
}

/// Wrapper for Instant to make it serializable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstantWrapper {
    /// The timestamp represented as milliseconds since `UNIX_EPOCH`
    timestamp_ms: u64,
}

impl InstantWrapper {
    /// Create a new `InstantWrapper` with the current time
    #[must_use]
    pub fn now() -> Self {
        Self {
            timestamp_ms: timestamp_now(),
        }
    }
    
    /// Create an instant wrapper from a timestamp in milliseconds
    #[must_use]
    pub const fn from_timestamp_ms(timestamp_ms: u64) -> Self {
        Self { timestamp_ms }
    }
    
    /// Get the timestamp in milliseconds
    #[must_use]
    pub const fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
    
    /// Get the time elapsed since this instant was created
    #[must_use]
    pub fn elapsed(&self) -> std::time::Duration {
        let now = timestamp_now();
        let elapsed_ms = now.saturating_sub(self.timestamp_ms);
        std::time::Duration::from_millis(elapsed_ms)
    }
}

impl Default for InstantWrapper {
    fn default() -> Self {
        Self::now()
    }
}

impl From<std::time::Instant> for InstantWrapper {
    fn from(_: std::time::Instant) -> Self {
        Self::now()
    }
}

/// Calculate the fidelity of a quantum state with decoherence
///
/// # Arguments
///
/// * `fidelity` - Current fidelity (0.0-1.0)
/// * `decay` - Decay factor (0.0-1.0)
/// * `mixed_component` - Mixed state component (0.0-1.0)
///
/// # Returns
///
/// Updated fidelity value
#[must_use]
pub fn calculate_fidelity_with_decoherence(fidelity: f64, decay: f64, mixed_component: f64) -> f64 {
    fidelity.mul_add(decay, 0.5 * mixed_component)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generate_id() {
        let id = generate_id("test");
        assert!(id.starts_with("test-"));
        assert_ne!(id, generate_id("test")); // IDs should be unique
    }
    
    #[test]
    fn test_generate_timestamped_id() {
        let id = generate_timestamped_id("test");
        let parts: Vec<&str> = id.split('-').collect();
        assert_eq!(parts[0], "test");
        
        // Second part should be a timestamp
        assert!(parts[1].parse::<u64>().is_ok());
        
        // IDs should be unique
        assert_ne!(id, generate_timestamped_id("test"));
    }
    
    #[test]
    fn test_calculate_fidelity() {
        // Test with identical states (should have fidelity 1.0)
        let p = vec![1.0, 0.0, 0.0]; // State vector for |0⟩
        let q = vec![1.0, 0.0, 0.0]; // Same state vector
        
        // Use assert_approx_eq with epsilon instead of assert_eq for float comparison
        let epsilon = 1e-10;
        assert!((calculate_fidelity(&p, &q) - 1.0).abs() < epsilon);

        // Test with orthogonal states (should have fidelity 0.0)
        let p = vec![1.0, 0.0, 0.0]; // State vector for |0⟩
        let q = vec![0.0, 1.0, 0.0]; // State vector for |1⟩
        
        // Use assert_approx_eq with epsilon instead of assert_eq for float comparison
        assert!((calculate_fidelity(&p, &q) - 0.0).abs() < epsilon);
    }
    
    #[test]
    fn test_bell_measurement_conversion() {
        let measurement = BellMeasurement::PhiPlus;
        let (b1, b2) = measurement.to_bits();
        assert_eq!(BellMeasurement::from_bits(b1, b2), measurement);
        
        for measurement in [
            BellMeasurement::PhiPlus,
            BellMeasurement::PhiMinus,
            BellMeasurement::PsiPlus,
            BellMeasurement::PsiMinus,
        ] {
            let (b1, b2) = measurement.to_bits();
            assert_eq!(BellMeasurement::from_bits(b1, b2), measurement);
        }
    }
    
    #[test]
    fn test_operation_success_probability() {
        // High fidelity, low complexity should have high success probability
        let prob = operation_success_probability(0.99, 1.0);
        assert!(prob > 0.8);
        
        // Low fidelity, high complexity should have low success probability
        let prob = operation_success_probability(0.5, 8.0);
        assert!(prob < 0.2);
    }
} 