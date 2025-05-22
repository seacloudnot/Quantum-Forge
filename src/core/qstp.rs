// Quantum State Transfer Protocol (QSTP)
//
// This protocol defines how quantum states are transmitted between nodes
// in a quantum blockchain network.

use crate::core::quantum_state::QuantumState;
use crate::util;
use crate::util::InstantWrapper;

use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};
use thiserror::Error;
use rand::{Rng, thread_rng};

/// Errors that can occur during quantum state transfer
#[derive(Debug, Error, Clone)]
pub enum QSTPError {
    #[error("State transfer timeout after {0:?}")]
    Timeout(Duration),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Quantum state decoherence")]
    Decoherence,
    
    #[error("Invalid state data")]
    InvalidStateData,
    
    #[error("Unauthorized transfer")]
    Unauthorized,
    
    #[error("State verification failed")]
    VerificationFailed,
    
    #[error("Node not found: {0}")]
    NodeNotFound(String),
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Result of a quantum state transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferResult {
    /// Whether the transfer was successful
    pub success: bool,
    
    /// The fidelity of the transferred state
    pub fidelity: f64,
    
    /// Duration of the transfer in milliseconds
    pub duration_ms: u64,
    
    /// ID of the transferred state
    pub state_id: String,
    
    /// Error message if unsuccessful
    pub error: Option<String>,
    
    /// Transaction ID for this transfer
    pub transaction_id: String,
}

impl fmt::Display for TransferResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.success {
            write!(f, "Transfer[{}] succeeded with fidelity {:.4} in {}ms", 
                self.transaction_id,
                self.fidelity,
                self.duration_ms)
        } else {
            write!(f, "Transfer[{}] failed: {}", 
                self.transaction_id,
                self.error.as_deref().unwrap_or("Unknown error"))
        }
    }
}

/// Request to transfer a quantum state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferRequest {
    /// ID of the source node
    pub source_node: String,
    
    /// ID of the destination node
    pub dest_node: String,
    
    /// Serialized quantum state data
    pub state_data: Vec<u8>,
    
    /// Number of qubits in the state
    pub num_qubits: usize,
    
    /// ID of the state being transferred
    pub state_id: String,
    
    /// When this request was created
    pub creation_time: InstantWrapper,
    
    /// Authentication token
    pub auth_token: String,
    
    /// Request ID
    pub request_id: String,
}

/// Configuration for QSTP
#[derive(Debug, Clone)]
pub struct QSTPConfig {
    /// Default timeout for transfers in milliseconds
    pub transfer_timeout_ms: u64,
    
    /// Whether to verify state fidelity
    pub verify_fidelity: bool,
    
    /// Minimum acceptable fidelity
    pub min_fidelity: f64,
    
    /// Whether to use entanglement for teleportation
    pub use_teleportation: bool,
    
    /// Default noise level for transfers
    pub noise_level: f64,
    
    /// Whether to simulate network delays
    pub simulate_delays: bool,
    
    /// Speed factor for simulated network (higher = faster)
    pub network_speed_factor: f64,
}

impl Default for QSTPConfig {
    fn default() -> Self {
        Self {
            transfer_timeout_ms: 5000, // 5 seconds
            verify_fidelity: true,
            min_fidelity: 0.75,
            use_teleportation: true,
            noise_level: 0.01,
            simulate_delays: true,
            network_speed_factor: 10.0,
        }
    }
}

/// Interface for quantum state transfer
#[async_trait]
pub trait QSTPTransport {
    /// Send a quantum state to a destination node
    async fn send_state(&mut self, state: &QuantumState, dest_node_id: &str) -> Result<TransferResult, QSTPError>;
    
    /// Receive a quantum state from a transfer request
    async fn receive_state(&mut self, request: TransferRequest) -> Result<QuantumState, QSTPError>;
    
    /// Check if a node is available for quantum communication
    async fn check_node_availability(&self, node_id: &str) -> Result<bool, QSTPError>;
    
    /// Send a quantum state via teleportation (using pre-shared entanglement)
    async fn teleport_state(&mut self, state: &QuantumState, dest_node_id: &str) -> Result<TransferResult, QSTPError>;
}

/// Implementation of Quantum State Transfer Protocol
#[derive(Debug, Clone)]
pub struct QSTP {
    /// Configuration for this QSTP instance
    config: QSTPConfig,
    
    /// ID of this node
    node_id: String,
    
    /// Known states received from other nodes
    received_states: HashMap<String, QuantumState>,
    
    /// Authentication tokens for other nodes
    auth_tokens: HashMap<String, String>,
    
    /// Record of transfer times for nodes
    transfer_times: HashMap<String, Vec<u64>>,
}

impl QSTP {
    /// Create a new QSTP instance
    pub fn new(node_id: String) -> Self {
        Self {
            config: QSTPConfig::default(),
            node_id,
            received_states: HashMap::new(),
            auth_tokens: HashMap::new(),
            transfer_times: HashMap::new(),
        }
    }
    
    /// Create a new QSTP instance with custom configuration
    pub fn with_config(node_id: String, config: QSTPConfig) -> Self {
        Self {
            config,
            node_id,
            received_states: HashMap::new(),
            auth_tokens: HashMap::new(),
            transfer_times: HashMap::new(),
        }
    }
    
    /// Get the node ID
    pub fn node_id(&self) -> &str {
        &self.node_id
    }
    
    /// Get a reference to the configuration
    pub fn config(&self) -> &QSTPConfig {
        &self.config
    }
    
    /// Get a mutable reference to the configuration
    pub fn config_mut(&mut self) -> &mut QSTPConfig {
        &mut self.config
    }
    
    /// Store an authentication token for a node
    pub fn store_auth_token(&mut self, node_id: String, token: String) {
        self.auth_tokens.insert(node_id, token);
    }
    
    /// Get an authentication token for a node
    pub fn get_auth_token(&self, node_id: &str) -> Option<&String> {
        self.auth_tokens.get(node_id)
    }
    
    /// Get a reference to a received state
    pub fn get_received_state(&self, state_id: &str) -> Option<&QuantumState> {
        self.received_states.get(state_id)
    }
    
    /// Get the average transfer time for a node in milliseconds
    pub fn avg_transfer_time(&self, node_id: &str) -> Option<u64> {
        if let Some(times) = self.transfer_times.get(node_id) {
            if !times.is_empty() {
                let sum: u64 = times.iter().sum();
                Some(sum / times.len() as u64)
            } else {
                None
            }
        } else {
            None
        }
    }
    
    /// Record a transfer time for a node
    fn record_transfer_time(&mut self, node_id: &str, time_ms: u64) {
        self.transfer_times.entry(node_id.to_string())
            .or_default()
            .push(time_ms);
        
        // Keep only the last 10 measurements
        let times = self.transfer_times.get_mut(node_id).unwrap();
        if times.len() > 10 {
            times.remove(0);
        }
    }
    
    /// Compute estimated fidelity loss for transferring to a node
    fn compute_fidelity_loss(&self, dest_node_id: &str, distance: Option<f64>) -> f64 {
        // In a real system, this would depend on the physical distance
        // and the quantum channel quality
        
        // Use average transfer time as a proxy for distance if not provided
        let distance_factor = if let Some(dist) = distance {
            dist / 100.0
        } else if let Some(avg_time) = self.avg_transfer_time(dest_node_id) {
            avg_time as f64 / 1000.0
        } else {
            0.1 // Default distance factor
        };
        
        // Calculate loss based on distance and noise level
        self.config.noise_level * distance_factor.min(1.0)
    }
    
    /// Simulate network delay for a transfer
    async fn simulate_delay(&self, dest_node_id: &str) -> Duration {
        if !self.config.simulate_delays {
            return Duration::from_millis(0);
        }
        
        // Use average transfer time if available
        let base_delay = if let Some(avg_time) = self.avg_transfer_time(dest_node_id) {
            avg_time
        } else {
            // Otherwise use a random delay between 50-200ms
            thread_rng().gen_range(50..200)
        };
        
        // Add some randomness
        let jitter = thread_rng().gen_range(-10..10);
        let delay = ((base_delay as i64 + jitter) as u64).max(10);
        
        let duration = Duration::from_millis(delay);
        tokio::time::sleep(duration).await;
        
        duration
    }
    
    /// Verify the fidelity of a quantum state
    fn verify_fidelity(&self, state: &QuantumState) -> bool {
        state.fidelity() >= self.config.min_fidelity
    }
    
    /// Create a transfer request for a quantum state
    fn create_transfer_request(&self, state: &QuantumState, dest_node_id: &str) -> TransferRequest {
        let auth_token = self.auth_tokens.get(dest_node_id)
            .cloned()
            .unwrap_or_else(|| "anonymous".to_string());
        
        TransferRequest {
            source_node: self.node_id.clone(),
            dest_node: dest_node_id.to_string(),
            state_data: state.serialize(),
            num_qubits: state.num_qubits(),
            state_id: state.id().to_string(),
            creation_time: InstantWrapper::now(),
            auth_token,
            request_id: util::generate_id("request"),
        }
    }
    
    /// Handle receiving a quantum state
    fn handle_receive_state(&mut self, request: &TransferRequest) -> Result<QuantumState, QSTPError> {
        // Check if this is meant for us
        if request.dest_node != self.node_id {
            return Err(QSTPError::Unauthorized);
        }
        
        // Validate auth token if we have one
        if let Some(expected_token) = self.auth_tokens.get(&request.source_node) {
            if *expected_token != request.auth_token {
                return Err(QSTPError::Unauthorized);
            }
        }
        
        // Deserialize the state
        let state = QuantumState::deserialize(&request.state_data, request.num_qubits)
            .map_err(|_e| QSTPError::InvalidStateData)?;
        
        // Check state coherence
        if state.is_decohered() {
            return Err(QSTPError::Decoherence);
        }
        
        // Verify fidelity if configured
        if self.config.verify_fidelity && !self.verify_fidelity(&state) {
            return Err(QSTPError::VerificationFailed);
        }
        
        // Store the state
        self.received_states.insert(state.id().to_string(), state.clone());
        
        Ok(state)
    }
}

#[async_trait]
impl QSTPTransport for QSTP {
    async fn send_state(&mut self, state: &QuantumState, dest_node_id: &str) -> Result<TransferResult, QSTPError> {
        let start_time = Instant::now();
        
        // Create the transfer request
        let _request = self.create_transfer_request(state, dest_node_id);
        
        // Simulate network delay
        let _delay = self.simulate_delay(dest_node_id).await;
        
        // Calculate fidelity loss
        let fidelity_loss = self.compute_fidelity_loss(dest_node_id, None);
        
        // Calculate remaining fidelity
        let fidelity = (state.fidelity() * (1.0 - fidelity_loss)).max(0.0);
        
        // Check if the transfer is successful
        let success = fidelity >= self.config.min_fidelity;
        
        // Record transfer time
        let duration = start_time.elapsed();
        let duration_ms = duration.as_millis() as u64;
        self.record_transfer_time(dest_node_id, duration_ms);
        
        // Create the result
        let result = TransferResult {
            success,
            fidelity,
            duration_ms,
            state_id: state.id().to_string(),
            error: if !success {
                Some("Insufficient fidelity".to_string())
            } else {
                None
            },
            transaction_id: util::generate_id("tx"),
        };
        
        if success {
            Ok(result)
        } else {
            Err(QSTPError::VerificationFailed)
        }
    }
    
    async fn receive_state(&mut self, request: TransferRequest) -> Result<QuantumState, QSTPError> {
        // Check if the request has timed out
        let timeout = Duration::from_millis(self.config.transfer_timeout_ms);
        let elapsed = request.creation_time.elapsed();
        
        if elapsed > timeout {
            return Err(QSTPError::Timeout(elapsed));
        }
        
        // Handle the request
        self.handle_receive_state(&request)
    }
    
    async fn check_node_availability(&self, node_id: &str) -> Result<bool, QSTPError> {
        // Simulate network check
        let delay = if self.config.simulate_delays {
            let base_delay = thread_rng().gen_range(10..50);
            Duration::from_millis(base_delay)
        } else {
            Duration::from_millis(0)
        };
        
        tokio::time::sleep(delay).await;
        
        // In a real system, this would perform an actual network check
        // For simulation, we just check if we have a token for this node
        let _has_token = self.auth_tokens.contains_key(node_id);
        
        // For the demo, we'll return true even for unknown nodes
        Ok(true)
    }
    
    async fn teleport_state(&mut self, state: &QuantumState, dest_node_id: &str) -> Result<TransferResult, QSTPError> {
        let start_time = Instant::now();
        
        // In a real system, this would use quantum teleportation
        // with pre-shared entanglement between nodes
        
        // For simulation, we'll just make it slightly faster and higher fidelity
        // than regular state transfer
        
        // Create the transfer request
        let _request = self.create_transfer_request(state, dest_node_id);
        
        // Simulate network delay, but shorter for teleportation
        let normal_delay = self.simulate_delay(dest_node_id).await;
        let _teleport_delay = normal_delay / 2;
        
        // Teleportation has less fidelity loss
        let normal_loss = self.compute_fidelity_loss(dest_node_id, None);
        let teleport_loss = normal_loss * 0.5;
        
        // Calculate remaining fidelity
        let fidelity = (state.fidelity() * (1.0 - teleport_loss)).max(0.0);
        
        // Check if the transfer is successful
        let success = fidelity >= self.config.min_fidelity;
        
        // Record transfer time
        let duration = start_time.elapsed();
        let duration_ms = duration.as_millis() as u64;
        self.record_transfer_time(dest_node_id, duration_ms);
        
        // Create the result
        let result = TransferResult {
            success,
            fidelity,
            duration_ms,
            state_id: state.id().to_string(),
            error: if !success {
                Some("Insufficient fidelity".to_string())
            } else {
                None
            },
            transaction_id: util::generate_id("tx"),
        };
        
        if success {
            Ok(result)
        } else {
            Err(QSTPError::VerificationFailed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_send_receive_state() {
        let mut sender = QSTP::new("sender-node".to_string());
        let mut receiver = QSTP::new("receiver-node".to_string());
        
        // Set up auth tokens
        let token = "test-token".to_string();
        sender.store_auth_token("receiver-node".to_string(), token.clone());
        receiver.store_auth_token("sender-node".to_string(), token);
        
        // Create a test state
        let state = QuantumState::new(2);
        
        // Send the state
        let result = sender.send_state(&state, "receiver-node").await;
        
        assert!(result.is_ok());
        
        let transfer_result = result.unwrap();
        assert!(transfer_result.success);
        assert!(transfer_result.fidelity > 0.9);
    }
    
    #[tokio::test]
    async fn test_teleportation() {
        let mut sender = QSTP::new("sender-node".to_string());
        let mut receiver = QSTP::new("receiver-node".to_string());
        
        // Set up auth tokens
        let token = "test-token".to_string();
        sender.store_auth_token("receiver-node".to_string(), token.clone());
        receiver.store_auth_token("sender-node".to_string(), token);
        
        // Create a test state
        let state = QuantumState::new(2);
        
        // Teleport the state
        let result = sender.teleport_state(&state, "receiver-node").await;
        
        assert!(result.is_ok());
        
        let teleport_result = result.unwrap();
        assert!(teleport_result.success);
        assert!(teleport_result.fidelity > 0.9);
    }
    
    #[tokio::test]
    async fn test_node_availability() {
        let qstp = QSTP::new("test-node".to_string());
        
        // Check availability
        let result = qstp.check_node_availability("other-node").await;
        
        assert!(result.is_ok());
        assert!(result.unwrap());
    }
} 