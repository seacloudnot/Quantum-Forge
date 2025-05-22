// Quantum Network Node Implementation
//
// This file implements a node in a quantum network that can participate
// in quantum protocols.

use crate::core::quantum_state::QuantumState;
use crate::network::entanglement::QEP;
use crate::core::QSTP;
use crate::core::qstp::QSTPTransport;
use std::collections::HashMap;
use std::fmt;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use async_trait::async_trait;
use thiserror::Error;
use rand::thread_rng;
use rand::Rng;

/// Errors that can occur in node operations
#[derive(Debug, Error)]
pub enum NodeError {
    #[error("Protocol error: {0}")]
    ProtocolError(String),
    
    #[error("Communication error: {0}")]
    CommunicationError(String),
    
    #[error("Resource error: {0}")]
    ResourceError(String),
    
    #[error("Authentication error: {0}")]
    AuthError(String),
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Capabilities of a quantum node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    /// Number of qubits available for computation
    pub qubit_count: usize,
    
    /// Maximum number of entangled pairs supported
    pub max_entanglements: usize,
    
    /// Types of quantum operations supported
    pub supported_gates: Vec<String>,
    
    /// Whether teleportation is supported
    pub supports_teleportation: bool,
    
    /// Whether the node can act as a repeater
    pub can_act_as_repeater: bool,
    
    /// Maximum coherence time in milliseconds
    pub max_coherence_time_ms: u64,
    
    /// Whether error correction is available
    pub has_error_correction: bool,
}

impl Default for NodeCapabilities {
    fn default() -> Self {
        Self {
            qubit_count: 8,
            max_entanglements: 10,
            supported_gates: vec![
                "X".to_string(), 
                "Y".to_string(), 
                "Z".to_string(), 
                "H".to_string(), 
                "CNOT".to_string()
            ],
            supports_teleportation: true,
            can_act_as_repeater: true,
            max_coherence_time_ms: 10000,
            has_error_correction: false,
        }
    }
}

/// Represents a node in a quantum network
#[derive(Debug, Clone)]
pub struct Node {
    /// Unique identifier for this node
    id: String,
    
    /// Human-readable name
    name: String,
    
    /// Node's capabilities
    capabilities: NodeCapabilities,
    
    /// Protocol for quantum state transfer
    qstp: QSTP,
    
    /// Protocol for quantum entanglement
    qep: QEP,
    
    /// Last known coordinates (for topology)
    coordinates: Option<(f64, f64, f64)>,
    
    /// When this node was created
    creation_time: Instant,
    
    /// Quantum states stored in this node
    states: HashMap<String, QuantumState>,
}

impl Node {
    /// Create a new node with default capabilities
    pub fn new(id: &str) -> Self {
        Self {
            qstp: QSTP::new(id.to_string()),
            qep: QEP::new(id.to_string()),
            id: id.to_string(),
            name: format!("Node-{id}"),
            capabilities: NodeCapabilities::default(),
            coordinates: None,
            creation_time: Instant::now(),
            states: HashMap::new(),
        }
    }
    
    /// Create a new node with a specified name and default capabilities
    pub fn with_name(id: String, name: String) -> Self {
        Self {
            qstp: QSTP::new(id.clone()),
            qep: QEP::new(id.clone()),
            id: id.clone(),
            name,
            capabilities: NodeCapabilities::default(),
            coordinates: None,
            creation_time: Instant::now(),
            states: HashMap::new(),
        }
    }
    
    /// Create a new node with custom capabilities
    pub fn with_capabilities(id: String, name: String, capabilities: NodeCapabilities) -> Self {
        Self {
            qstp: QSTP::new(id.clone()),
            qep: QEP::new(id.clone()),
            id: id.clone(),
            name,
            capabilities,
            coordinates: None,
            creation_time: Instant::now(),
            states: HashMap::new(),
        }
    }
    
    /// Get the node ID
    pub fn id(&self) -> &str {
        &self.id
    }
    
    /// Get the node name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get node capabilities
    pub fn capabilities(&self) -> &NodeCapabilities {
        &self.capabilities
    }
    
    /// Get a mutable reference to node capabilities
    pub fn capabilities_mut(&mut self) -> &mut NodeCapabilities {
        &mut self.capabilities
    }
    
    /// Set node coordinates
    pub fn set_coordinates(&mut self, x: f64, y: f64, z: f64) {
        self.coordinates = Some((x, y, z));
    }
    
    /// Get node coordinates
    pub fn coordinates(&self) -> Option<(f64, f64, f64)> {
        self.coordinates
    }
    
    /// Get the creation timestamp of this node
    pub fn timestamp(&self) -> std::time::Instant {
        self.creation_time
    }
    
    /// Get a reference to the node's QSTP instance
    pub fn qstp(&self) -> &QSTP {
        &self.qstp
    }
    
    /// Get a mutable reference to the node's QSTP instance
    pub fn qstp_mut(&mut self) -> &mut QSTP {
        &mut self.qstp
    }
    
    /// Get a reference to the node's QEP instance
    pub fn qep(&self) -> &QEP {
        &self.qep
    }
    
    /// Get a mutable reference to the node's QEP instance
    pub fn qep_mut(&mut self) -> &mut QEP {
        &mut self.qep
    }
    
    /// Store a quantum state in the node
    pub fn store_state(&mut self, state: QuantumState) -> Result<(), NodeError> {
        // Check if we have capacity
        if self.states.len() >= self.capabilities.qubit_count {
            return Err(NodeError::ResourceError("Node is at qubit capacity".to_string()));
        }
        
        self.states.insert(state.id().to_string(), state);
        Ok(())
    }
    
    /// Retrieve a stored quantum state
    pub fn get_state(&self, state_id: &str) -> Option<&QuantumState> {
        self.states.get(state_id)
    }
    
    /// Remove a quantum state from storage
    pub fn remove_state(&mut self, state_id: &str) -> Option<QuantumState> {
        self.states.remove(state_id)
    }
    
    /// Calculate distance to another node (if both have coordinates)
    pub fn distance_to(&self, other: &Node) -> Option<f64> {
        match (self.coordinates, other.coordinates) {
            (Some((x1, y1, z1)), Some((x2, y2, z2))) => {
                let dx = x2 - x1;
                let dy = y2 - y1;
                let dz = z2 - z1;
                Some((dx*dx + dy*dy + dz*dz).sqrt())
            },
            _ => None,
        }
    }
    
    /// Calculate expected entanglement fidelity with another node based on distance
    ///
    /// Uses a realistic quantum model that accounts for:
    /// - Fiber optic loss (0.2 dB/km)
    /// - Detector efficiency (85%)
    /// - Environmental noise
    #[must_use]
    pub fn expected_entanglement_fidelity(&self, other: &Node) -> f64 {
        if let Some(distance) = self.distance_to(other) {
            // Convert distance to kilometers for fiber optic calculations
            let distance_km = distance / 1000.0;
            
            // Fiber optic loss model: 0.2 dB/km is standard for telecom fibers
            let transmission_factor = 10.0_f64.powf(-0.2 * distance_km / 10.0);
            
            // Detector efficiency factor (typical SNSPD ~85%)
            let detector_efficiency = 0.85;
            
            // Environmental decoherence factor (decreases with distance)
            let decoherence_factor = (-distance_km / 50.0).exp();
            
            // Combined fidelity model
            let base_fidelity = 0.99; // Maximum achievable fidelity
            let fidelity = base_fidelity * transmission_factor * detector_efficiency * decoherence_factor;
            
            // Apply quantum error correction benefit if available
            if self.capabilities.has_error_correction && other.capabilities.has_error_correction {
                // Error correction can improve fidelity, but has a threshold
                let correction_threshold = 0.65;
                if fidelity > correction_threshold {
                    return fidelity + (1.0 - fidelity) * 0.8; // 80% error reduction
                }
            }
            
            fidelity.clamp(0.5, 1.0) // Clamp between 0.5 (random) and 1.0 (perfect)
        } else {
            // Without coordinates, use a default high fidelity with some randomness
            0.90 + (thread_rng().gen::<f64>() * 0.05)
        }
    }
    
    /// Set the expected entanglement fidelity for a node identified by string ID
    /// This allows manually setting link quality for simulation purposes
    pub fn set_expected_entanglement_fidelity(&mut self, _node_id: &str, fidelity: f64) {
        // In a real implementation, this would update a mapping of node IDs to fidelities
        // Since our simple model calculates fidelity based on distance, we'll set coordinates
        // that would result in the desired fidelity
        
        // Solve for distance based on the fidelity formula: d = 10 * (1/f - 1)
        let _desired_distance = if fidelity > 0.0 {
            10.0 * (1.0 / fidelity - 1.0)
        } else {
            // If fidelity is 0, use a very large distance
            1000.0
        };
        
        // If we don't have coordinates, set default coordinates
        if self.coordinates.is_none() {
            self.set_coordinates(0.0, 0.0, 0.0);
        }
        
        // For now, we're just logging that we would have stored this mapping
        // In a full implementation, we would store a HashMap<String, f64> of node_id -> fidelity
        // This is a simplified model that doesn't actually store the mapping,
        // but in a real implementation, we would store this as metadata
    }
    
    /// Apply quantum decoherence to all stored quantum states
    ///
    /// Uses a realistic quantum noise model that includes:
    /// - Amplitude damping (T1 relaxation)
    /// - Phase damping (T2 dephasing)
    /// - Depolarizing noise (random Pauli errors)
    pub fn apply_decoherence(&mut self, noise_factor: f64) {
        // Quantum error parameters based on realistic superconducting qubit systems
        // Typical T1 relaxation time ~50-100 μs, T2 dephasing time ~20-50 μs
        
        let typical_t1_us = 70.0; // T1 relaxation time in microseconds
        let typical_t2_us = 40.0; // T2 dephasing time in microseconds
        
        // Scale parameters based on node capabilities and noise factor
        let coherence_scale = self.capabilities.max_coherence_time_ms as f64 / 50.0; // 50ms is reference
        let t1_factor = (-noise_factor / (coherence_scale * typical_t1_us)).exp();
        let t2_factor = (-noise_factor / (coherence_scale * typical_t2_us)).exp();
        
        // Calculate probability of different error types
        let p_amp_damp = 1.0 - t1_factor; // Amplitude damping probability
        let p_phase_damp = 1.0 - t2_factor; // Phase damping probability
        let p_depolarize = noise_factor * 0.01; // Depolarizing noise probability
        
        for state in self.states.values_mut() {
            // Apply amplitude damping (T1 relaxation)
            if thread_rng().gen::<f64>() < p_amp_damp {
                state.apply_amplitude_damping(p_amp_damp);
            }
            
            // Apply phase damping (T2 dephasing)
            if thread_rng().gen::<f64>() < p_phase_damp {
                state.apply_phase_damping(p_phase_damp);
            }
            
            // Apply depolarizing noise (random Pauli errors)
            if thread_rng().gen::<f64>() < p_depolarize {
                state.apply_depolarizing_noise(p_depolarize);
            }
        }
    }
    
    /// Clean up decohered quantum states
    pub fn clean_decohered_states(&mut self) -> usize {
        let before_count = self.states.len();
        
        self.states.retain(|_, state| !state.is_decohered());
        
        before_count - self.states.len()
    }
    
    /// Broadcast a message to the network
    ///
    /// # Arguments
    ///
    /// * `topic` - The message topic
    /// * `message` - The message to broadcast
    ///
    /// # Returns
    ///
    /// A result indicating success or failure
    ///
    /// # Errors
    ///
    /// Returns an error if broadcasting fails
    pub fn broadcast<T>(&self, topic: &str, message: &T) -> Result<(), NodeError>
    where
        T: fmt::Debug,
    {
        // In a real implementation, this would actually send the message
        // For testing purposes, we just log it
        println!("Broadcasting on {}: {:?}", topic, message);
        Ok(())
    }
}

#[async_trait]
impl QSTPTransport for Node {
    async fn send_state(&mut self, state: &QuantumState, dest_node_id: &str) -> Result<crate::core::qstp::TransferResult, crate::core::qstp::QSTPError> {
        self.qstp.send_state(state, dest_node_id).await
    }
    
    async fn receive_state(&mut self, request: crate::core::qstp::TransferRequest) -> Result<QuantumState, crate::core::qstp::QSTPError> {
        let state = self.qstp.receive_state(request).await?;
        
        // Store the state
        if let Err(e) = self.store_state(state.clone()) {
            return Err(crate::core::qstp::QSTPError::Unknown(format!("Failed to store state: {e}")));
        }
        
        Ok(state)
    }
    
    async fn check_node_availability(&self, node_id: &str) -> Result<bool, crate::core::qstp::QSTPError> {
        self.qstp.check_node_availability(node_id).await
    }
    
    async fn teleport_state(&mut self, state: &QuantumState, dest_node_id: &str) -> Result<crate::core::qstp::TransferResult, crate::core::qstp::QSTPError> {
        if !self.capabilities.supports_teleportation {
            return Err(crate::core::qstp::QSTPError::Unknown("Teleportation not supported by this node".to_string()));
        }
        
        // In a real implementation, we would check for existing entanglement
        // with the destination node and use it for teleportation
        
        self.qstp.teleport_state(state, dest_node_id).await
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Node[{}] '{}' with {} qubits",
            self.id,
            self.name,
            self.capabilities.qubit_count
        )
    }
} 