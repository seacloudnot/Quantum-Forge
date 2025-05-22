// Quantum Entanglement Protocol (QEP)
//
// This protocol defines how quantum entanglement is established and maintained
// between nodes in a quantum blockchain network.

use crate::util;
use crate::util::InstantWrapper;
use std::collections::HashMap;
use std::fmt;
use std::time::Duration;
use async_trait::async_trait;
use thiserror::Error;
use rand::{Rng, thread_rng};
use serde::{Serialize, Deserialize};

/// Errors that can occur during entanglement operations
#[derive(Debug, Error)]
pub enum QEPError {
    #[error("Entanglement operation timeout after {0:?}")]
    Timeout(Duration),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Entanglement decoherence")]
    Decoherence,
    
    #[error("Invalid entanglement pair ID: {0}")]
    InvalidPairId(String),
    
    #[error("Node not found: {0}")]
    NodeNotFound(String),
    
    #[error("Insufficient resources")]
    InsufficientResources,
    
    #[error("Unauthorized operation")]
    Unauthorized,
    
    #[error("No entanglement exists between nodes")]
    NoEntanglement,
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Purpose of entanglement pair
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntanglementPurpose {
    /// General purpose entanglement
    General,
    
    /// Entanglement for quantum teleportation
    Teleportation,
    
    /// Entanglement for secure communication
    SecureCommunication,
    
    /// Entanglement for consensus
    Consensus,
    
    /// Entanglement for computation
    Computation,
    
    /// Entanglement for repeater operations
    Repeater,
}

/// Represents a quantum entanglement between two nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntanglementPair {
    /// Unique identifier for this entanglement pair
    pub id: String,
    
    /// ID of the first node
    pub node_a_id: String,
    
    /// ID of the second node
    pub node_b_id: String,
    
    /// Fidelity of the entanglement (0.0-1.0)
    pub fidelity: f64,
    
    /// When this entanglement was created
    pub creation_time: InstantWrapper,
    
    /// Expected lifetime in milliseconds
    pub lifetime_ms: u64,
    
    /// Purpose of this entanglement
    pub purpose: EntanglementPurpose,
    
    /// Whether this pair was created via entanglement swapping
    pub is_swapped: bool,
    
    /// IDs of original pairs if this was created via swapping
    pub source_pair_ids: Vec<String>,
    
    /// Metadata associated with the entanglement pair
    pub metadata: HashMap<String, String>,
}

impl EntanglementPair {
    /// Create a new entanglement pair
    pub fn new(node_a: &str, node_b: &str, purpose: EntanglementPurpose) -> Self {
        Self {
            id: util::generate_id("ent"),
            node_a_id: node_a.to_string(),
            node_b_id: node_b.to_string(),
            fidelity: 0.95 + thread_rng().gen_range(-0.05..0.02),
            creation_time: InstantWrapper::now(),
            lifetime_ms: 10000, // 10 seconds by default
            purpose,
            is_swapped: false,
            source_pair_ids: Vec::new(),
            metadata: HashMap::new(),
        }
    }
    
    /// Check if this entanglement has decohered
    pub fn is_decohered(&self) -> bool {
        #[allow(clippy::cast_possible_truncation)]
        let elapsed_ms = self.creation_time.elapsed().as_millis() as u64;
        elapsed_ms > self.lifetime_ms || self.fidelity < 0.5
    }
    
    /// Calculate remaining coherence as a percentage
    #[must_use]
    pub fn coherence_remaining(&self) -> f64 {
        if self.is_decohered() {
            return 0.0;
        }
        
        #[allow(clippy::cast_precision_loss)]
        let elapsed_ms = self.creation_time.elapsed().as_millis() as f64;
        #[allow(clippy::cast_precision_loss)]
        let lifetime_ms = self.lifetime_ms as f64;
        
        let time_factor = (lifetime_ms - elapsed_ms) / lifetime_ms;
        time_factor.max(0.0) * 100.0
    }
    
    /// Check if this entanglement connects the specified nodes
    pub fn connects(&self, node_a: &str, node_b: &str) -> bool {
        (self.node_a_id == node_a && self.node_b_id == node_b) ||
        (self.node_a_id == node_b && self.node_b_id == node_a)
    }
}

impl fmt::Display for EntanglementPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ent[{}] {} ⟷ {} (F={:.2}, {:?})",
            self.id,
            self.node_a_id,
            self.node_b_id,
            self.fidelity,
            self.purpose)
    }
}

/// Interface for entanglement protocol
#[async_trait]
pub trait EntanglementProtocol {
    /// Create entanglement between two nodes
    async fn create_entanglement(
        &mut self,
        node_a: &str,
        node_b: &str,
        purpose: EntanglementPurpose,
    ) -> Result<EntanglementPair, QEPError>;
    
    /// Check if entanglement exists between two nodes
    async fn check_entanglement(
        &self,
        node_a: &str,
        node_b: &str,
    ) -> Result<Option<EntanglementPair>, QEPError>;
    
    /// Swap entanglement to connect non-adjacent nodes
    async fn swap_entanglement(
        &mut self,
        pair_ab_id: &str,
        pair_bc_id: &str,
    ) -> Result<EntanglementPair, QEPError>;
    
    /// Refresh an existing entanglement to extend its lifetime
    async fn refresh_entanglement(
        &mut self,
        pair_id: &str,
    ) -> Result<EntanglementPair, QEPError>;
    
    /// Purify multiple entanglement pairs to create one higher-fidelity pair
    async fn purify_entanglement(
        &mut self,
        pair_ids: &[String],
    ) -> Result<EntanglementPair, QEPError>;
    
    /// Get all entanglement pairs for a node
    async fn get_node_entanglements(
        &self,
        node_id: &str,
    ) -> Result<Vec<EntanglementPair>, QEPError>;
}

/// Implementation of Quantum Entanglement Protocol
#[derive(Debug, Clone)]
pub struct QEP {
    /// ID of this node
    node_id: String,
    
    /// Active entanglement pairs
    pairs: HashMap<String, EntanglementPair>,
    
    /// Configuration
    config: QEPConfig,
    
    /// Entanglement capabilities (which nodes this node can create direct entanglement with)
    entanglement_capabilities: HashMap<String, f64>,
}

/// Configuration for QEP
#[derive(Debug, Clone)]
pub struct QEPConfig {
    /// Maximum number of entanglement pairs to maintain
    pub max_pairs: usize,
    
    /// Default lifetime for entanglement in milliseconds
    pub default_lifetime_ms: u64,
    
    /// Minimum acceptable fidelity
    pub min_fidelity: f64,
    
    /// Noise factor for decoherence simulation
    pub noise_factor: f64,
    
    /// Whether to simulate network delays
    pub simulate_delays: bool,
}

impl Default for QEPConfig {
    fn default() -> Self {
        Self {
            max_pairs: 100,
            default_lifetime_ms: 10000, // 10 seconds
            min_fidelity: 0.7,
            noise_factor: 0.01,
            simulate_delays: true,
        }
    }
}

impl QEP {
    /// Create a new QEP instance
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            pairs: HashMap::new(),
            config: QEPConfig::default(),
            entanglement_capabilities: HashMap::new(),
        }
    }
    
    /// Create a new QEP instance with custom configuration
    pub fn with_config(node_id: String, config: QEPConfig) -> Self {
        Self {
            node_id,
            pairs: HashMap::new(),
            config,
            entanglement_capabilities: HashMap::new(),
        }
    }
    
    /// Get this node's ID
    pub fn node_id(&self) -> &str {
        &self.node_id
    }
    
    /// Get the current configuration
    pub fn config(&self) -> &QEPConfig {
        &self.config
    }
    
    /// Get mutable access to the configuration
    pub fn config_mut(&mut self) -> &mut QEPConfig {
        &mut self.config
    }
    
    /// Get all entanglement pairs
    pub fn entanglement_pairs(&self) -> &HashMap<String, EntanglementPair> {
        &self.pairs
    }
    
    /// Get an entanglement pair by ID
    pub fn get_entanglement_pair(&self, pair_id: &str) -> Option<&EntanglementPair> {
        self.pairs.get(pair_id)
    }
    
    /// Get an entanglement pair by ID (alias for `get_entanglement_pair`)
    #[must_use]
    pub fn get_pair(&self, pair_id: &str) -> Option<&EntanglementPair> {
        self.get_entanglement_pair(pair_id)
    }
    
    /// Clean up decohered entanglement pairs
    pub fn cleanup_decohered(&mut self) -> usize {
        let before_count = self.pairs.len();
        
        self.pairs.retain(|_, pair| !pair.is_decohered());
        
        before_count - self.pairs.len()
    }
    
    /// Simulate network delay
    async fn simulate_delay(&self) -> Duration {
        if !self.config.simulate_delays {
            return Duration::from_millis(0);
        }
        
        let delay = thread_rng().gen_range(50..200);
        let duration = Duration::from_millis(delay);
        
        tokio::time::sleep(duration).await;
        
        duration
    }
    
    /// Check if this node can create direct entanglement with another node
    #[must_use]
    pub fn can_create_direct_entanglement(&self, node_a: &str, node_b: &str) -> bool {
        // Either we are one of the nodes and have capability with the other
        if node_a == self.node_id {
            self.entanglement_capabilities.contains_key(node_b)
        } else if node_b == self.node_id {
            self.entanglement_capabilities.contains_key(node_a)
        } else {
            // Or we're checking if two other nodes can entangle (less accurate)
            // For this, we just assume they can if they're different nodes
            node_a != node_b
        }
    }
    
    /// Add entanglement capability with a node
    pub fn add_entanglement_capability(&mut self, node_id: &str, fidelity: f64) {
        self.entanglement_capabilities.insert(node_id.to_string(), fidelity);
    }
    
    /// Get a mutable reference to an entanglement pair by ID
    pub fn get_entanglement_pair_mut(&mut self, pair_id: &str) -> Option<&mut EntanglementPair> {
        self.pairs.get_mut(pair_id)
    }
    
    /// Remove an entanglement pair by ID
    pub fn remove_entanglement_pair(&mut self, pair_id: &str) -> Option<EntanglementPair> {
        self.pairs.remove(pair_id)
    }
    
    /// Add an entanglement pair
    pub fn add_entanglement_pair(&mut self, pair: EntanglementPair) {
        self.pairs.insert(pair.id.clone(), pair);
    }
}

#[async_trait]
impl EntanglementProtocol for QEP {
    async fn create_entanglement(
        &mut self,
        node_a: &str,
        node_b: &str,
        purpose: EntanglementPurpose,
    ) -> Result<EntanglementPair, QEPError> {
        // First clean up any decohered pairs
        self.cleanup_decohered();
        
        // Check if we have capacity
        if self.pairs.len() >= self.config.max_pairs {
            return Err(QEPError::InsufficientResources);
        }
        
        // Simulate network delay
        self.simulate_delay().await;
        
        // Create the entanglement pair
        let mut pair = EntanglementPair::new(node_a, node_b, purpose);
        pair.lifetime_ms = self.config.default_lifetime_ms;
        
        // Store the pair
        let pair_id = pair.id.clone();
        self.pairs.insert(pair_id, pair.clone());
        
        Ok(pair)
    }
    
    async fn check_entanglement(
        &self,
        node_a: &str,
        node_b: &str,
    ) -> Result<Option<EntanglementPair>, QEPError> {
        // Simulate network delay
        self.simulate_delay().await;
        
        // Find a non-decohered pair connecting these nodes
        for pair in self.pairs.values() {
            if pair.connects(node_a, node_b) && !pair.is_decohered() {
                return Ok(Some(pair.clone()));
            }
        }
        
        Ok(None)
    }
    
    async fn swap_entanglement(
        &mut self,
        pair_ab_id: &str,
        pair_bc_id: &str,
    ) -> Result<EntanglementPair, QEPError> {
        // First clean up any decohered pairs
        self.cleanup_decohered();
        
        // Check if we have capacity
        if self.pairs.len() >= self.config.max_pairs {
            return Err(QEPError::InsufficientResources);
        }
        
        // Get the pairs
        let pair_ab = match self.pairs.get(pair_ab_id) {
            Some(pair) => {
                if pair.is_decohered() {
                    return Err(QEPError::Decoherence);
                }
                pair.clone()
            },
            None => return Err(QEPError::InvalidPairId(pair_ab_id.to_string())),
        };
        
        let pair_bc = match self.pairs.get(pair_bc_id) {
            Some(pair) => {
                if pair.is_decohered() {
                    return Err(QEPError::Decoherence);
                }
                pair.clone()
            },
            None => return Err(QEPError::InvalidPairId(pair_bc_id.to_string())),
        };
        
        // Check if the pairs can be swapped (must share a node)
        let (node_a, _node_b, node_c) = if pair_ab.node_b_id == pair_bc.node_a_id {
            (pair_ab.node_a_id.clone(), pair_ab.node_b_id.clone(), pair_bc.node_b_id.clone())
        } else if pair_ab.node_b_id == pair_bc.node_b_id {
            (pair_ab.node_a_id.clone(), pair_ab.node_b_id.clone(), pair_bc.node_a_id.clone())
        } else if pair_ab.node_a_id == pair_bc.node_a_id {
            (pair_ab.node_b_id.clone(), pair_ab.node_a_id.clone(), pair_bc.node_b_id.clone())
        } else if pair_ab.node_a_id == pair_bc.node_b_id {
            (pair_ab.node_b_id.clone(), pair_ab.node_a_id.clone(), pair_bc.node_a_id.clone())
        } else {
            return Err(QEPError::InvalidPairId("Pairs do not share a node".to_string()));
        };
        
        // Simulate network delay for the swap operation
        self.simulate_delay().await;
        
        // Create the swapped pair
        let mut swapped_pair = EntanglementPair::new(&node_a, &node_c, pair_ab.purpose);
        swapped_pair.lifetime_ms = self.config.default_lifetime_ms;
        swapped_pair.is_swapped = true;
        swapped_pair.source_pair_ids = vec![pair_ab_id.to_string(), pair_bc_id.to_string()];
        
        // Swapping reduces fidelity
        swapped_pair.fidelity = (pair_ab.fidelity * pair_bc.fidelity).sqrt();
        
        // Store the new pair
        let pair_id = swapped_pair.id.clone();
        self.pairs.insert(pair_id, swapped_pair.clone());
        
        Ok(swapped_pair)
    }
    
    async fn refresh_entanglement(
        &mut self,
        pair_id: &str,
    ) -> Result<EntanglementPair, QEPError> {
        // Get the pair
        let pair = match self.pairs.get(pair_id) {
            Some(pair) => pair.clone(),
            None => return Err(QEPError::InvalidPairId(pair_id.to_string())),
        };
        
        // Simulate network delay
        self.simulate_delay().await;
        
        if pair.is_decohered() {
            // If fully decohered, create a new pair
            self.create_entanglement(&pair.node_a_id, &pair.node_b_id, pair.purpose).await
        } else {
            // Otherwise, create a refreshed copy
            let mut refreshed = pair.clone();
            refreshed.id = util::generate_id("ent");
            refreshed.creation_time = InstantWrapper::now();
            
            // Slightly lower fidelity due to refresh operation
            refreshed.fidelity = (refreshed.fidelity * 0.95).min(0.98);
            
            // Store the refreshed pair
            let new_id = refreshed.id.clone();
            self.pairs.insert(new_id, refreshed.clone());
            
            Ok(refreshed)
        }
    }
    
    async fn purify_entanglement(
        &mut self,
        pair_ids: &[String],
    ) -> Result<EntanglementPair, QEPError> {
        if pair_ids.len() < 2 {
            return Err(QEPError::InsufficientResources);
        }
        
        // Get the pairs
        let mut pairs = Vec::new();
        let mut node_a = None;
        let mut node_b = None;
        
        for id in pair_ids {
            match self.pairs.get(id) {
                Some(pair) => {
                    if pair.is_decohered() {
                        return Err(QEPError::Decoherence);
                    }
                    
                    // Ensure all pairs connect the same nodes
                    if node_a.is_none() {
                        node_a = Some(pair.node_a_id.clone());
                        node_b = Some(pair.node_b_id.clone());
                    } else if !pair.connects(node_a.as_ref().unwrap(), node_b.as_ref().unwrap()) {
                        return Err(QEPError::InvalidPairId("Pairs do not connect same nodes".to_string()));
                    }
                    
                    pairs.push(pair.clone());
                },
                None => return Err(QEPError::InvalidPairId(id.to_string())),
            }
        }
        
        // Simulate network delay
        self.simulate_delay().await;
        
        // Create the purified pair
        let mut purified = EntanglementPair::new(
            node_a.as_ref().unwrap(),
            node_b.as_ref().unwrap(),
            pairs[0].purpose,
        );
        
        // Calculate purified fidelity
        // In a real system, this would depend on the purification protocol
        // For simulation, we use a simple model: combined fidelity increases
        // and is bounded by theoretical maximum
        #[allow(clippy::cast_precision_loss)]
        let avg_fidelity = pairs.iter().map(|p| p.fidelity).sum::<f64>() / pairs.len() as f64;
        let max_gain = (1.0 - avg_fidelity) * 0.5;
        purified.fidelity = (avg_fidelity + max_gain).min(0.99);
        
        // Store the purified pair
        let pair_id = purified.id.clone();
        self.pairs.insert(pair_id, purified.clone());
        
        Ok(purified)
    }
    
    async fn get_node_entanglements(
        &self,
        node_id: &str,
    ) -> Result<Vec<EntanglementPair>, QEPError> {
        // Simulate network delay
        self.simulate_delay().await;
        
        // Find all non-decohered pairs involving this node
        let entanglements = self.pairs.values()
            .filter(|pair| !pair.is_decohered() && 
                   (pair.node_a_id == node_id || pair.node_b_id == node_id))
            .cloned()
            .collect();
        
        Ok(entanglements)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_create_entanglement() {
        let mut qep = QEP::new("test-node".to_string());
        
        let result = qep.create_entanglement(
            "node-a",
            "node-b",
            EntanglementPurpose::General,
        ).await;
        
        assert!(result.is_ok());
        
        let pair = result.unwrap();
        assert_eq!(pair.node_a_id, "node-a");
        assert_eq!(pair.node_b_id, "node-b");
        assert!(!pair.is_decohered());
    }
    
    #[tokio::test]
    async fn test_swap_entanglement() {
        let mut qep = QEP::new("test-node".to_string());
        
        // Create two entanglement pairs
        let pair_ab = qep.create_entanglement(
            "node-a",
            "node-b",
            EntanglementPurpose::General,
        ).await.unwrap();
        
        let pair_bc = qep.create_entanglement(
            "node-b",
            "node-c",
            EntanglementPurpose::General,
        ).await.unwrap();
        
        // Swap the entanglement
        let result = qep.swap_entanglement(&pair_ab.id, &pair_bc.id).await;
        
        assert!(result.is_ok());
        
        let swapped = result.unwrap();
        assert_eq!(swapped.node_a_id, "node-a");
        assert_eq!(swapped.node_b_id, "node-c");
        assert!(swapped.is_swapped);
        assert!(!swapped.is_decohered());
    }
} 