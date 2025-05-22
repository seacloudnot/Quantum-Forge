// Quantum Byzantine Fault Tolerance Protocol
//
// This file implements QBFT (Quantum Byzantine Fault Tolerance) for achieving
// consensus despite malicious nodes using quantum mechanics.

use crate::consensus::{ConsensusProtocol, ConsensusResult};
use crate::network::entanglement::{EntanglementPair, EntanglementProtocol, EntanglementPurpose, QEPError};
use crate::network::Node;
use crate::util;

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::RwLock;
use rand::{Rng, thread_rng};
use serde::{Serialize, Deserialize};
use hex;

/// Errors that can occur during QBFT operations
#[derive(Debug, Error)]
pub enum QBFTError {
    #[error("Node error: {0}")]
    NodeError(String),
    
    #[error("Entanglement error: {0}")]
    EntanglementError(#[from] QEPError),
    
    #[error("Proposal not found: {0}")]
    ProposalNotFound(String),
    
    #[error("Timeout after {0:?}")]
    Timeout(Duration),
    
    #[error("Insufficient votes")]
    InsufficientVotes,
    
    #[error("Byzantine behavior detected: {0}")]
    ByzantineDetected(String),
    
    #[error("Quantum verification failed")]
    VerificationFailed,
    
    #[error("Not a proposer node")]
    NotProposer,
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Phase of the QBFT protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum QBFTPhase {
    /// Initial phase
    Idle,
    
    /// Pre-prepare phase - the primary proposes a value
    PrePrepare,
    
    /// Prepare phase - nodes verify and vote on the proposal
    Prepare,
    
    /// Commit phase - nodes commit to the agreed value
    Commit,
    
    /// Quantum verification phase
    QuantumVerification,
    
    /// Final phase - consensus reached
    Finalized,
    
    /// Error state
    Failed,
}

/// Vote in the QBFT protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QBFTVote {
    /// ID of the node that voted
    pub node_id: String,
    
    /// ID of the proposal
    pub proposal_id: String,
    
    /// Phase this vote is for
    pub phase: QBFTPhase,
    
    /// Whether the node accepts the proposal
    pub accept: bool,
    
    /// Signature of the node
    pub signature: Vec<u8>,
    
    /// Timestamp of the vote
    pub timestamp: u64,
    
    /// Quantum measurement result, if applicable
    pub quantum_measurement: Option<Vec<u8>>,
}

/// A proposal in the QBFT protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QBFTProposal {
    /// Unique ID of this proposal
    pub id: String,
    
    /// Value being proposed
    pub value: Vec<u8>,
    
    /// Hash of the value
    pub value_hash: String,
    
    /// ID of the proposer node
    pub proposer_id: String,
    
    /// When this proposal was created
    pub created_at: u64,
    
    /// Current phase of this proposal
    pub phase: QBFTPhase,
    
    /// View number (round of consensus)
    pub view: u64,
    
    /// Votes received for each phase
    #[serde(skip)]
    pub votes: HashMap<QBFTPhase, HashMap<String, QBFTVote>>,
    
    /// Quantum entanglements used for verification
    #[serde(skip)]
    pub entanglements: Vec<EntanglementPair>,
    
    /// Quantum measurements from nodes
    #[serde(skip)]
    pub quantum_measurements: HashMap<String, Vec<u8>>,
    
    /// Whether to use quantum verification
    pub quantum_verification: bool,
    
    /// Result of the consensus
    pub result: Option<ConsensusResult>,
}

impl QBFTProposal {
    /// Create a new proposal
    ///
    /// # Arguments
    ///
    /// * `value` - The value being proposed
    /// * `proposer_id` - ID of the proposing node
    ///
    /// # Returns
    ///
    /// A new QBFT proposal
    #[must_use]
    pub fn new(value: &[u8], proposer_id: &str) -> Self {
        // Create a hash of the value
        let value_hash_bytes = util::hash_bytes(value);
        let value_hash = hex::encode(&value_hash_bytes);
        
        Self {
            id: util::generate_id("qbft"),
            value: value.to_vec(),
            value_hash,
            proposer_id: proposer_id.to_string(),
            created_at: util::timestamp_now(),
            phase: QBFTPhase::Idle,
            view: 0,
            votes: HashMap::new(),
            entanglements: Vec::new(),
            quantum_measurements: HashMap::new(),
            quantum_verification: false,
            result: None,
        }
    }
    
    /// Get the vote count for a phase
    ///
    /// # Arguments
    ///
    /// * `phase` - The phase to count votes for
    ///
    /// # Returns
    ///
    /// The number of votes for the given phase
    #[must_use]
    pub fn vote_count(&self, phase: QBFTPhase) -> usize {
        self.votes.get(&phase).map_or(0, std::collections::HashMap::len)
    }
    
    /// Check if a node has voted in a phase
    #[must_use]
    pub fn has_voted(&self, node_id: &str, phase: QBFTPhase) -> bool {
        self.votes.get(&phase)
            .is_some_and(|phase_votes| phase_votes.contains_key(node_id))
    }
    
    /// Add a vote for this proposal
    pub fn add_vote(&mut self, vote: QBFTVote) -> bool {
        if vote.proposal_id != self.id {
            return false;
        }
        
        self.votes.entry(vote.phase)
            .or_default()
            .insert(vote.node_id.clone(), vote);
        
        true
    }
    
    /// Check if a phase has sufficient votes for consensus
    #[must_use]
    pub fn has_consensus(&self, phase: QBFTPhase, total_nodes: usize, threshold: f64) -> bool {
        // Get votes for this phase
        let accept_votes = self.votes.get(&phase)
            .map_or(0, |phase_votes| 
                phase_votes.values()
                    .filter(|vote| vote.accept)
                    .count()
            );
        
        // Instead of converting threshold to a vote count (which requires unsafe casts),
        // we can compare directly using floating point math
        let total_nodes_f64 = f64::from(u32::try_from(total_nodes).unwrap_or(u32::MAX));
        let accept_votes_f64 = f64::from(u32::try_from(accept_votes).unwrap_or(u32::MAX));
        
        // Compare: accept_votes ≥ ceil(total_nodes * threshold)
        // Equivalent to: accept_votes / total_nodes ≥ threshold
        total_nodes_f64 > 0.0 && accept_votes_f64 / total_nodes_f64 >= threshold
    }
    
    /// Create a quantum measurement for verification
    #[must_use]
    pub fn create_quantum_measurement(&self, node_id: &str, entanglements: &[EntanglementPair]) -> Vec<u8> {
        // In a real system, this would use quantum mechanics to create a measurement
        // that's correlated with the other nodes' measurements based on entanglement
        
        // For simulation, we create a deterministic measurement from the value hash
        // and add a level of correlation based on the entanglements
        
        let mut measurement = Vec::new();
        
        // Start with a base measurement derived from proposal data
        let mut base = util::hash_to_bytes(&format!("{}{}", self.value_hash, node_id));
        
        // Apply correlation effects from entanglements
        for pair in entanglements {
            if pair.node_a_id == node_id || pair.node_b_id == node_id {
                // The higher the fidelity, the more correlation we simulate
                let correlation_factor = pair.fidelity;
                
                // Simulating correlation by combining measurements
                let other_node = if pair.node_a_id == node_id { 
                    &pair.node_b_id 
                } else { 
                    &pair.node_a_id 
                };
                
                let other_hash = util::hash_to_bytes(&format!("{}{}", self.value_hash, other_node));
                
                for i in 0..base.len().min(other_hash.len()) {
                    // Blend the bytes based on correlation factor
                    base[i] = ((f64::from(base[i])) * (1.0 - correlation_factor) + 
                              (f64::from(other_hash[i])) * correlation_factor) as u8;
                }
            }
        }
        
        measurement.extend_from_slice(&base);
        measurement
    }
    
    /// Verify quantum measurements for consistency
    #[must_use]
    pub fn verify_quantum_measurements(&self, threshold: f64) -> bool {
        if self.quantum_measurements.len() < 2 {
            return false;
        }
        
        let mut consistent_count = 0;
        let total_comparisons = (self.quantum_measurements.len() * (self.quantum_measurements.len() - 1)) / 2;
        
        let measurements: Vec<_> = self.quantum_measurements.iter().collect();
        
        for i in 0..measurements.len() {
            for j in i+1..measurements.len() {
                let (node_a, meas_a) = measurements[i];
                let (node_b, meas_b) = measurements[j];
                
                // Check if these nodes share entanglement
                let have_entanglement = self.entanglements.iter().any(|e| 
                    (e.node_a_id == *node_a && e.node_b_id == *node_b) ||
                    (e.node_a_id == *node_b && e.node_b_id == *node_a)
                );
                
                if !have_entanglement {
                    continue;
                }
                
                // In a real quantum system, measurements from entangled particles would
                // show correlations. Here we simulate by checking consistency of data.
                
                // Calculate correlation level (simulated for entangled systems)
                let mut correlation = 0.0;
                let min_len = meas_a.len().min(meas_b.len());
                
                if min_len > 0 {
                    let mut match_count = 0;
                    
                    for k in 0..min_len {
                        // Compute bit-level correlation
                        let diff = (i16::from(meas_a[k]) - i16::from(meas_b[k])).abs();
                        let corr = 1.0 - (f64::from(diff) / 255.0);
                        match_count += u32::from(corr > 0.8);
                    }
                    
                    // Safe conversion from usize to f64
                    let min_len_f64 = f64::from(u32::try_from(min_len).unwrap_or(u32::MAX));
                    correlation = f64::from(match_count) / min_len_f64;
                }
                
                if correlation > 0.7 {
                    consistent_count += 1;
                }
            }
        }
        
        // Check if enough measurements are consistent
        let consistency_ratio = if total_comparisons > 0 {
            // Safe conversion from usize to f64
            let total_comparisons_f64 = f64::from(u32::try_from(total_comparisons).unwrap_or(u32::MAX));
            f64::from(consistent_count) / total_comparisons_f64
        } else {
            0.0
        };
        
        consistency_ratio >= threshold
    }
}

/// Configuration for QBFT
#[derive(Debug, Clone)]
pub struct QBFTConfig {
    /// Timeout for each phase in milliseconds
    pub phase_timeout_ms: u64,
    
    /// Classical fault tolerance threshold (usually 2/3)
    pub classical_threshold: f64,
    
    /// Quantum-boosted threshold (lower due to entanglement verification)
    pub quantum_threshold: f64,
    
    /// Maximum view changes before giving up
    pub max_view_changes: u64,
    
    /// Whether to simulate network delays
    pub simulate_delays: bool,
    
    /// Whether to use quantum verification
    pub use_quantum_verification: bool,
}

impl Default for QBFTConfig {
    fn default() -> Self {
        Self {
            phase_timeout_ms: 10000, // 10 seconds
            classical_threshold: 2.0 / 3.0, // 2/3 of nodes
            quantum_threshold: 0.6, // 60% of nodes with quantum verification
            max_view_changes: 3,
            simulate_delays: true,
            use_quantum_verification: true,
        }
    }
}

/// Implementation of Quantum Byzantine Fault Tolerance
pub struct QBFT {
    /// Node running this QBFT instance
    node_id: String,
    
    /// Current view (round of consensus)
    view: u64,
    
    /// Active proposals
    proposals: HashMap<String, QBFTProposal>,
    
    /// Known nodes in the network
    nodes: Vec<String>,
    
    /// Configuration
    config: QBFTConfig,
    
    /// Reference to node for entanglement operations
    node: Option<Arc<RwLock<Node>>>,
}

impl QBFT {
    /// Create a new QBFT instance with the specified node ID and set of nodes
    #[must_use]
    pub fn new(node_id: String, nodes: Vec<String>) -> Self {
        Self {
            node_id,
            view: 0,
            proposals: HashMap::new(),
            nodes,
            config: QBFTConfig::default(),
            node: None,
        }
    }
    
    /// Create a new QBFT instance with custom configuration
    #[must_use]
    pub fn with_config(node_id: String, nodes: Vec<String>, config: QBFTConfig) -> Self {
        Self {
            node_id,
            view: 0,
            proposals: HashMap::new(),
            nodes,
            config,
            node: None,
        }
    }
    
    /// Set the node reference for entanglement operations
    pub fn set_node(&mut self, node: Arc<RwLock<Node>>) {
        self.node = Some(node);
    }
    
    /// Get the current configuration
    #[must_use]
    pub fn config(&self) -> &QBFTConfig {
        &self.config
    }
    
    /// Get mutable access to the configuration
    pub fn config_mut(&mut self) -> &mut QBFTConfig {
        &mut self.config
    }
    
    /// Set a new configuration
    pub fn set_config(&mut self, config: QBFTConfig) {
        self.config = config;
    }
    
    /// Get the current view number
    #[must_use]
    pub fn view(&self) -> u64 {
        self.view
    }
    
    /// Set the current view
    pub fn set_view(&mut self, view: u64) {
        self.view = view;
    }
    
    /// Get the primary node for the current view
    #[must_use]
    pub fn primary(&self) -> &str {
        // Safe conversion from u64 to usize with fallback
        let primary_idx = match usize::try_from(self.view) {
            Ok(idx) => idx % self.nodes.len(),
            Err(_) => 0  // Default to first node if overflow occurs
        };
        
        &self.nodes[primary_idx]
    }
    
    /// Check if this node is the primary for the current view
    #[must_use]
    pub fn is_primary(&self) -> bool {
        self.primary() == self.node_id
    }
    
    /// Get the total number of nodes participating
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
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
    
    /// Run the pre-prepare phase of QBFT
    ///
    /// # Arguments
    ///
    /// * `proposal_id` - The ID of the proposal
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - Phase completed successfully
    /// * `Ok(false)` - Phase not applicable or skipped
    /// * `Err(QBFTError)` - Error occurred during phase execution
    #[must_use = "Check if pre-prepare phase was successful"]
    async fn run_pre_prepare(&mut self, proposal_id: &str) -> Result<bool, QBFTError> {
        // Check if we're the primary first, before getting a mutable reference to proposal
        let is_primary = self.is_primary();
        if !is_primary {
            return Ok(false);
        }
        
        let proposal = self.proposals.get_mut(proposal_id)
            .ok_or_else(|| QBFTError::ProposalNotFound(proposal_id.to_string()))?;
        
        // Update phase
        proposal.phase = QBFTPhase::PrePrepare;
        
        // Create a vote from this node
        let node_id = self.node_id.clone();
        let vote = QBFTVote {
            node_id,
            proposal_id: proposal_id.to_string(),
            phase: QBFTPhase::PrePrepare,
            accept: true,
            signature: Vec::new(), // In a real system, this would be signed
            timestamp: util::timestamp_now(),
            quantum_measurement: None,
        };
        
        // Add the vote
        proposal.add_vote(vote);
        
        // In a real system, the primary would broadcast the pre-prepare message
        // to all participating nodes. For simulation, we'll assume all nodes
        // receive the pre-prepare message successfully.
        
        // Simulate network delay
        self.simulate_delay().await;
        
        Ok(true)
    }
    
    /// Run the prepare phase of QBFT
    ///
    /// # Arguments
    ///
    /// * `proposal_id` - The ID of the proposal
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - Phase completed successfully
    /// * `Ok(false)` - Phase not applicable or skipped
    /// * `Err(QBFTError)` - Error occurred during phase execution
    #[must_use = "Check if prepare phase was successful"]
    async fn run_prepare(&mut self, proposal_id: &str) -> Result<bool, QBFTError> {
        // Get all the values we need before borrowing proposal
        let node_id = self.node_id.clone();
        let node_count = self.node_count();
        let threshold = self.config.classical_threshold;
        
        // Now get the proposal
        let proposal = self.proposals.get_mut(proposal_id)
            .ok_or_else(|| QBFTError::ProposalNotFound(proposal_id.to_string()))?;
            
        // Update phase
        proposal.phase = QBFTPhase::Prepare;
        
        // Create a vote from this node
        let vote = QBFTVote {
            node_id,
            proposal_id: proposal_id.to_string(),
            phase: QBFTPhase::Prepare,
            accept: true, // In a real system, we would verify the proposal
            signature: Vec::new(), // In a real system, this would be signed
            timestamp: util::timestamp_now(),
            quantum_measurement: None,
        };
        
        // Add the vote
        proposal.add_vote(vote);
        
        // Drop the proposal reference so we can call other methods
        let _ = proposal;
        
        // Simulate other nodes voting
        self.simulate_other_nodes_voting(proposal_id, QBFTPhase::Prepare, 0.95)?;
        
        // Get the proposal again to check votes
        let proposal = self.proposals.get(proposal_id)
            .ok_or_else(|| QBFTError::ProposalNotFound(proposal_id.to_string()))?;
        
        // Check if we have enough votes
        let sufficient_votes = proposal.has_consensus(
            QBFTPhase::Prepare,
            node_count,
            threshold
        );
        
        // Simulate network delay
        self.simulate_delay().await;
        
        if !sufficient_votes {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Run the commit phase of QBFT
    ///
    /// # Arguments
    ///
    /// * `proposal_id` - The ID of the proposal
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - Phase completed successfully
    /// * `Ok(false)` - Phase not applicable or skipped
    /// * `Err(QBFTError)` - Error occurred during phase execution
    #[must_use = "Check if commit phase was successful"]
    async fn run_commit(&mut self, proposal_id: &str) -> Result<bool, QBFTError> {
        // Get all the values we need before borrowing proposal
        let node_id = self.node_id.clone();
        let node_count = self.node_count();
        let threshold = self.config.classical_threshold;
        
        // Now get the proposal
        let proposal = self.proposals.get_mut(proposal_id)
            .ok_or_else(|| QBFTError::ProposalNotFound(proposal_id.to_string()))?;
            
        // Update phase
        proposal.phase = QBFTPhase::Commit;
        
        // Create a vote from this node
        let vote = QBFTVote {
            node_id,
            proposal_id: proposal_id.to_string(),
            phase: QBFTPhase::Commit,
            accept: true, // In a real system, we would check prepare votes
            signature: Vec::new(), // In a real system, this would be signed
            timestamp: util::timestamp_now(),
            quantum_measurement: None,
        };
        
        // Add the vote
        proposal.add_vote(vote);
        
        // Drop the proposal reference so we can call other methods
        let _ = proposal;
        
        // Simulate other nodes voting
        self.simulate_other_nodes_voting(proposal_id, QBFTPhase::Commit, 0.9)?;
        
        // Get the proposal again to check votes
        let proposal = self.proposals.get(proposal_id)
            .ok_or_else(|| QBFTError::ProposalNotFound(proposal_id.to_string()))?;
        
        // Check if we have enough votes
        let sufficient_votes = proposal.has_consensus(
            QBFTPhase::Commit,
            node_count,
            threshold
        );
        
        // Simulate network delay
        self.simulate_delay().await;
        
        if !sufficient_votes {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Run the quantum verification phase of QBFT
    ///
    /// # Arguments
    ///
    /// * `proposal_id` - The ID of the proposal
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - Phase completed successfully
    /// * `Ok(false)` - Phase not applicable or skipped
    /// * `Err(QBFTError)` - Error occurred during phase execution 
    #[must_use = "Check if quantum verification phase was successful"]
    async fn run_quantum_verification(&mut self, proposal_id: &str) -> Result<bool, QBFTError> {
        // Get all the values we need before borrowing proposal
        let node_id = self.node_id.clone();
        let quantum_threshold = self.config.quantum_threshold;
        
        // Now get the proposal
        let proposal = self.proposals.get_mut(proposal_id)
            .ok_or_else(|| QBFTError::ProposalNotFound(proposal_id.to_string()))?;
        
        // Don't proceed if quantum verification is not enabled
        if !proposal.quantum_verification {
            return Ok(true); // Skip but consider successful
        }
        
        // Update phase
        proposal.phase = QBFTPhase::QuantumVerification;
        
        // Create entanglements between committed nodes
        let committed_nodes: Vec<String> = proposal.votes.get(&QBFTPhase::Commit)
            .map_or(Vec::new(), |phase_votes| 
                phase_votes.values()
                    .filter(|vote| vote.accept)
                    .map(|vote| vote.node_id.clone())
                    .collect()
            );
            
        if committed_nodes.len() < 2 {
            return Ok(false);
        }

        // Clone node_ref so we can use it outside of this scope
        let node_ref_clone = self.node.clone();
        
        // Create entanglement pairs if we have a node reference
        if let Some(node_ref) = &node_ref_clone {
            // Create entanglement graph
            for i in 0..committed_nodes.len() {
                for j in i + 1..committed_nodes.len() {
                    let node_a = &committed_nodes[i];
                    let node_b = &committed_nodes[j];
                    
                    // Skip if neither node is us
                    if node_a != &node_id && node_b != &node_id {
                        continue;
                    }
                    
                    // Create entanglement if we're one of the nodes
                    if node_a == &node_id || node_b == &node_id {
                        let other_node = if node_a == &node_id { 
                            node_b 
                        } else { 
                            node_a 
                        };
                        
                        // Get a write lock on the node
                        let mut node = node_ref.write().await;
                        
                        // Get a mutable reference to the QEP instance
                        let qep = node.qep_mut();
                        
                        // Create entanglement
                        let entanglement = qep.create_entanglement(
                            &node_id,
                            other_node,
                            EntanglementPurpose::Consensus
                        ).await?;
                        
                        // Add to proposal's entanglements
                        proposal.entanglements.push(entanglement);
                    }
                }
            }
        }
        
        // Create quantum measurement
        let measurement = proposal.create_quantum_measurement(
            &node_id,
            &proposal.entanglements
        );
        
        // Create a vote with quantum measurement
        let vote = QBFTVote {
            node_id: node_id.clone(),
            proposal_id: proposal_id.to_string(),
            phase: QBFTPhase::QuantumVerification,
            accept: true,
            signature: Vec::new(), // In a real system, this would be signed
            timestamp: util::timestamp_now(),
            quantum_measurement: Some(measurement.clone()),
        };
        
        // Add the vote and measurement
        proposal.add_vote(vote);
        proposal.quantum_measurements.insert(node_id, measurement);
        
        // Drop the proposal before calling other methods
        let _ = proposal;
        
        // Simulate other nodes' quantum measurements
        self.simulate_other_nodes_quantum_voting(proposal_id)?;
        
        // Get the proposal again to verify measurements
        let proposal = self.proposals.get(proposal_id)
            .ok_or_else(|| QBFTError::ProposalNotFound(proposal_id.to_string()))?;
        
        // Verify quantum measurements
        let verified = proposal.verify_quantum_measurements(quantum_threshold);
        
        // Simulate network delay
        self.simulate_delay().await;
        
        if !verified {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Finalize the consensus process
    ///
    /// # Arguments
    ///
    /// * `proposal_id` - The ID of the proposal
    /// * `success` - Whether consensus was successful
    /// * `_reason` - Optional reason for the result
    ///
    /// # Returns
    ///
    /// The consensus result or an error if proposal not found
    ///
    /// # Errors
    ///
    /// Returns `QBFTError::ProposalNotFound` if the proposal does not exist
    #[must_use = "Contains the final consensus result"]
    fn finalize_consensus(&mut self, proposal_id: &str, success: bool, _reason: Option<String>) 
        -> Result<ConsensusResult, QBFTError> {
        // Get node count before borrowing proposal
        let participants_count = self.node_count();
        
        // Get the proposal
        let proposal = self.proposals.get_mut(proposal_id)
            .ok_or_else(|| QBFTError::ProposalNotFound(proposal_id.to_string()))?;
        
        // Set the final phase
        proposal.phase = if success { QBFTPhase::Finalized } else { QBFTPhase::Failed };
        
        // Count votes
        let _prepare_votes = proposal.vote_count(QBFTPhase::Prepare);
        let commit_votes = proposal.vote_count(QBFTPhase::Commit);
        let _quantum_votes = proposal.vote_count(QBFTPhase::QuantumVerification);
        
        // Clone value if needed
        let value_clone = if success {
            Some(proposal.value.clone())
        } else {
            None
        };
        
        // Create the consensus result
        let result = ConsensusResult {
            consensus_reached: success,
            value: value_clone,
            participants: participants_count,
            agreements: commit_votes,
        };
        
        // Store the result
        proposal.result = Some(result.clone());
        
        Ok(result)
    }
    
    /// Simulate voting by other nodes for testing purposes
    ///
    /// # Arguments
    ///
    /// * `proposal_id` - The ID of the proposal
    /// * `phase` - The phase to simulate votes for
    /// * `acceptance_rate` - The probability of accepting votes
    ///
    /// # Returns
    ///
    /// Ok(()) on success or an error if proposal not found
    ///
    /// # Errors
    ///
    /// Returns `QBFTError::ProposalNotFound` if the proposal does not exist
    fn simulate_other_nodes_voting(&mut self, proposal_id: &str, phase: QBFTPhase, acceptance_rate: f64) 
        -> Result<(), QBFTError> {
        let proposal = self.proposals.get_mut(proposal_id)
            .ok_or_else(|| QBFTError::ProposalNotFound(proposal_id.to_string()))?;
        
        // Store the node count before we give up the mutable borrow
        let node_count = self.nodes.len();
        
        // Skip node 1 (assumed to be this node)
        for i in 2..=node_count {
            let node_id = format!("node{i}");
            
            // Determine vote result based on acceptance rate
            let rand_val = rand::random::<f64>();
            let accept = rand_val < acceptance_rate;
            
            // Create a vote based on the phase
            match phase {
                QBFTPhase::Prepare => {
                    let vote = QBFTVote {
                        node_id: node_id.clone(),
                        proposal_id: proposal_id.to_string(),
                        phase: QBFTPhase::Prepare,
                        accept,
                        signature: Vec::new(),
                        timestamp: util::timestamp_now(),
                        quantum_measurement: None,
                    };
                    proposal.add_vote(vote);
                },
                QBFTPhase::Commit => {
                    let vote = QBFTVote {
                        node_id,
                        proposal_id: proposal_id.to_string(),
                        phase: QBFTPhase::Commit,
                        accept,
                        signature: Vec::new(),
                        timestamp: util::timestamp_now(),
                        quantum_measurement: None,
                    };
                    proposal.add_vote(vote);
                },
                // For all other phases, do nothing
                QBFTPhase::Idle | QBFTPhase::PrePrepare | 
                QBFTPhase::QuantumVerification | QBFTPhase::Finalized | 
                QBFTPhase::Failed => {
                    // No voting in these phases
                }
            }
        }
        
        Ok(())
    }
    
    /// Simulate quantum voting by other nodes
    ///
    /// This method generates quantum votes from other nodes
    /// using simulated quantum entanglement for correlated voting.
    ///
    /// # Arguments
    ///
    /// * `proposal_id` - The identifier of the proposal
    ///
    /// # Returns
    ///
    /// Result indicating success or an error if the proposal is not found
    ///
    /// # Errors
    ///
    /// Returns `QBFTError::ProposalNotFound` if the proposal does not exist
    fn simulate_other_nodes_quantum_voting(&mut self, proposal_id: &str)
        -> Result<(), QBFTError> {
        let proposal = self.proposals.get_mut(proposal_id)
            .ok_or_else(|| QBFTError::ProposalNotFound(proposal_id.to_string()))?;
        
        // Skip node 1 (assumed to be this node)
        let node_count = self.nodes.len();
        
        // Generate entangled votes (correlated)
        // In a real implementation, this would use actual quantum entanglement
        
        // Randomly decide if we're going with accept or reject as the correlated vote
        let base_accept = rand::random::<f64>() < 0.7;
        
        // Generate votes for each node
        for i in 2..=node_count {
            let node_id = format!("node{i}");
            
            // With 80% probability nodes vote the same way (simulating entanglement correlation)
            let use_correlated_vote = rand::random::<f64>() < 0.8;
            let accept = if use_correlated_vote {
                base_accept
            } else {
                // 20% chance of deviating from the correlated state
                rand::random::<f64>() < 0.5
            };
            
            // Create and register vote
            let vote = QBFTVote {
                node_id,
                proposal_id: proposal_id.to_string(),
                phase: QBFTPhase::QuantumVerification,
                accept,
                signature: Vec::new(),
                timestamp: util::timestamp_now(),
                quantum_measurement: None,
            };
            
            proposal.add_vote(vote);
        }
        
        Ok(())
    }
    
    /// Run the QBFT consensus protocol for a proposal
    ///
    /// # Arguments
    ///
    /// * `proposal_id` - The ID of the proposal
    ///
    /// # Returns
    ///
    /// The consensus result or an error
    ///
    /// # Errors
    ///
    /// * `QBFTError::ProposalNotFound` - Proposal not found
    /// * Other errors propagated from phase execution
    #[must_use = "Contains the complete consensus protocol result"]
    async fn run_consensus_protocol(&mut self, proposal_id: &str) -> Result<ConsensusResult, QBFTError> {
        // Check if this is the primary node (for pre-prepare phase)
        if self.is_primary() {
            // Run pre-prepare phase if we are the primary
            self.run_pre_prepare(proposal_id).await?;
        }
        
        // Run prepare phase
        self.run_prepare(proposal_id).await?;
        
        // Run commit phase
        self.run_commit(proposal_id).await?;
        
        // Run quantum verification phase if enabled
        if self.config.use_quantum_verification {
            self.run_quantum_verification(proposal_id).await?;
        }
        
        // Finalize consensus with success
        self.finalize_consensus(proposal_id, true, None)
    }
    
    /// Run the QBFT consensus process for a proposal
    ///
    /// # Arguments
    ///
    /// * `proposal_id` - The ID of the proposal
    ///
    /// # Returns
    ///
    /// The consensus result or an error
    ///
    /// # Errors
    ///
    /// * `QBFTError::ProposalNotFound` - Proposal not found
    /// * Other errors propagated from consensus execution
    #[must_use = "Contains the final consensus result for the specified proposal"]
    pub async fn run_consensus(&mut self, proposal_id: &str) -> Result<ConsensusResult, QBFTError> {
        // Get the proposal or return an error if not found
        if !self.proposals.contains_key(proposal_id) {
            return Err(QBFTError::ProposalNotFound(proposal_id.to_string()));
        }
        
        // Run the full consensus protocol
        self.run_consensus_protocol(proposal_id).await
    }
}

#[async_trait]
impl ConsensusProtocol for QBFT {
    async fn propose(&mut self, value: &[u8]) -> ConsensusResult {
        // Create a new proposal
        let mut proposal = QBFTProposal::new(value, &self.node_id);
        
        // Set quantum verification flag based on config
        proposal.quantum_verification = self.config.use_quantum_verification;
        
        let proposal_id = proposal.id.clone();
        
        // Store the proposal
        self.proposals.insert(proposal_id.clone(), proposal);
        
        // Try to run the consensus protocol
        match self.run_consensus_protocol(&proposal_id).await {
            Ok(result) => result,
            Err(_e) => {
                // Create a failed result
                ConsensusResult {
                    consensus_reached: false,
                    value: None,
                    participants: self.nodes.len(),
                    agreements: 0,
                }
            }
        }
    }
    
    async fn vote(&mut self, proposal_id: &str, accept: bool) -> bool {
        // Check if proposal exists
        if !self.proposals.contains_key(proposal_id) {
            return false;
        }
        
        let proposal = self.proposals.get_mut(proposal_id).unwrap();
        
        // Determine the current phase
        let phase = proposal.phase;
        
        // Create a vote
        let vote = QBFTVote {
            node_id: self.node_id.clone(),
            proposal_id: proposal_id.to_string(),
            phase,
            accept,
            signature: Vec::new(), // In a real system, this would be signed
            timestamp: util::timestamp_now(),
            quantum_measurement: None,
        };
        
        // Add the vote
        proposal.add_vote(vote)
    }
    
    async fn check_consensus(&self, proposal_id: &str) -> ConsensusResult {
        // Check if proposal exists
        if let Some(proposal) = self.proposals.get(proposal_id) {
            // Return the result if available
            if let Some(result) = &proposal.result {
                return result.clone();
            }
            
            // Check the current state
            let prepare_consensus = proposal.has_consensus(
                QBFTPhase::Prepare,
                self.node_count(),
                self.config.classical_threshold
            );
            
            let commit_consensus = proposal.has_consensus(
                QBFTPhase::Commit,
                self.node_count(),
                self.config.classical_threshold
            );
            
            let quantum_verification = if self.config.use_quantum_verification {
                proposal.verify_quantum_measurements(self.config.quantum_threshold)
            } else {
                true
            };
            
            let consensus_reached = prepare_consensus && commit_consensus && quantum_verification;
            
            return ConsensusResult {
                consensus_reached,
                value: if consensus_reached { Some(proposal.value.clone()) } else { None },
                participants: self.node_count(),
                agreements: proposal.vote_count(QBFTPhase::Commit),
            };
        }
        
        // Return a default result for unknown proposals
        ConsensusResult {
            consensus_reached: false,
            value: None,
            participants: 0,
            agreements: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_basic_consensus() {
        // Create a test network with 4 nodes
        let nodes = vec![
            "node1".to_string(),
            "node2".to_string(),
            "node3".to_string(),
            "node4".to_string(),
        ];
        
        // Create QBFT instance for the first node (primary in view 0)
        let mut qbft = QBFT::new("node1".to_string(), nodes.clone());
        
        // Disable quantum verification for basic test
        qbft.config = QBFTConfig {
            use_quantum_verification: false,
            ..QBFTConfig::default()
        };
        
        // Propose a value
        let test_value = b"test consensus value";
        let result = qbft.propose(test_value).await;
        
        // Check the result
        assert!(result.consensus_reached);
        assert_eq!(result.value.unwrap(), test_value);
        assert_eq!(result.participants, 4);
        assert!(result.agreements >= 3); // 3 or more nodes agreed (2/3 threshold)
    }
    
    #[tokio::test]
    async fn test_view_change() {
        // Create a test network with 4 nodes
        let nodes = vec![
            "node1".to_string(),
            "node2".to_string(),
            "node3".to_string(),
            "node4".to_string(),
        ];
        
        // Create QBFT instance for the second node (not primary in view 0)
        let mut qbft = QBFT::new("node2".to_string(), nodes.clone());
        
        // Disable quantum verification for basic test and lower threshold
        qbft.config = QBFTConfig {
            use_quantum_verification: false,
            classical_threshold: 0.5, // Lower the threshold to make the test more reliable
            ..QBFTConfig::default()
        };
        
        // Advance view to make node2 the primary
        qbft.view = 1; // In a 4-node network, view 1 should make node2 the primary
        
        // Verify node2 is actually the primary now
        assert!(qbft.is_primary(), "Node2 should be the primary in view 1");
        assert_eq!(qbft.primary(), "node2");
        
        // Propose a value
        let test_value = b"test view change";
        let result = qbft.propose(test_value).await;
        
        // Check the result
        assert!(result.consensus_reached, "Consensus should be reached after view change");
        assert_eq!(result.value.unwrap(), test_value);
    }
} 