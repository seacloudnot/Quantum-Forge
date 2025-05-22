// Consensus Module
//
// This module provides consensus protocols for quantum blockchain systems.

use async_trait::async_trait;
use serde::{Serialize, Deserialize};

/// Result of a consensus process
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsensusResult {
    /// Whether consensus was reached
    pub consensus_reached: bool,
    
    /// The value agreed upon (if any)
    pub value: Option<Vec<u8>>,
    
    /// Number of participants involved in the consensus
    pub participants: usize,
    
    /// Number of nodes that agreed with the result
    pub agreements: usize,
}

/// Trait for implementing consensus protocols
#[async_trait]
pub trait ConsensusProtocol {
    /// Propose a value to the consensus network
    async fn propose(&mut self, value: &[u8]) -> ConsensusResult;
    
    /// Vote on a proposed value
    async fn vote(&mut self, proposal_id: &str, accept: bool) -> bool;
    
    /// Check if consensus has been reached on a proposal
    async fn check_consensus(&self, proposal_id: &str) -> ConsensusResult;
}

pub mod qbft;
pub mod qvcp;
pub mod qpcp;
pub mod qtsp;

pub use qbft::QBFT;
pub use qbft::QBFTConfig;
pub use qvcp::QVCP;
pub use qvcp::QVCPConfig;
pub use qpcp::QPCP;
pub use qpcp::QPCPConfig;
pub use qtsp::QTSP;
pub use qtsp::QTSPConfig; 