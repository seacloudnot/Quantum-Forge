// Quantum Protocols Rust Implementation
//
// This library provides Rust implementations of quantum protocols for blockchain systems,
// with a focus on simulation for development until actual quantum hardware becomes available.

pub mod core;
pub mod network;
pub mod consensus;
pub mod error_correction;
pub mod security;
pub mod integration;
pub mod simulation;
pub mod util;
pub mod error;
pub mod test;
pub mod integration_tests;
pub mod benchmark;

// Re-export all the public modules
pub use crate::core::*;
pub use crate::network::*;
pub use crate::consensus::*;
pub use crate::error_correction::*;
pub use crate::security::*;
pub use crate::integration::*;
pub use crate::error::Error;
pub use crate::error::Result;

// Re-export important traits directly
pub use crate::security::QuantumKeyDistribution;
pub use crate::security::PostQuantumCrypto;
pub use crate::network::entanglement::EntanglementProtocol;
pub use crate::network::routing::QuantumRouter;
pub use crate::consensus::QBFTConfig;

/// The prelude module provides a convenient import for all commonly used types and traits.
///
/// Import this module with `use quantum_protocols::prelude::*;` to get access to the most
/// frequently used components without having to import them individually.
///
/// # Example
///
/// ```
/// use quantum_protocols::prelude::*;
///
/// // Now you can use types like QuantumState, QEP, QKD etc. directly
/// ```
/// Main prelude that exports commonly used types
pub mod prelude {
    // Core
    pub use crate::core::{QuantumState, Qubit, QuantumRegister, QSTP, QMP, QMEM};
    
    // Network
    pub use crate::network::{Node, NetworkTopology, QEP, QESP, QREP, QSYP};
    pub use crate::network::routing::QNRPRouter;
    pub use crate::network::adapters::QESPNetworkAdapter;
    
    // Consensus
    pub use crate::consensus::{ConsensusProtocol, ConsensusResult};
    pub use crate::consensus::qbft::{QBFT, QBFTConfig};
    pub use crate::consensus::qpcp::{QPCP, QPCPConfig, ProbabilityModel};
    pub use crate::consensus::qvcp::QVCP;
    pub use crate::consensus::qtsp::QTSP;
    
    // Security
    pub use crate::security::qkd::{QKD, QKDConfig, QKDBasis};
    pub use crate::security::qrng::QRNG;
    pub use crate::security::qspp::{QSPP, QSPPConfig, SideChannelAttackType, CountermeasureTechnique};
    pub use crate::security::pqc::{PQC, PQCConfig};
    pub use crate::security::QuantumResistantAlgorithm;
    
    // Error correction
    pub use crate::error_correction::{QECC, CorrectionCode};
    pub use crate::error_correction::qdcp::{QDCP, QDCPConfig, RefreshStrategy};
    pub use crate::error_correction::qftp::QFTP;
    pub use crate::error_correction::qcqp::{QCQP, QCQPConfig, VerificationMethod, Domain};
    
    // Integration
    pub use crate::integration::qcbp::{QCBP, QCBPConfig, DataFormat, ConversionMode, SerializationMethod};
    pub use crate::integration::qdap::{QDAP, QDAPConfig, EncodingScheme, CompressionLevel};
    pub use crate::integration::qhep::{QHEP, QHEPConfig, HardwareArchitecture, ResourceStrategy};
    
    // Capability announcement
    pub use crate::network::qcap::{QCAP, QuantumCapability, CapabilityLevel};
    
    // Utilities
    pub use crate::util::{calculate_fidelity, apply_decoherence, timestamp_now, format_duration};
    pub use crate::util::{BellMeasurement, InstantWrapper};
    pub use crate::util::{generate_id, generate_timestamped_id};
    
    // Testing utilities
    #[cfg(test)]
    pub use crate::test::{
        create_test_network, create_ring_network, create_fully_connected_network,
        create_test_node, create_wrapped_test_node, create_test_entanglement,
        create_test_quantum_state, create_bell_pair, TestQNetwork
    };
    
    // Common types
    pub use crate::Result;
    pub use crate::Error;
} 