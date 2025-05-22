// Error handling for Quantum Protocols
//
// This module provides a centralized error handling system for quantum protocols.

use std::time::Duration;
use thiserror::Error;

/// Central error type for all quantum protocol operations
#[derive(Error, Debug)]
pub enum Error {
    /// Errors from quantum state operations
    #[error("Quantum state error: {0}")]
    QuantumState(#[from] QuantumStateError),
    
    /// Errors from network operations
    #[error("Network error: {0}")]
    Network(#[from] NetworkError),
    
    /// Errors from consensus operations
    #[error("Consensus error: {0}")]
    Consensus(#[from] ConsensusError),
    
    /// Errors from security operations
    #[error("Security error: {0}")]
    Security(#[from] SecurityError),
    
    /// Errors from error correction operations
    #[error("Error correction error: {0}")]
    ErrorCorrection(String),
    
    /// Errors from simulation
    #[error("Simulation error: {0}")]
    Simulation(String),
    
    /// General errors
    #[error("General error: {0}")]
    General(String),
    
    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    /// JSON serialization/deserialization errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    
    /// Custom errors
    #[error("Custom error: {0}")]
    Custom(String),
}

/// Result type for quantum protocol operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors related to quantum state operations
#[derive(Error, Debug)]
pub enum QuantumStateError {
    /// Invalid qubit index
    #[error("Invalid qubit index: {0}")]
    InvalidQubitIndex(usize),
    
    /// State has decohered
    #[error("Quantum state has decohered")]
    Decoherence,
    
    /// Invalid operation
    #[error("Invalid quantum operation: {0}")]
    InvalidOperation(String),
    
    /// State has been measured
    #[error("Cannot modify measured state")]
    AlreadyMeasured,
    
    /// Error during teleportation
    #[error("Teleportation error: {0}")]
    TeleportationError(String),
    
    /// Quantum resource exhausted
    #[error("Quantum resource exhausted: {0}")]
    ResourceExhausted(String),
    
    /// Quantum state not found
    #[error("Quantum state not found: {0}")]
    StateNotFound(String),
}

/// Errors related to network operations
#[derive(Error, Debug)]
pub enum NetworkError {
    /// Node not found
    #[error("Node not found: {0}")]
    NodeNotFound(String),
    
    /// Path not found
    #[error("No path found from {0} to {1}")]
    NoPath(String, String),
    
    /// Entanglement error
    #[error("Entanglement error: {0}")]
    EntanglementError(String),
    
    /// Routing error
    #[error("Routing error: {0}")]
    RoutingError(String),
    
    /// Timeout
    #[error("Network operation timed out after {0:?}")]
    Timeout(Duration),
}

/// Errors related to consensus operations
#[derive(Error, Debug)]
pub enum ConsensusError {
    /// Invalid proposal
    #[error("Invalid proposal: {0}")]
    InvalidProposal(String),
    
    /// Not enough votes
    #[error("Not enough votes to reach consensus")]
    InsufficientVotes,
    
    /// Byzantine error detected
    #[error("Byzantine error detected: {0}")]
    ByzantineError(String),
    
    /// View change error
    #[error("View change error: {0}")]
    ViewChangeError(String),
}

/// Errors related to security operations
#[derive(Error, Debug)]
pub enum SecurityError {
    /// Authentication error
    #[error("Authentication error: {0}")]
    AuthenticationError(String),
    
    /// Key exchange error
    #[error("Key exchange error: {0}")]
    KeyExchangeError(String),
    
    /// Eavesdropper detected
    #[error("Eavesdropper detected: QBER = {0:.4}")]
    EavesdropperDetected(f64),
    
    /// Cryptographic error
    #[error("Cryptographic error: {0}")]
    CryptoError(String),
}

// Conversions from String to various errors for ergonomic error creation

impl From<String> for Error {
    fn from(msg: String) -> Self {
        Self::General(msg)
    }
}

impl From<&str> for Error {
    fn from(msg: &str) -> Self {
        Self::General(msg.to_string())
    }
}

impl From<String> for QuantumStateError {
    fn from(msg: String) -> Self {
        Self::InvalidOperation(msg)
    }
}

impl From<String> for NetworkError {
    fn from(msg: String) -> Self {
        Self::RoutingError(msg)
    }
}

impl From<String> for ConsensusError {
    fn from(msg: String) -> Self {
        Self::InvalidProposal(msg)
    }
}

impl From<String> for SecurityError {
    fn from(msg: String) -> Self {
        Self::KeyExchangeError(msg)
    }
}

// Import the QEPError from entanglement module
use crate::network::entanglement::QEPError;

// Implement From<QEPError> for Error
impl From<QEPError> for Error {
    fn from(err: QEPError) -> Self {
        Self::Network(NetworkError::EntanglementError(err.to_string()))
    }
}

// Import the QSTPError
use crate::core::qstp::QSTPError;

// Implement From<QSTPError> for Error
impl From<QSTPError> for Error {
    fn from(err: QSTPError) -> Self {
        match err {
            QSTPError::Timeout(duration) => 
                Self::Network(NetworkError::Timeout(duration)),
            QSTPError::NetworkError(msg) => 
                Self::Network(NetworkError::RoutingError(msg)),
            QSTPError::NodeNotFound(node) => 
                Self::Network(NetworkError::NodeNotFound(node)),
            QSTPError::Decoherence => 
                Self::QuantumState(QuantumStateError::Decoherence),
            _ => Self::General(err.to_string())
        }
    }
}

// Import the QTSPError
use crate::consensus::qtsp::QTSPError;

// Implement From<QTSPError> for Error
impl From<QTSPError> for Error {
    fn from(err: QTSPError) -> Self {
        match err {
            QTSPError::Timeout(duration) => 
                Self::Network(NetworkError::Timeout(duration)),
            QTSPError::NetworkError(msg) => 
                Self::Network(NetworkError::RoutingError(msg)),
            QTSPError::KeySharingError(msg) => 
                Self::Security(SecurityError::KeyExchangeError(msg)),
            QTSPError::VerificationError(msg) => 
                Self::Security(SecurityError::CryptoError(msg)),
            _ => Self::General(err.to_string())
        }
    }
}

// Import the QDCPError
use crate::error_correction::qdcp::QDCPError;

// Implement From<QDCPError> for Error
impl From<QDCPError> for Error {
    fn from(err: QDCPError) -> Self {
        match err {
            QDCPError::Timeout(duration) => 
                Self::Network(NetworkError::Timeout(duration)),
            QDCPError::LowFidelity(_fidelity) => 
                Self::QuantumState(QuantumStateError::Decoherence),
            QDCPError::RefreshFailed(msg) => 
                Self::ErrorCorrection(format!("Refresh failed: {msg}")),
            QDCPError::IsolationError(msg) => 
                Self::ErrorCorrection(format!("Isolation error: {msg}")),
            _ => Self::ErrorCorrection(err.to_string())
        }
    }
}

// Import the QCQPError
use crate::error_correction::qcqp::QCQPError;

// Implement From for QCQPError
impl From<QCQPError> for Error {
    fn from(err: QCQPError) -> Self {
        Self::Custom(err.to_string())
    }
}

// Import the QDAPError
use crate::integration::qdap::QDAPError;

// Implement From for QDAPError
impl From<QDAPError> for Error {
    fn from(err: QDAPError) -> Self {
        Self::Custom(err.to_string())
    }
}

// Import the QHEPError
use crate::integration::qhep::QHEPError;

// Implement From for QHEPError
impl From<QHEPError> for Error {
    fn from(err: QHEPError) -> Self {
        Self::Custom(err.to_string())
    }
}

// Import the QCBPError
use crate::integration::qcbp::QCBPError;

// Implement From for QCBPError
impl From<QCBPError> for Error {
    fn from(err: QCBPError) -> Self {
        Self::Custom(err.to_string())
    }
}

// Import the HardwareInterfaceError
use crate::integration::qhep_hardware::HardwareInterfaceError;

// Implement From<HardwareInterfaceError> for Error
impl From<HardwareInterfaceError> for Error {
    fn from(err: HardwareInterfaceError) -> Self {
        match err {
            HardwareInterfaceError::ConnectionFailed(msg) => 
                Self::General(format!("Hardware connection failed: {msg}")),
            HardwareInterfaceError::NotInitialized(msg) => 
                Self::General(format!("Hardware not initialized: {msg}")),
            HardwareInterfaceError::CalibrationError(msg) => 
                Self::General(format!("Hardware calibration error: {msg}")),
            HardwareInterfaceError::ExecutionError(msg) => 
                Self::General(format!("Hardware execution error: {msg}")),
            HardwareInterfaceError::AuthenticationFailed(msg) => 
                Self::Security(SecurityError::AuthenticationError(msg)),
            HardwareInterfaceError::UnsupportedOperation(msg) => 
                Self::General(format!("Unsupported hardware operation: {msg}")),
            HardwareInterfaceError::HardwareBusy(msg) => 
                Self::General(format!("Hardware busy: {msg}")),
            HardwareInterfaceError::CommunicationError(msg) => 
                Self::Network(NetworkError::RoutingError(msg)),
            HardwareInterfaceError::InvalidConfiguration(msg) => 
                Self::General(format!("Invalid hardware configuration: {msg}")),
        }
    }
}

// Import the QSPPError
use crate::security::qspp::QSPPError;

// Implement From for QSPPError
impl From<QSPPError> for Error {
    fn from(err: QSPPError) -> Self {
        match err {
            QSPPError::TimingAttack(msg) => 
                Self::Security(SecurityError::CryptoError(format!("Timing attack: {msg}"))),
            QSPPError::PowerAnalysis(msg) => 
                Self::Security(SecurityError::CryptoError(format!("Power analysis: {msg}"))),
            QSPPError::PhotonEmission(msg) => 
                Self::Security(SecurityError::CryptoError(format!("Side-channel: {msg}"))),
            QSPPError::ConfigError(msg) => 
                Self::General(format!("QSPP configuration error: {msg}")),
            _ => Self::Security(SecurityError::CryptoError(err.to_string())),
        }
    }
} 