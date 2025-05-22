// Error Correction Module
//
// This module provides quantum error correction protocols for quantum states.

// Export sub-modules
pub mod qecc;
pub mod qdcp;
pub mod qftp;
pub mod qcqp;
pub mod bit_flip_code;
pub mod phase_flip_code;
pub mod steane_code;

// Re-export main types
pub use qecc::QECC;
pub use qecc::QECCConfig;
pub use qecc::ErrorSyndrome;
pub use qecc::QECCError;

pub use qdcp::QDCP;
pub use qdcp::QDCPConfig;
pub use qdcp::DecouplingSequence;
pub use qdcp::RefreshStrategy;
pub use qdcp::QDCPError;

pub use qftp::QFTP;
pub use qftp::QFTPConfig;
pub use qftp::FailureDetection;
pub use qftp::RecoveryStrategy;
pub use qftp::NodeStatus;
pub use qftp::PathStatus;
pub use qftp::QFTPError;
pub use qftp::SystemHealth;

pub use qcqp::QCQP;
pub use qcqp::QCQPConfig;
pub use qcqp::VerificationMethod;
pub use qcqp::Domain;
pub use qcqp::ProtectedData;
pub use qcqp::VerificationResult;
pub use qcqp::QCQPError;

pub use bit_flip_code::BitFlipCode;
pub use phase_flip_code::PhaseFlipCode;
pub use steane_code::SteaneCode;

/// Type of error correction code
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectionCode {
    /// No error correction
    None,
    
    /// Basic repetition code
    Repetition,
    
    /// Shor's 9-qubit code
    Shor9,
    
    /// Steane's 7-qubit code
    Steane7,
    
    /// Surface code
    Surface,
    
    /// Three-qubit bit flip code
    BitFlip3,
    
    /// Three-qubit phase flip code
    PhaseFlip3,
}

/// Trait for implementing quantum error correction
pub trait ErrorCorrection {
    /// Encode a quantum state with error correction
    fn encode(&mut self, code: CorrectionCode) -> bool;
    
    /// Decode and correct a quantum state
    fn decode_and_correct(&mut self) -> bool;
    
    /// Detect errors in a quantum state
    fn detect_errors(&self) -> Vec<usize>;
    
    /// Get the current error correction code
    fn current_code(&self) -> CorrectionCode;
}

/// Error correction code type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCorrectionType {
    /// Three-qubit bit flip code
    BitFlip,
    
    /// Three-qubit phase flip code
    PhaseFlip,
    
    /// Shor's nine-qubit code
    Shor,
    
    /// Steane seven-qubit code
    Steane,
    
    /// Five-qubit perfect code
    FiveQubit,
} 