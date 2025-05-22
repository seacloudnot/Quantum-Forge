// Core Quantum Protocol Types and Traits
//
// This module provides the fundamental quantum types and traits used
// by the various protocols in the system.

pub mod quantum_state;
pub mod qubit;
pub mod register;
pub mod qstp;  // Quantum State Transfer Protocol
pub mod qmp;   // Quantum Measurement Protocol
pub mod qmem;  // Quantum Memory Protocol
pub mod quantum_algorithm; // New module for quantum algorithms

pub use quantum_state::QuantumState;
pub use qubit::Qubit;
pub use register::QuantumRegister;
pub use qstp::QSTP;
pub use qmp::QMP;
pub use qmem::QMEM;
pub use quantum_algorithm::{GroverSearch, QuantumFourierTransform, QuantumPhaseEstimation, QuantumAlgorithm}; 