// Network Module
//
// This module contains components for quantum network communication,
// including topology management, routing, and protocol implementations.

pub mod node;
pub mod topology;
pub mod routing;
pub mod entanglement;
pub mod qesp;
pub mod qrep;
pub mod qsyp;
pub mod qcap;
pub mod adapters;

// Re-exports
pub use node::Node;
pub use topology::NetworkTopology; 

// Export protocols
pub use entanglement::QEP;
pub use qesp::QESP;
pub use qrep::QREP;
pub use qsyp::QSYP;
pub use qcap::QCAP;
pub use adapters::QESPNetworkAdapter; 