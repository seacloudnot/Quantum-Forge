// Integration Module
//
// This module provides integration between quantum and classical components.

use async_trait::async_trait;

/// Interface for integrating with blockchain systems
#[async_trait]
pub trait BlockchainIntegration {
    /// Initialize connection to blockchain
    async fn connect(&mut self, endpoint: &str) -> Result<(), String>;
    
    /// Submit a transaction to the blockchain
    async fn submit_transaction(&self, transaction: &[u8]) -> Result<String, String>;
    
    /// Query the blockchain state
    async fn query_state(&self, query: &str) -> Result<Vec<u8>, String>;
    
    /// Subscribe to blockchain events
    async fn subscribe_events(&mut self, event_type: &str) -> Result<(), String>;
    
    /// Close connection
    async fn disconnect(&mut self) -> Result<(), String>;
}

/// Interface for smart contract integration
#[async_trait]
pub trait SmartContractIntegration {
    /// Deploy a smart contract
    async fn deploy_contract(&self, code: &[u8], args: &[u8]) -> Result<String, String>;
    
    /// Call a contract method
    async fn call_contract(&self, contract_id: &str, method: &str, args: &[u8]) -> Result<Vec<u8>, String>;
    
    /// Query contract state
    async fn query_contract(&self, contract_id: &str, query: &str) -> Result<Vec<u8>, String>;
    
    /// Update contract state
    async fn update_contract(&self, contract_id: &str, update: &[u8]) -> Result<(), String>;
}

// Export sub-modules
pub mod qcbp;
pub mod qdap;
pub mod qhep;
pub mod qhep_hardware;
pub mod protocol_bridge;

// Re-export main types
pub use qcbp::QCBP;
pub use qcbp::QCBPConfig;
pub use qcbp::DataFormat;
pub use qcbp::ConversionMode;
pub use qcbp::SerializationMethod;
pub use qcbp::QCBPError;
pub use qcbp::BridgeMetadata;

pub use qdap::QDAP;
pub use qdap::QDAPConfig;
pub use qdap::EncodingScheme;
pub use qdap::CompressionLevel;
pub use qdap::AdaptabilityStrategy;
pub use qdap::EncodingMetadata;
pub use qdap::QDAPError;

pub use qhep::QHEP;
pub use qhep::QHEPConfig;
pub use qhep::HardwareArchitecture;
pub use qhep::InstructionSetType;
pub use qhep::ResourceStrategy;
pub use qhep::HardwareCapability;
pub use qhep::QHEPError;

pub use qhep_hardware::HardwareRegistry;
pub use qhep_hardware::HardwareExecutor;
pub use qhep_hardware::HardwareConnectionConfig;
pub use qhep_hardware::HardwareStatus;
pub use qhep_hardware::CalibrationResult;
pub use qhep_hardware::HardwareExecutorFactory;
pub use qhep_hardware::AuthMethod;

pub use protocol_bridge::QuantumProtocolBridge; 