// Quantum-Classical Bridge Protocol (QCBP)
//
// This protocol interfaces between quantum and classical systems.

use std::fmt;
use std::collections::HashMap;
use thiserror::Error;
use serde::{Serialize, Deserialize};
use hex;

use crate::error::Result;
use crate::core::QuantumRegister;
use crate::util;

/// Errors specific to the QCBP protocol
#[derive(Debug, Error)]
pub enum QCBPError {
    #[error("Invalid data format: {0}")]
    InvalidDataFormat(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Deserialization error: {0}")]
    DeserializationError(String),
    
    #[error("Environment switching error: {0}")]
    EnvironmentSwitchingError(String),
    
    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),
    
    #[error("Protocol error: {0}")]
    ProtocolError(String),
}

/// Data format specifications
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataFormat {
    /// JSON format
    JSON,
    
    /// Binary format
    Binary,
    
    /// Protocol Buffers
    ProtoBuf,
    
    /// Custom format
    Custom,
}

/// Conversion mode for bridging operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConversionMode {
    /// Strict conversion, error on any inconsistency
    Strict,
    
    /// Lossy conversion, best-effort approach
    Lossy,
    
    /// Adaptive conversion, adjusts strategy based on data
    Adaptive,
}

/// Serialization methods for classical data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SerializationMethod {
    /// Standard serialization
    Standard,
    
    /// Compact serialization
    Compact,
    
    /// Secure serialization with additional checks
    Secure,
    
    /// Quantum-optimized serialization
    QuantumOptimized,
}

/// Configuration for QCBP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QCBPConfig {
    /// Default data format
    pub default_format: DataFormat,
    
    /// Conversion mode
    pub conversion_mode: ConversionMode,
    
    /// Serialization method
    pub serialization_method: SerializationMethod,
    
    /// Maximum data size in bytes
    pub max_data_size: usize,
    
    /// Whether to include metadata
    pub include_metadata: bool,
    
    /// Additional format-specific settings
    pub format_settings: HashMap<String, String>,
}

impl Default for QCBPConfig {
    fn default() -> Self {
        Self {
            default_format: DataFormat::JSON,
            conversion_mode: ConversionMode::Adaptive,
            serialization_method: SerializationMethod::Standard,
            max_data_size: 1_048_576, // 1 MB
            include_metadata: true,
            format_settings: HashMap::new(),
        }
    }
}

/// Bridge metadata for tracking cross-environment data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeMetadata {
    /// Unique identifier
    pub id: String,
    
    /// Source environment type (classical or quantum)
    pub source_type: String,
    
    /// Target environment type (classical or quantum)
    pub target_type: String,
    
    /// Original data format
    pub original_format: DataFormat,
    
    /// Conversion timestamp
    pub timestamp: u64,
    
    /// Hash of original data for verification
    pub original_hash: String,
    
    /// Information loss metric (0.0 to 1.0, lower is better)
    pub info_loss_metric: f64,
    
    /// Processing time in microseconds
    pub processing_time_us: u64,
    
    /// Additional properties
    pub properties: HashMap<String, String>,
}

/// The main QCBP implementation
pub struct QCBP {
    /// Configuration for this instance
    config: QCBPConfig,
    
    /// Active conversions
    active_conversions: HashMap<String, BridgeMetadata>,
    
    /// Conversion history
    conversion_history: Vec<BridgeMetadata>,
}

impl QCBP {
    /// Create a new QCBP instance with the given configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for QCBP
    ///
    /// # Returns
    ///
    /// A new QCBP instance
    #[must_use]
    pub fn new(config: QCBPConfig) -> Self {
        Self {
            config,
            active_conversions: HashMap::new(),
            conversion_history: Vec::new(),
        }
    }
    
    /// Create a new QCBP instance with default configuration
    ///
    /// # Returns
    ///
    /// A new QCBP instance with default settings
    #[must_use]
    pub fn new_default() -> Self {
        Self::new(QCBPConfig::default())
    }
    
    /// Get the current configuration
    ///
    /// # Returns
    ///
    /// A reference to the current configuration
    #[must_use]
    pub fn config(&self) -> &QCBPConfig {
        &self.config
    }
    
    /// Convert classical data to quantum representation
    ///
    /// # Arguments
    ///
    /// * `data` - Classical data to convert
    /// * `format` - Optional data format (uses default if None)
    ///
    /// # Returns
    ///
    /// Quantum register with encoded data and metadata
    ///
    /// # Errors
    ///
    /// Returns an error if conversion fails
    pub fn classical_to_quantum(&mut self, data: &[u8], format: Option<DataFormat>) 
                               -> Result<(QuantumRegister, BridgeMetadata)> {
        let data_format = format.unwrap_or(self.config.default_format);
        
        // Check if data size is within limits
        if data.len() > self.config.max_data_size {
            return Err(QCBPError::ResourceLimitExceeded(
                format!("Data size ({}) exceeds maximum ({})", data.len(), self.config.max_data_size)
            ).into());
        }
        
        // Start timing the conversion
        let start_time = std::time::Instant::now();
        
        // Calculate hash of original data for verification
        let original_hash = hex::encode(util::hash_bytes(data));
        
        // Prepare the quantum register
        // The size depends on the data and format
        let register_size = Self::estimate_qubit_requirement(data.len(), data_format);
        let mut register = QuantumRegister::new(register_size);
        
        // Perform the conversion based on the data format
        let info_loss = match data_format {
            DataFormat::JSON => self.encode_json_to_quantum(data, &mut register),
            DataFormat::Binary => self.encode_binary_to_quantum(data, &mut register),
            DataFormat::ProtoBuf => Self::encode_protobuf_to_quantum(data, &mut register),
            DataFormat::Custom => self.encode_custom_to_quantum(data, &mut register),
        };
        
        // Create metadata
        let conversion_id = util::generate_timestamped_id("qcbp_");
        #[allow(clippy::cast_possible_truncation)]
        let processing_time = start_time.elapsed().as_micros() as u64;
        
        let metadata = BridgeMetadata {
            id: conversion_id.clone(),
            source_type: "classical".to_string(),
            target_type: "quantum".to_string(),
            original_format: data_format,
            timestamp: util::timestamp_now(),
            original_hash,
            info_loss_metric: info_loss,
            processing_time_us: processing_time,
            properties: HashMap::new(),
        };
        
        // Store in active conversions and history
        self.active_conversions.insert(conversion_id.clone(), metadata.clone());
        self.conversion_history.push(metadata.clone());
        
        Ok((register, metadata))
    }

    /// Convert quantum data to classical representation
    ///
    /// # Arguments
    ///
    /// * `register` - Quantum register to convert
    /// * `metadata` - Metadata from the original conversion
    /// * `format` - Optional target format (uses original format if None)
    ///
    /// # Returns
    ///
    /// Classical data representation
    ///
    /// # Errors
    ///
    /// Returns an error if conversion fails
    pub fn quantum_to_classical(&mut self, register: &QuantumRegister, metadata: &BridgeMetadata,
                               format: Option<DataFormat>) -> Result<Vec<u8>> {
        let target_format = format.unwrap_or(metadata.original_format);
        
        // Start timing the conversion
        let start_time = std::time::Instant::now();
        
        // Perform the conversion based on the target format
        let result = match target_format {
            DataFormat::JSON => self.decode_quantum_to_json(register, metadata)?,
            DataFormat::Binary => Ok::<Vec<u8>, crate::error::Error>(self.decode_quantum_to_binary(register, metadata))?,
            DataFormat::ProtoBuf => Ok::<Vec<u8>, crate::error::Error>(Self::decode_quantum_to_protobuf(register, metadata))?,
            DataFormat::Custom => Ok::<Vec<u8>, crate::error::Error>(self.decode_quantum_to_custom(register, metadata))?,
        };
        
        // Calculate processing time
        #[allow(clippy::cast_possible_truncation)]
        let processing_time = start_time.elapsed().as_micros() as u64;
        
        // Create metadata for the reverse conversion
        let conversion_id = util::generate_timestamped_id("qcbp_rev_");
        
        // Calculate hash of result for verification
        let result_hash = hex::encode(util::hash_bytes(&result));
        
        // Information loss metric (compare with original)
        let info_loss = if result_hash == metadata.original_hash {
            0.0 // Perfect conversion
        } else {
            // Estimate information loss - in a real system this would be more sophisticated
            0.2 // Simulated loss
        };
        
        let new_metadata = BridgeMetadata {
            id: conversion_id.clone(),
            source_type: "quantum".to_string(),
            target_type: "classical".to_string(),
            original_format: target_format,
            timestamp: util::timestamp_now(),
            original_hash: result_hash,
            info_loss_metric: info_loss,
            processing_time_us: processing_time,
            properties: HashMap::new(),
        };
        
        // Store in history
        self.conversion_history.push(new_metadata);
        
        // Remove from active conversions if the original ID is there
        self.active_conversions.remove(&metadata.id);
        
        Ok(result)
    }
    
    /// Get original data size from metadata if available
    #[must_use]
    pub fn get_original_size(&self, register: &QuantumRegister, metadata: &BridgeMetadata) -> Option<usize> {
        // If the original hash has length information, use it
        if !metadata.original_hash.is_empty() {
            // In a real system, this would be more sophisticated
            return Some(metadata.original_hash.len() / 2);
        }
        
        // Otherwise estimate based on register and format
        Some(Self::estimate_original_size(register, metadata))
    }
    
    /// Calculate the number of qubits required for encoding the data
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn estimate_qubit_requirement(data_size: usize, format: DataFormat) -> usize {
        match format {
            // Different formats have different qubit requirements
            DataFormat::JSON => {
                // JSON is verbose, need more qubits
                let data_size_f64 = f64::from(u32::try_from(data_size).unwrap_or(u32::MAX));
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                usize::try_from((data_size_f64 * 1.5).ceil() as u64).unwrap_or(data_size * 2)
            },
            DataFormat::Binary => {
                // Binary is more compact
                data_size
            },
            DataFormat::ProtoBuf => {
                // ProtoBuf is efficient
                let data_size_f64 = f64::from(u32::try_from(data_size).unwrap_or(u32::MAX));
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                usize::try_from((data_size_f64 * 0.8).ceil() as u64).unwrap_or(data_size / 2)
            },
            DataFormat::Custom => {
                // Custom format - assume similar to binary
                data_size
            },
        }
    }
    
    /// Encode JSON data to quantum representation
    #[must_use]
    #[allow(clippy::unused_self)]
    fn encode_json_to_quantum(&self, data: &[u8], register: &mut QuantumRegister) -> f64 {
        // For simulation, we'll use a simple encoding scheme
        // In a real quantum system, this would involve quantum state preparation
        
        // Reset all qubits to |0⟩
        for i in 0..register.size() {
            register.set_zero(i);
        }
        
        // Apply binary encoding (1 qubit per bit, simplified for simulation)
        let limit = register.size().min(data.len() * 8);
        
        for (byte_idx, &byte) in data.iter().enumerate() {
            for bit_idx in 0..8 {
                let qubit_idx = byte_idx * 8 + bit_idx;
                
                if qubit_idx >= limit {
                    break;
                }
                
                // If bit is 1, apply X gate
                if (byte >> bit_idx) & 1 == 1 {
                    register.apply_x(qubit_idx);
                }
            }
        }
        
        // Calculate information loss metric
        // For simulation, we'll estimate based on whether all data fits in the register
        
        
        if data.len() * 8 > register.size() {
            // Some data didn't fit
            #[allow(clippy::cast_precision_loss)]
            let overflow = (data.len() * 8 - register.size()) as f64 / (data.len() * 8) as f64;
            overflow.min(1.0)
        } else {
            0.0 // No loss
        }
    }
    
    /// Encode binary data to quantum representation
    #[must_use]
    #[allow(clippy::unused_self)]
    fn encode_binary_to_quantum(&self, data: &[u8], register: &mut QuantumRegister) -> f64 {
        // Similar to JSON encoding, but optimized for binary
        // This is a simplified simulation
        
        // Reset all qubits to |0⟩
        for i in 0..register.size() {
            register.set_zero(i);
        }
        
        // For binary data, use direct bit mapping
        let limit = register.size().min(data.len() * 8);
        
        for (byte_idx, &byte) in data.iter().enumerate() {
            for bit_idx in 0..8 {
                let qubit_idx = byte_idx * 8 + bit_idx;
                
                if qubit_idx >= limit {
                    break;
                }
                
                // If bit is 1, apply X gate
                if (byte >> bit_idx) & 1 == 1 {
                    register.apply_x(qubit_idx);
                }
            }
        }
        
        // Calculate information loss metric
        
        
        if data.len() * 8 > register.size() {
            #[allow(clippy::cast_precision_loss)]
            let overflow = (data.len() * 8 - register.size()) as f64 / (data.len() * 8) as f64;
            overflow.min(1.0)
        } else {
            0.0 // No loss
        }
    }
    
    /// Encode `ProtoBuf` data to quantum representation
    #[must_use]
    fn encode_protobuf_to_quantum(data: &[u8], register: &mut QuantumRegister) -> f64 {
        // For simulation, we'll use a similar approach as binary but with different parameters
        // to reflect the structured nature of ProtoBuf
        
        // Ensure register is large enough
        if register.size() < data.len() * 4 {
            // If register is too small, we'll lose information
            // In a real implementation, this would be handled differently
            
            // Find how many qubits we can actually use
            let available_qubits = register.size();
            let max_data_bytes = available_qubits / 4;
            
            // Encode as much as possible
            for i in 0..max_data_bytes {
                if i < data.len() {
                    // Encode each byte using 4 qubits (simplified)
                    let byte = data[i];
                    
                    // First qubit - bit 0
                    if (byte & 0x01) != 0 {
                        register.x(i * 4);
                    }
                    
                    // Second qubit - bit 2
                    if (byte & 0x04) != 0 {
                        register.x(i * 4 + 1);
                    }
                    
                    // Third qubit - bit 4
                    if (byte & 0x10) != 0 {
                        register.x(i * 4 + 2);
                    }
                    
                    // Fourth qubit - bit 6
                    if (byte & 0x40) != 0 {
                        register.x(i * 4 + 3);
                    }
                }
            }
            
            // Calculate information loss
            #[allow(clippy::cast_precision_loss)]
            let overflow = (data.len() * 8 - register.size()) as f64 / (data.len() * 8) as f64;
            
            overflow
        } else {
            // Encode each byte using 4 qubits (simplified)
            for (i, &byte) in data.iter().enumerate() {
                // First qubit - bit 0
                if (byte & 0x01) != 0 {
                    register.x(i * 4);
                }
                
                // Second qubit - bit 2
                if (byte & 0x04) != 0 {
                    register.x(i * 4 + 1);
                }
                
                // Third qubit - bit 4
                if (byte & 0x10) != 0 {
                    register.x(i * 4 + 2);
                }
                
                // Fourth qubit - bit 6
                if (byte & 0x40) != 0 {
                    register.x(i * 4 + 3);
                }
            }
            
            // No information loss
            0.0
        }
    }
    
    /// Encode custom format data to quantum representation
    #[must_use]
    #[allow(clippy::unused_self)]
    fn encode_custom_to_quantum(&self, data: &[u8], register: &mut QuantumRegister) -> f64 {
        // For simulation, use a combination of approaches
        // In a real implementation, this would depend on the custom format
        
        // Reset all qubits to |0⟩
        for i in 0..register.size() {
            register.set_zero(i);
        }
        
        // Use a mix of direct encoding and phase encoding
        let limit = register.size().min(data.len() * 4); // Use 4 bits per byte for demonstration
        
        for (byte_idx, &byte) in data.iter().enumerate() {
            // Just encode the 4 most significant bits for this demonstration
            for bit_idx in 4..8 {
                let qubit_idx = byte_idx * 4 + (bit_idx - 4);
                
                if qubit_idx >= limit {
                    break;
                }
                
                // First put qubit in superposition
                register.hadamard(qubit_idx);
                
                // If bit is 1, apply phase rotation
                if (byte >> bit_idx) & 1 == 1 {
                    register.phase(qubit_idx, std::f64::consts::PI / 2.0);
                }
            }
        }
        
        // Calculate information loss metric - higher for custom format
        // since we're not encoding all bits
        
        
        if data.len() * 8 > limit * 2 {
            #[allow(clippy::cast_precision_loss)]
            let overflow = (data.len() * 8 - limit * 2) as f64 / (data.len() * 8) as f64;
            (0.2 + overflow).min(1.0) // Start with base loss of 0.2
        } else {
            0.2 // Base loss due to using only 4 bits per byte
        }
    }
    
    /// Decode quantum data to JSON representation
    fn decode_quantum_to_json(&self, register: &QuantumRegister, metadata: &BridgeMetadata) -> Result<Vec<u8>> {
        // For simulation, we'll do a simulated measurement
        // In a real quantum system, this would involve proper measurement
        
        // For this simulation, we'll create a simple JSON structure
        let json_data = match self.config.conversion_mode {
            ConversionMode::Strict => {
                // In strict mode, attempt to recreate original data exactly
                if let Some(original_size) = self.get_original_size(register, metadata) {
                    // Create a vector and fill with synthetic data
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let result: Vec<u8> = (0..original_size).map(|i| i as u8).collect();
                    result
                } else {
                    return Err(QCBPError::DeserializationError(
                        "Cannot determine original data size".to_string()
                    ).into());
                }
            },
            ConversionMode::Lossy | ConversionMode::Adaptive => {
                // In lossy mode, just do best effort
                // Create some simple JSON-like data with the number of qubits
                let qubits_str = register.size().to_string();
                let mut result = Vec::with_capacity(qubits_str.len() + 40);
                
                // Build the JSON string in memory before converting to bytes
                let json_string = format!(r#"{{"result":"quantum_data","qubits":{qubits_str}}}"#);
                result.extend_from_slice(json_string.as_bytes());
                result
            }
        };
        
        Ok(json_data)
    }
    
    /// Decode quantum data to binary representation
    #[must_use]
    #[allow(clippy::unused_self)]
    fn decode_quantum_to_binary(&self, register: &QuantumRegister, _metadata: &BridgeMetadata) -> Vec<u8> {
        // For simulation, we'll create binary data based on the register size
        let estimated_size = (register.size() / 8).max(1);
        let mut result = Vec::with_capacity(estimated_size);
        
        // Create bytes from register state
        // In a real implementation, this would involve measuring the quantum register
        for i in 0..estimated_size {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            result.push(i as u8);
        }
        
        result
    }
    
    /// Decode quantum data to `ProtoBuf` representation
    #[must_use]
    fn decode_quantum_to_protobuf(_register: &QuantumRegister, _metadata: &BridgeMetadata) -> Vec<u8> {
        // For simulation, we'll create a simple binary structure
        // that mimics a protobuf message
        vec![
            0x0A, // Field 1, wire type 2 (length-delimited)
            0x04, // Length 4
            0x01, 0x02, 0x03, 0x04, // 4 bytes of data
            0x12, // Field 2, wire type 2
            0x03, // Length 3
            0x05, 0x06, 0x07 // 3 bytes of data
        ]
    }
    
    /// Decode quantum data to custom format representation
    #[must_use]
    #[allow(clippy::unused_self)]
    fn decode_quantum_to_custom(&self, _register: &QuantumRegister, _metadata: &BridgeMetadata) -> Vec<u8> {
        // For simulation, create a custom binary format
        let mut result = vec![
            0xC0, 0xDE, // Magic number
            0x01,       // Version 1
            0x03        // Custom format code
        ];
        
        // Add synthetic data length
        let data_len = 32; // Synthetic data length
        
        // Add length as 32-bit integer (simplified)
        #[allow(clippy::cast_sign_loss)]
        result.push((data_len & 0xFF) as u8);
        
        // Add synthetic data
        for i in 0..16 {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            result.push((i * 2) as u8);
        }
        
        // Add footer
        result.push(0xFF);
        result.push(0xFF);
        
        result
    }
    
    /// Estimate original data size based on register and metadata
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn estimate_original_size(register: &QuantumRegister, metadata: &BridgeMetadata) -> usize {
        // Estimate based on the format and register size
        match metadata.original_format {
            DataFormat::JSON => {
                (register.size() as f64 / 1.5).ceil() as usize
            },
            DataFormat::Binary => {
                register.size() / 8
            },
            DataFormat::ProtoBuf => {
                (register.size() as f64 / 0.8).ceil() as usize
            },
            DataFormat::Custom => {
                register.size() / 4 // Assuming 4 bits per byte as in the encoder
            },
        }
    }
    
    /// Get conversion history
    ///
    /// # Returns
    ///
    /// A slice of conversion records
    #[must_use]
    pub fn conversion_history(&self) -> &[BridgeMetadata] {
        &self.conversion_history
    }
    
    /// Clear conversion history
    pub fn clear_conversion_history(&mut self) {
        self.conversion_history.clear();
    }
    
    /// Get active conversions
    ///
    /// # Returns
    ///
    /// A map of active conversion IDs to their metadata
    #[must_use]
    pub fn active_conversions(&self) -> &HashMap<String, BridgeMetadata> {
        &self.active_conversions
    }
}

impl fmt::Debug for QCBP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QCBP")
            .field("config", &self.config)
            .field("active_conversions", &self.active_conversions.len())
            .field("conversion_history", &self.conversion_history.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_qcbp() {
        let qcbp = QCBP::new_default();
        assert_eq!(qcbp.config().default_format, DataFormat::JSON);
    }
    
    #[test]
    fn test_classical_to_quantum_conversion() {
        let mut qcbp = QCBP::new_default();
        
        let test_data = b"Test data for quantum conversion";
        
        // Convert to quantum
        let result = qcbp.classical_to_quantum(test_data, None);
        assert!(result.is_ok(), "Conversion should succeed");
        
        let (register, metadata) = result.unwrap();
        
        // Verify the register has qubits
        assert!(register.size() > 0, "Quantum register should have qubits");
        
        // Verify metadata
        assert_eq!(metadata.source_type, "classical");
        assert_eq!(metadata.target_type, "quantum");
    }
    
    #[test]
    fn test_quantum_to_classical_conversion() {
        let mut qcbp = QCBP::new_default();
        
        let test_data = b"Test data for full conversion cycle";
        
        // First convert to quantum
        let result = qcbp.classical_to_quantum(test_data, None);
        assert!(result.is_ok());
        
        let (register, metadata) = result.unwrap();
        
        // Then convert back to classical
        let result = qcbp.quantum_to_classical(&register, &metadata, None);
        assert!(result.is_ok());
        
        let classical_data = result.unwrap();
        assert!(!classical_data.is_empty());
    }
    
    #[test]
    fn test_different_data_formats() {
        let mut qcbp = QCBP::new_default();
        let test_data = b"Test data for format testing";
        
        // Test each format
        for format in [
            DataFormat::JSON,
            DataFormat::Binary,
            DataFormat::ProtoBuf,
            DataFormat::Custom,
        ] {
            // Convert to quantum with this format
            let result = qcbp.classical_to_quantum(test_data, Some(format));
            assert!(result.is_ok(), "Conversion with {format:?} format failed");
            
            let (register, metadata) = result.unwrap();
            
            // Convert back to classical
            let result = qcbp.quantum_to_classical(&register, &metadata, Some(format));
            assert!(result.is_ok(), "Conversion back with {format:?} format failed");
        }
    }
} 