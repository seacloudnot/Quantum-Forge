// Quantum Hardware Extraction Protocol (QHEP)
//
// This protocol abstracts hardware details for portable implementations.

use std::fmt;
use std::collections::HashMap;
use thiserror::Error;
use serde::{Serialize, Deserialize};

use crate::error::Result;
use crate::util;

/// Errors specific to the QHEP protocol
#[derive(Debug, Error)]
pub enum QHEPError {
    #[error("Unsupported hardware capability: {0}")]
    UnsupportedCapability(String),
    
    #[error("Hardware instruction translation failed: {0}")]
    InstructionTranslationFailed(String),
    
    #[error("Resource negotiation failed: {0}")]
    ResourceNegotiationFailed(String),
    
    #[error("Hardware detection failed: {0}")]
    HardwareDetectionFailed(String),
    
    #[error("Instruction set incompatibility: {0}")]
    InstructionSetIncompatibility(String),
    
    #[error("Protocol error: {0}")]
    ProtocolError(String),
}

/// Quantum hardware architecture types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareArchitecture {
    /// Superconducting qubits
    Superconducting,
    
    /// Trapped ion qubits
    TrappedIon,
    
    /// Photonic qubits
    Photonic,
    
    /// Neutral atom qubits
    NeutralAtom,
    
    /// Topological qubits
    Topological,
    
    /// Semiconductor spin qubits
    SemiconductorSpin,
    
    /// Simulated quantum hardware
    Simulated,
}

/// Quantum instruction set categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InstructionSetType {
    /// Pulse-level control
    PulseLevel,
    
    /// Gate-level control
    GateLevel,
    
    /// Circuit-level control
    CircuitLevel,
    
    /// High-level control (algorithmic)
    HighLevel,
}

/// Resource negotiation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceStrategy {
    /// Use minimum resources required
    Minimal,
    
    /// Balance speed and resource use
    Balanced,
    
    /// Prioritize performance
    Performance,
    
    /// Prioritize reliability
    Reliability,
    
    /// Adaptive based on availability
    Adaptive,
}

/// Configuration for QHEP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QHEPConfig {
    /// Target hardware architecture
    pub target_architecture: Option<HardwareArchitecture>,
    
    /// Preferred instruction set
    pub preferred_instruction_set: InstructionSetType,
    
    /// Resource negotiation strategy
    pub resource_strategy: ResourceStrategy,
    
    /// Timeout for hardware detection (ms)
    pub detection_timeout_ms: u64,
    
    /// Allow fallback to simulation
    pub allow_simulation_fallback: bool,
    
    /// Require hardware verification
    pub require_hardware_verification: bool,
}

impl Default for QHEPConfig {
    fn default() -> Self {
        Self {
            target_architecture: None, // Auto-detect
            preferred_instruction_set: InstructionSetType::GateLevel,
            resource_strategy: ResourceStrategy::Balanced,
            detection_timeout_ms: 5000, // 5 seconds
            allow_simulation_fallback: true,
            require_hardware_verification: false,
        }
    }
}

/// Hardware capability descriptor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareCapability {
    /// Unique identifier
    pub id: String,
    
    /// Hardware architecture
    pub architecture: HardwareArchitecture,
    
    /// Available instruction sets
    pub instruction_sets: Vec<InstructionSetType>,
    
    /// Number of qubits available
    pub qubit_count: usize,
    
    /// Supported gate operations
    pub supported_gates: Vec<String>,
    
    /// Error rates
    pub error_rates: HashMap<String, f64>,
    
    /// Coherence times (in μs)
    pub coherence_times: HashMap<String, f64>,
    
    /// Connectivity map (adjacency list)
    pub connectivity: HashMap<usize, Vec<usize>>,
    
    /// Hardware-specific parameters
    pub additional_params: HashMap<String, String>,
}

/// Instruction translation map
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstructionTranslation {
    /// Source instruction
    pub source: String,
    
    /// Target hardware instruction
    pub target: String,
    
    /// Required parameters
    pub parameters: Vec<String>,
    
    /// Fidelity of translation
    pub fidelity: f64,
    
    /// Execution time (in ns)
    pub execution_time_ns: u64,
}

/// Resource allocation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// Assigned qubits
    pub assigned_qubits: Vec<usize>,
    
    /// Assigned classical memory
    pub assigned_memory: usize,
    
    /// Maximum circuit depth
    pub max_circuit_depth: usize,
    
    /// Time quantum (in μs)
    pub time_allocation_us: u64,
    
    /// Priority level
    pub priority: u8,
}

/// The main QHEP implementation
pub struct QHEP {
    /// Configuration for this instance
    config: QHEPConfig,
    
    /// Detected hardware capabilities
    hardware_capabilities: Option<HardwareCapability>,
    
    /// Instruction translations
    instruction_translations: HashMap<String, InstructionTranslation>,
    
    /// Current resource allocation
    resource_allocation: Option<ResourceAllocation>,
    
    /// Whether hardware is available
    hardware_available: bool,
    
    /// Currently active operation
    current_operation: Option<String>,
}

impl QHEP {
    /// Create a new QHEP instance with the given configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for QHEP
    ///
    /// # Returns
    ///
    /// A new QHEP instance
    #[must_use]
    pub fn new(config: QHEPConfig) -> Self {
        Self {
            config,
            hardware_capabilities: None,
            instruction_translations: HashMap::new(),
            resource_allocation: None,
            hardware_available: false,
            current_operation: None,
        }
    }
    
    /// Create a new QHEP instance with default configuration
    ///
    /// # Returns
    ///
    /// A new QHEP instance with default settings
    #[must_use]
    pub fn new_default() -> Self {
        Self::new(QHEPConfig::default())
    }
    
    /// Get the current configuration
    ///
    /// # Returns
    ///
    /// A reference to the current configuration
    #[must_use]
    pub fn config(&self) -> &QHEPConfig {
        &self.config
    }
    
    /// Detect available quantum hardware
    ///
    /// # Returns
    ///
    /// Hardware capability information if available
    ///
    /// # Errors
    ///
    /// Returns an error if hardware detection fails
    ///
    /// # Panics
    ///
    /// Panics if hardware detection succeeds but the internal state is inconsistent
    pub fn detect_hardware(&mut self) -> Result<&HardwareCapability> {
        // In a real implementation, this would communicate with hardware drivers
        // to discover available quantum hardware
        
        // For this implementation, we'll simulate hardware detection
        
        // Check if the targeted architecture is available
        let simulated_architecture = self.config.target_architecture.unwrap_or(HardwareArchitecture::Simulated);
        
        // Prepare capability information
        let capability = HardwareCapability {
            id: util::generate_id("hw_"),
            architecture: simulated_architecture,
            instruction_sets: vec![
                InstructionSetType::GateLevel,
                InstructionSetType::CircuitLevel,
            ],
            qubit_count: 16,
            supported_gates: vec![
                "x".to_string(),
                "y".to_string(),
                "z".to_string(),
                "h".to_string(),
                "cnot".to_string(),
                "swap".to_string(),
                "t".to_string(),
                "s".to_string(),
            ],
            error_rates: {
                let mut rates = HashMap::new();
                rates.insert("single_qubit".to_string(), 0.001);
                rates.insert("two_qubit".to_string(), 0.01);
                rates.insert("measurement".to_string(), 0.05);
                rates
            },
            coherence_times: {
                let mut times = HashMap::new();
                times.insert("t1".to_string(), 50.0);
                times.insert("t2".to_string(), 25.0);
                times
            },
            connectivity: {
                let mut connectivity = HashMap::new();
                // Simulated connectivity map (linear)
                for i in 0..15 {
                    connectivity.insert(i, vec![i + 1]);
                }
                connectivity.insert(15, vec![0]); // Close the loop
                connectivity
            },
            additional_params: HashMap::new(),
        };
        
        self.hardware_capabilities = Some(capability);
        self.hardware_available = true;
        
        // Return a reference to the stored capability
        self.hardware_capabilities.as_ref().ok_or_else(|| QHEPError::HardwareDetectionFailed("Hardware detection succeeded but internal state is inconsistent".to_string()).into())
    }
    
    /// Create instruction translations for target hardware
    ///
    /// # Arguments
    ///
    /// * `source_instructions` - List of source instructions to translate
    ///
    /// # Returns
    ///
    /// Map of source instructions to their hardware translations
    ///
    /// # Errors
    ///
    /// Returns an error if translation fails
    ///
    /// # Panics
    ///
    /// May panic if decomposition of an instruction fails unexpectedly
    pub fn create_instruction_translations(&mut self, source_instructions: &[String]) -> Result<HashMap<String, InstructionTranslation>> {
        // Ensure we have hardware capabilities
        if self.hardware_capabilities.is_none() {
            // Try to detect hardware first
            self.detect_hardware()?;
        }
        
        let capabilities = self
            .hardware_capabilities
            .as_ref()
            .ok_or_else(|| QHEPError::HardwareDetectionFailed("Hardware detection required first".to_string()))?;
        
        // Create translations for each instruction
        let translations: HashMap<String, InstructionTranslation> = source_instructions.iter()
            .map(|source| {
                // Check if this instruction is directly supported
                let is_directly_supported = capabilities.supported_gates.contains(source);
                
                let translation = if is_directly_supported {
                    // Direct translation
                    InstructionTranslation {
                        source: source.clone(),
                        target: source.clone(),
                        parameters: vec!["qubit".to_string()],
                        fidelity: 0.99,
                        execution_time_ns: 100,
                    }
                } else {
                    // Need to decompose to supported gates
                    Self::decompose_to_supported_gates(source, capabilities)
                        .unwrap_or_else(|_| panic!("Failed to decompose instruction: {source}"))
                };
                
                (source.clone(), translation)
            })
            .collect();
        
        // Update the stored translations
        self.instruction_translations.clone_from(&translations);
        
        Ok(translations)
    }
    
    /// Decompose an instruction into supported gates
    #[allow(dead_code)]
    fn decompose_to_supported_gates(instruction: &str, capabilities: &HardwareCapability) -> Result<InstructionTranslation> {
        // In a real implementation, this would use a decomposition algorithm
        // to express a higher-level gate in terms of the hardware's native gates
        
        // For this simulation, we'll handle a few common cases
        let decomposition = match instruction.to_lowercase().as_str() {
            "rx" => {
                if capabilities.supported_gates.contains(&"x".to_string()) {
                    ("x", vec!["qubit".to_string(), "angle".to_string()], 0.98, 150)
                } else {
                    ("h;z;h", vec!["qubit".to_string(), "angle".to_string()], 0.95, 300)
                }
            },
            "ry" => {
                if capabilities.supported_gates.contains(&"y".to_string()) {
                    ("y", vec!["qubit".to_string(), "angle".to_string()], 0.98, 150)
                } else {
                    ("x;h;x", vec!["qubit".to_string(), "angle".to_string()], 0.94, 350)
                }
            },
            "toffoli" => {
                if capabilities.supported_gates.contains(&"cnot".to_string()) {
                    ("h;cnot;t;cnot;t;h;cnot;t;cnot", 
                     vec!["control1".to_string(), "control2".to_string(), "target".to_string()],
                     0.92, 800)
                } else {
                    return Err(QHEPError::UnsupportedCapability(
                        format!("Cannot decompose {instruction} without CNOT capability")
                    ).into());
                }
            },
            _ => {
                return Err(QHEPError::UnsupportedCapability(
                    format!("Unsupported instruction: {instruction}")
                ).into());
            }
        };
        
        let (target, params, fidelity, time) = decomposition;
        
        Ok(InstructionTranslation {
            source: instruction.to_string(),
            target: target.to_string(),
            parameters: params,
            fidelity,
            execution_time_ns: time,
        })
    }
    
    /// Negotiate resources based on requirements
    ///
    /// # Arguments
    ///
    /// * `required_qubits` - Number of qubits required
    /// * `circuit_depth` - Expected circuit depth
    /// * `execution_time` - Expected execution time in μs
    ///
    /// # Returns
    ///
    /// Resource allocation if successful
    ///
    /// # Errors
    ///
    /// Returns an error if resource negotiation fails
    pub fn negotiate_resources(&mut self, required_qubits: usize, circuit_depth: usize, execution_time: u64) -> Result<ResourceAllocation> {
        // Ensure we have hardware capabilities
        if self.hardware_capabilities.is_none() {
            // Try to detect hardware first
            self.detect_hardware()?;
        }
        
        let capabilities = self
            .hardware_capabilities
            .as_ref()
            .ok_or_else(|| QHEPError::HardwareDetectionFailed("Hardware detection required first".to_string()))?;
        
        // Check if we have enough qubits
        if required_qubits > capabilities.qubit_count {
            if self.config.allow_simulation_fallback {
                // Fall back to simulation
                println!("Falling back to simulation due to insufficient qubits");
            } else {
                return Err(QHEPError::ResourceNegotiationFailed(
                    format!("Required qubits ({required_qubits}) exceeds available ({})", capabilities.qubit_count)
                ).into());
            }
        }
        
        // Determine which qubits to assign based on the resource strategy
        let assigned_qubits = match self.config.resource_strategy {
            ResourceStrategy::Minimal => {
                // Just allocate the minimum required
                (0..required_qubits).collect()
            },
            ResourceStrategy::Balanced => {
                // Try to allocate qubits with good connectivity
                Self::find_well_connected_qubits(required_qubits, capabilities)
            },
            ResourceStrategy::Performance => {
                // Try to allocate the best performing qubits
                Self::find_best_performing_qubits(required_qubits, capabilities)
            },
            ResourceStrategy::Reliability => {
                // Try to allocate the most reliable qubits
                Self::find_most_reliable_qubits(required_qubits, capabilities) 
            },
            ResourceStrategy::Adaptive => {
                // Use a mix of strategies based on requirements
                if circuit_depth > 100 {
                    // Deep circuits need reliable qubits
                    Self::find_most_reliable_qubits(required_qubits, capabilities)
                } else if execution_time > 1000 {
                    // Long-running needs good coherence
                    Self::find_best_coherence_qubits(required_qubits, capabilities)
                } else {
                    // Default to well-connected
                    Self::find_well_connected_qubits(required_qubits, capabilities)
                }
            }
        };
        
        // Create resource allocation
        let allocation = ResourceAllocation {
            assigned_qubits,
            assigned_memory: required_qubits * 10, // 10 classical bits per qubit
            max_circuit_depth: circuit_depth.min(1000), // Cap at 1000 for simulation
            time_allocation_us: execution_time,
            priority: 1, // Default priority
        };
        
        // Store the allocation
        self.resource_allocation = Some(allocation.clone());
        
        Ok(allocation)
    }
    
    /// Find well-connected qubits for operation
    #[must_use]
    fn find_well_connected_qubits(count: usize, capabilities: &HardwareCapability) -> Vec<usize> {
        // In a real system, we would analyze topology
        // For simulation, just find qubits with most connections
        
        // Count connections for each qubit
        let mut connection_counts: Vec<(usize, usize)> = capabilities.connectivity.iter()
            .map(|(&qubit, connections)| (qubit, connections.len()))
            .collect();
        
        // Sort by number of connections (descending)
        connection_counts.sort_by(|a, b| b.1.cmp(&a.1));
        
        // Take the top 'count' qubits
        connection_counts.iter()
            .take(count)
            .map(|&(qubit, _)| qubit)
            .collect()
    }
    
    /// Find qubits with best performance
    #[must_use]
    fn find_best_performing_qubits(count: usize, capabilities: &HardwareCapability) -> Vec<usize> {
        // In a real system, we would have performance metrics
        // For simulation, use error rates from the capability info
        
        // Get single qubit error rate, if available
        let error_rate = capabilities.error_rates.get("single_qubit").copied().unwrap_or(0.01);
        
        // Simulate differences between qubits (in a real system, each qubit would have different performance)
        let mut qubit_rates: Vec<(usize, f64)> = (0..capabilities.qubit_count)
            .map(|i| {
                // Simulate slight variations between qubits
                #[allow(clippy::cast_precision_loss)]
                let rate_factor = 1.0 - ((i.min(50)) as f64 * 0.01).min(0.5);
                (i, error_rate * rate_factor)
            })
            .collect();
        
        // Sort by error rate (ascending)
        qubit_rates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take the top 'count' qubits
        qubit_rates.iter()
            .take(count)
            .map(|&(qubit, _)| qubit)
            .collect()
    }
    
    /// Find qubits with best reliability
    #[must_use]
    fn find_most_reliable_qubits(count: usize, capabilities: &HardwareCapability) -> Vec<usize> {
        // In a real system, this would use reliability metrics
        // For simulation, just return a sequence in reverse order
        let available_qubits = capabilities.qubit_count;
        (0..count.min(available_qubits)).collect()
    }
    
    /// Find qubits with best coherence times
    #[must_use]
    fn find_best_coherence_qubits(count: usize, capabilities: &HardwareCapability) -> Vec<usize> {
        // In a real implementation, this would select qubits with longest coherence times
        
        // For simulation, we'll pretend the latter half have better coherence
        let available_qubits = capabilities.qubit_count;
        let start = available_qubits.saturating_sub(count);
        (start..available_qubits).take(count).collect()
    }
    
    /// Translate a circuit to hardware-specific instructions
    ///
    /// # Arguments
    ///
    /// * `circuit` - List of circuit operations
    ///
    /// # Returns
    ///
    /// Translated circuit ready for hardware execution
    ///
    /// # Errors
    ///
    /// Returns an error if translation fails
    pub fn translate_circuit(&mut self, circuit: &[String]) -> Result<Vec<String>> {
        // Ensure we have translations
        if self.instruction_translations.is_empty() {
            // Extract unique instructions from the circuit using functional approach
            let unique_instructions: Vec<String> = circuit
                .iter()
                .filter_map(|op| op.split_whitespace().next())
                .map(String::from)
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            
            // Create translations for these instructions
            self.create_instruction_translations(&unique_instructions)?;
        }
        
        // Translate each operation in the circuit using functional approach
        let translated_circuit = circuit.iter()
            .flat_map(|operation| {
                // Parse the operation into instruction and parameters
                let parts: Vec<&str> = operation.split_whitespace().collect();
                if parts.is_empty() {
                    return Vec::new();
                }
                
                let instruction = parts[0];
                let params = &parts[1..];
                
                // Look up the translation
                if let Some(translation) = self.instruction_translations.get(instruction) {
                    // Split the target into individual operations and apply parameters
                    translation.target.split(';')
                        .map(|target_op| {
                            // Build the translated operation with parameters
                            let base_op = target_op.to_string();
                            
                            // Add parameters as needed
                            params.iter()
                                .take(translation.parameters.len())
                                .enumerate()
                                .fold(base_op, |mut op, (i, param)| {
                                    if i < translation.parameters.len() {
                                        op.push(' ');
                                        op.push_str(param);
                                    }
                                    op
                                })
                        })
                        .collect()
                } else {
                    // If translation not found, return empty vec - we'll handle error after flat_map
                    Vec::new()
                }
            })
            .collect::<Vec<String>>();
        
        // Check if we had any untranslated instructions (would result in empty translated_circuit 
        // if the original circuit wasn't empty)
        if circuit.is_empty() || !translated_circuit.is_empty() {
            Ok(translated_circuit)
        } else {
            // Find the first instruction that couldn't be translated
            if let Some(first_untranslated) = circuit.iter()
                .filter_map(|op| op.split_whitespace().next())
                .find(|&instr| !self.instruction_translations.contains_key(instr)) {
                
                Err(QHEPError::InstructionTranslationFailed(
                    format!("No translation found for instruction: {first_untranslated}")
                ).into())
            } else {
                // This shouldn't happen if our logic is correct, but just in case
                Err(QHEPError::InstructionTranslationFailed(
                    "Failed to translate circuit for unknown reason".to_string()
                ).into())
            }
        }
    }
    
    /// Verify hardware capabilities against requirements
    ///
    /// # Arguments
    ///
    /// * `requirements` - Map of capability names to minimum required values
    ///
    /// # Returns
    ///
    /// True if all requirements are satisfied
    ///
    /// # Errors
    ///
    /// Returns an error if verification fails
    ///
    /// # Panics
    ///
    /// Panics when checking `error_rate` or coherence requirements with invalid format
    pub fn verify_hardware_capabilities(&mut self, requirements: &HashMap<String, f64>) -> Result<bool> {
        // Ensure we have hardware capabilities
        if self.hardware_capabilities.is_none() {
            // Try to detect hardware first
            self.detect_hardware()?;
        }
        
        let capabilities = self
            .hardware_capabilities
            .as_ref()
            .ok_or_else(|| QHEPError::HardwareDetectionFailed("Hardware detection required first".to_string()))?;
        
        // Check each requirement
        for (key, &required_value) in requirements {
            if key.starts_with("error_") || key == "single_qubit" || key == "two_qubit" || key == "measurement" {
                // For error rates, the hardware value must be LESS than the required value
                // (smaller error is better)
                let hardware_value = capabilities.error_rates.get(
                    // Map the common names to the ones used in our error_rates HashMap
                    if key == "single_qubit" { "single_qubit" } 
                    else if key == "two_qubit" { "two_qubit" }
                    else if key == "measurement" { "measurement" }
                    else if let Some(stripped) = key.strip_prefix("error_") { stripped }
                    else { key }
                );
                
                match hardware_value {
                    Some(&value) => {
                        if value > required_value {
                            // Error rate is too high
                            return Ok(false);
                        }
                    },
                    None => {
                        return Err(QHEPError::UnsupportedCapability(
                            format!("Error rate not available for: {key}")
                        ).into());
                    }
                }
            } else if key.starts_with("coherence_") || key == "t1" || key == "t2" {
                // For coherence times, the hardware value must be GREATER than the required value
                // (longer coherence is better)
                let hardware_value = capabilities.coherence_times.get(
                    // Map the common names to the ones used in our coherence_times HashMap
                    if key == "t1" { "t1" }
                    else if key == "t2" { "t2" }
                    else if let Some(stripped) = key.strip_prefix("coherence_") { stripped }
                    else { key }
                );
                
                match hardware_value {
                    Some(&value) => {
                        if value < required_value {
                            // Coherence time is too short
                            return Ok(false);
                        }
                    },
                    None => {
                        return Err(QHEPError::UnsupportedCapability(
                            format!("Coherence time not available for: {key}")
                        ).into());
                    }
                }
            } else {
                // For other capability types, we would need to implement specific comparison logic
                return Err(QHEPError::UnsupportedCapability(
                    format!("Unknown capability requirement: {key}")
                ).into());
            }
        }
        
        // All requirements are satisfied
        Ok(true)
    }
    
    /// Begin hardware operation with the current configuration
    ///
    /// # Returns
    ///
    /// Operation ID if successful
    ///
    /// # Errors
    ///
    /// Returns an error if operation cannot be started
    pub fn begin_operation(&mut self) -> Result<String> {
        // Ensure we have hardware capabilities and resource allocation
        if self.hardware_capabilities.is_none() {
            self.detect_hardware()?;
        }
        
        if self.resource_allocation.is_none() {
            return Err(QHEPError::ResourceNegotiationFailed(
                "Resource negotiation required before operation".to_string()
            ).into());
        }
        
        // Create an operation ID
        let operation_id = util::generate_timestamped_id("op_");
        self.current_operation = Some(operation_id.clone());
        
        Ok(operation_id)
    }
    
    /// End the current hardware operation
    ///
    /// # Arguments
    ///
    /// * `operation_id` - ID of the operation to end
    ///
    /// # Returns
    ///
    /// Success indicator
    ///
    /// # Errors
    ///
    /// Returns an error if operation cannot be ended
    pub fn end_operation(&mut self, operation_id: &str) -> Result<bool> {
        if let Some(current_id) = &self.current_operation {
            if current_id == operation_id {
                self.current_operation = None;
                self.resource_allocation = None;
                Ok(true)
            } else {
                Err(QHEPError::ProtocolError(
                    format!("Operation ID mismatch: expected {current_id}, got {operation_id}")
                ).into())
            }
        } else {
            Err(QHEPError::ProtocolError(
                "No active operation to end".to_string()
            ).into())
        }
    }
    
    /// Get the current hardware capabilities
    ///
    /// # Returns
    ///
    /// Current hardware capabilities if available
    #[must_use]
    pub fn hardware_capabilities(&self) -> Option<&HardwareCapability> {
        self.hardware_capabilities.as_ref()
    }
    
    /// Get the current resource allocation
    ///
    /// # Returns
    ///
    /// Current resource allocation if available
    #[must_use]
    pub fn resource_allocation(&self) -> Option<&ResourceAllocation> {
        self.resource_allocation.as_ref()
    }
    
    /// Check if hardware is available
    ///
    /// # Returns
    ///
    /// True if hardware is available
    #[must_use]
    pub fn is_hardware_available(&self) -> bool {
        self.hardware_available
    }
    
    /// Find available qubits
    #[must_use]
    #[allow(unused_variables)]
    pub fn find_available_qubits(&self, count: usize, _capabilities: &HardwareCapability) -> Vec<usize> {
        // In a real system, we would analyze hardware capabilities
        // For simulation, just return a sequence of qubit indices
        let available_qubits = self.hardware_capabilities.as_ref().map_or(0, |cap| cap.qubit_count);
        (0..count.min(available_qubits)).collect()
    }
    
    /// Get most used qubits for simulation
    ///
    /// This uses a heuristic to select qubits based on usage history
    #[must_use]
    #[allow(unused_variables)]
    pub fn get_most_used_qubits(&self, count: usize, _capabilities: &HardwareCapability) -> Vec<usize> {
        // In a real system, we would analyze historical data
        // For simulation, just return a sequence of qubit indices
        let available_qubits = self.hardware_capabilities.as_ref().map_or(0, |cap| cap.qubit_count);
        (0..count.min(available_qubits)).collect()
    }
    
    /// Get least noisy qubits for simulation
    ///
    /// This uses a heuristic to select qubits based on estimated noise levels
    #[must_use]
    pub fn get_least_noisy_qubits(&self, count: usize, _capabilities: &HardwareCapability) -> Vec<usize> {
        // In a real system, we would have noise characterization data
        // For simulation, assume lower indices have less noise
        let available_qubits = self.hardware_capabilities.as_ref().map_or(0, |cap| cap.qubit_count);
        (0..count.min(available_qubits)).collect()
    }
    
    /// Get most stable qubits for simulation
    ///
    /// This uses a heuristic to select qubits based on estimated stability
    #[must_use]
    pub fn get_most_stable_qubits(&self, count: usize, _capabilities: &HardwareCapability) -> Vec<usize> {
        // In a real system, we would have stability characterization data
        // For simulation, just return indices in reverse order
        let available_qubits = self.hardware_capabilities.as_ref().map_or(0, |cap| cap.qubit_count);
        let max = available_qubits;
        (0..count.min(max)).map(|i| max - i - 1).collect()
    }
}

impl fmt::Debug for QHEP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QHEP")
            .field("config", &self.config)
            .field("hardware_capabilities", &self.hardware_capabilities)
            .field("instruction_translations", &self.instruction_translations.len())
            .field("resource_allocation", &self.resource_allocation)
            .field("hardware_available", &self.hardware_available)
            .field("current_operation", &self.current_operation)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hardware_detection() {
        let mut qhep = QHEP::new_default();
        
        let result = qhep.detect_hardware();
        assert!(result.is_ok(), "Hardware detection failed");
        
        let capabilities = result.unwrap();
        assert!(capabilities.qubit_count > 0, "No qubits available");
        assert!(!capabilities.supported_gates.is_empty(), "No supported gates");
    }
    
    #[test]
    fn test_instruction_translation() {
        let mut qhep = QHEP::new_default();
        
        // First detect hardware
        qhep.detect_hardware().unwrap();
        
        // Test instruction translation
        let instructions = vec![
            "h".to_string(),
            "x".to_string(),
            "cnot".to_string(),
        ];
        
        let result = qhep.create_instruction_translations(&instructions);
        assert!(result.is_ok(), "Instruction translation failed");
        
        let translations = result.unwrap();
        assert_eq!(translations.len(), 3, "Wrong number of translations");
        
        // Check that all requested instructions were translated
        for instruction in &instructions {
            assert!(translations.contains_key(instruction), 
                    "Missing translation for {instruction}");
        }
    }
    
    #[test]
    fn test_resource_negotiation() {
        let mut qhep = QHEP::new_default();
        
        // First detect hardware
        qhep.detect_hardware().unwrap();
        
        // Test resource negotiation
        let result = qhep.negotiate_resources(8, 100, 1000);
        assert!(result.is_ok(), "Resource negotiation failed");
        
        let allocation = result.unwrap();
        assert_eq!(allocation.assigned_qubits.len(), 8, "Wrong number of qubits allocated");
        assert!(allocation.assigned_memory > 0, "No memory allocated");
    }
    
    #[test]
    fn test_circuit_translation() {
        let mut qhep = QHEP::new_default();
        
        // First detect hardware
        qhep.detect_hardware().unwrap();
        
        // Create a simple circuit
        let circuit = vec![
            "h 0".to_string(),
            "cnot 0 1".to_string(),
            "x 1".to_string(),
        ];
        
        // Translate the circuit
        let result = qhep.translate_circuit(&circuit);
        assert!(result.is_ok(), "Circuit translation failed");
        
        let translated_circuit = result.unwrap();
        assert!(!translated_circuit.is_empty(), "Translated circuit is empty");
    }
} 