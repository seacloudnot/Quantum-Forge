// QHEP Hardware Integration Module
//
// This module provides interfaces to real quantum hardware
// for the Quantum Hardware Extraction Protocol.

use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use rand::random;
use thiserror::Error;
use crate::util;
use rand::Rng;

use crate::integration::qhep::{HardwareArchitecture, HardwareCapability, InstructionSetType, QHEPError};

/// Errors that can occur during hardware interfacing
#[derive(Debug, Error)]
pub enum HardwareInterfaceError {
    #[error("Failed to connect to hardware: {0}")]
    ConnectionFailed(String),
    
    #[error("Hardware not initialized: {0}")]
    NotInitialized(String),
    
    #[error("Hardware calibration error: {0}")]
    CalibrationError(String),
    
    #[error("Hardware execution error: {0}")]
    ExecutionError(String),
    
    #[error("Hardware authentication failed: {0}")]
    AuthenticationFailed(String),
    
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
    
    #[error("Hardware busy: {0}")]
    HardwareBusy(String),
    
    #[error("Hardware communication error: {0}")]
    CommunicationError(String),
    
    #[error("Invalid hardware configuration: {0}")]
    InvalidConfiguration(String),
}

/// Authentication method for hardware interface
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthMethod {
    /// API key authentication
    ApiKey(String),
    
    /// Username and password
    UserPass {
        username: String,
        password: String,
    },
    
    /// OAuth token
    OAuth(String),
    
    /// Client certificate
    Certificate {
        cert_path: String,
        key_path: String,
    },
}

/// Hardware connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConnectionConfig {
    /// Hardware provider name
    pub provider: String,
    
    /// Connection endpoint (URL, IP, etc.)
    pub endpoint: String,
    
    /// Authentication method
    pub auth: AuthMethod,
    
    /// Connection timeout in milliseconds
    pub timeout_ms: u64,
    
    /// Whether to use secure connection
    pub secure: bool,
    
    /// Provider-specific configuration options
    pub options: HashMap<String, String>,
}

impl Default for HardwareConnectionConfig {
    fn default() -> Self {
        Self {
            provider: "simulator".to_string(),
            endpoint: "localhost:8000".to_string(),
            auth: AuthMethod::ApiKey("default_api_key".to_string()),
            timeout_ms: 5000,
            secure: true,
            options: HashMap::new(),
        }
    }
}

/// Provider-specific hardware executor trait
pub trait HardwareExecutor: Send + Sync {
    /// Initialize connection to hardware
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails due to connection issues,
    /// authentication problems, or invalid configuration.
    fn initialize(&mut self, config: &HardwareConnectionConfig) -> Result<()>;
    
    /// Check if hardware is initialized
    fn is_initialized(&self) -> bool;
    
    /// Discover hardware capabilities
    ///
    /// # Errors
    ///
    /// Returns an error if hardware is not initialized or if capability
    /// discovery fails due to connection issues.
    fn discover_capabilities(&mut self) -> Result<HardwareCapability>;
    
    /// Execute quantum circuit
    ///
    /// # Errors
    ///
    /// Returns an error if circuit execution fails due to hardware errors,
    /// invalid circuit specification, or connection issues.
    fn execute_circuit(&mut self, _circuit: &[String], qubit_mapping: &[usize]) -> Result<Vec<u8>>;
    
    /// Get hardware status information
    ///
    /// # Errors
    ///
    /// Returns an error if status retrieval fails due to connection issues
    /// or hardware not being initialized.
    fn get_status(&self) -> Result<HardwareStatus>;
    
    /// Calibrate hardware (if supported)
    ///
    /// # Errors
    ///
    /// Returns an error if calibration fails due to hardware issues,
    /// calibration not being supported, or connection problems.
    fn calibrate(&mut self) -> Result<CalibrationResult>;
    
    /// Close connection to hardware
    ///
    /// # Errors
    ///
    /// Returns an error if the connection cannot be properly closed.
    fn close(&mut self) -> Result<()>;
    
    /// Get provider name
    fn provider_name(&self) -> &'static str;
}

/// Hardware runtime status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareStatus {
    /// Whether hardware is online
    pub online: bool,
    
    /// Utilization percentage
    pub utilization: f64,
    
    /// Queue length for jobs
    pub queue_length: usize,
    
    /// Estimated wait time in seconds
    pub estimated_wait_sec: u64,
    
    /// System uptime in seconds
    pub uptime_sec: u64,
    
    /// Current error rates for gates
    pub error_rates: HashMap<String, f64>,
    
    /// Current coherence times in microseconds
    pub coherence_times: HashMap<String, f64>,
    
    /// Number of available qubits
    pub available_qubits: usize,
    
    /// Provider-specific status details
    pub details: HashMap<String, String>,
}

/// Result of hardware calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// Whether calibration was successful
    pub success: bool,
    
    /// Calibration metrics
    pub metrics: HashMap<String, f64>,
    
    /// Updated error rates
    pub updated_error_rates: HashMap<String, f64>,
    
    /// Updated coherence times
    pub updated_coherence_times: HashMap<String, f64>,
    
    /// Time taken for calibration in milliseconds
    pub calibration_time_ms: u64,
    
    /// Calibration messages
    pub messages: Vec<String>,
}

/// Factory for creating provider-specific hardware executors
pub struct HardwareExecutorFactory;

impl HardwareExecutorFactory {
    /// Create a hardware executor for the specified provider
    ///
    /// # Errors
    ///
    /// Returns an error if the requested provider is not supported.
    pub fn create(provider: &str) -> Result<Box<dyn HardwareExecutor>> {
        match provider {
            "ibmq" => Ok(Box::new(IBMQExecutor::new())),
            "ionq" => Ok(Box::new(IonQExecutor::new())),
            "rigetti" => Ok(Box::new(RigettiExecutor::new())),
            "simulator" => Ok(Box::new(SimulatedExecutor::new())),
            _ => Err(HardwareInterfaceError::UnsupportedOperation(
                format!("Unsupported hardware provider: {provider}")
            ).into())
        }
    }
}

/// `IBM` Quantum executor implementation
pub struct IBMQExecutor {
    /// Whether the executor is initialized
    initialized: bool,
    
    /// Connection configuration
    config: Option<HardwareConnectionConfig>,
    
    /// Discovered capabilities
    capabilities: Option<HardwareCapability>,
    
    /// API token for authentication
    api_token: Option<String>,
}

impl IBMQExecutor {
    /// Create a new `IBM` Quantum executor
    #[must_use]
    pub fn new() -> Self {
        Self {
            initialized: false,
            config: None,
            capabilities: None,
            api_token: None,
        }
    }
}

impl Default for IBMQExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareExecutor for IBMQExecutor {
    fn initialize(&mut self, config: &HardwareConnectionConfig) -> Result<()> {
        if let AuthMethod::ApiKey(key) = &config.auth {
            self.api_token = Some(key.clone());
            self.config = Some(config.clone());
            self.initialized = true;
            Ok(())
        } else {
            Err(HardwareInterfaceError::AuthenticationFailed(
                "IBM Quantum requires API key authentication".to_string()
            ).into())
        }
    }
    
    fn is_initialized(&self) -> bool {
        self.initialized
    }
    
    fn discover_capabilities(&mut self) -> Result<HardwareCapability> {
        if !self.initialized {
            return Err(HardwareInterfaceError::NotInitialized(
                "Executor not initialized".to_string()
            ).into());
        }
        
        // In a real implementation, this would query the IBM Q API
        // For now, we'll create a simulated capability set
        
        let mut capability = HardwareCapability {
            id: "ibmq_simulated".to_string(),
            architecture: HardwareArchitecture::Superconducting,
            instruction_sets: vec![
                InstructionSetType::GateLevel,
                InstructionSetType::CircuitLevel,
            ],
            qubit_count: 27,
            supported_gates: vec![
                "x".to_string(), "y".to_string(), "z".to_string(), 
                "h".to_string(), "cx".to_string(), "rz".to_string(),
                "sx".to_string(), "rx".to_string(), "ry".to_string(),
            ],
            error_rates: HashMap::new(),
            coherence_times: HashMap::new(),
            connectivity: HashMap::new(),
            additional_params: HashMap::new(),
        };
        
        // Add error rates
        capability.error_rates.insert("single_qubit".to_string(), 0.001);
        capability.error_rates.insert("two_qubit".to_string(), 0.01);
        capability.error_rates.insert("readout".to_string(), 0.02);
        
        // Add coherence times
        capability.coherence_times.insert("t1".to_string(), 100.0);
        capability.coherence_times.insert("t2".to_string(), 50.0);
        
        // Add connectivity (limited to first 5 qubits for simplicity)
        capability.connectivity.insert(0, vec![1, 2]);
        capability.connectivity.insert(1, vec![0, 3]);
        capability.connectivity.insert(2, vec![0, 4]);
        capability.connectivity.insert(3, vec![1]);
        capability.connectivity.insert(4, vec![2]);
        
        self.capabilities = Some(capability.clone());
        Ok(capability)
    }
    
    fn execute_circuit(&mut self, _circuit: &[String], qubit_mapping: &[usize]) -> Result<Vec<u8>> {
        if !self.initialized {
            return Err(HardwareInterfaceError::NotInitialized(
                "IBMQ executor not initialized".to_string(),
            ).into());
        }

        // In a real implementation, this would translate the circuit to IBMQ's format,
        // submit it to the quantum hardware, wait for results, and translate
        // them back into our expected format.
        
        // Just simulate for now
        let mut rng = rand::thread_rng();
        let result_size = qubit_mapping.len().max(1);
        let mut results = Vec::with_capacity(result_size);
        
        for _ in 0..result_size {
            let random_byte = rng.gen::<u8>();
            results.push(random_byte);
        }
        
        Ok(results)
    }
    
    fn get_status(&self) -> Result<HardwareStatus> {
        if !self.initialized {
            return Err(HardwareInterfaceError::NotInitialized(
                "Executor not initialized".to_string()
            ).into());
        }
        
        // In a real implementation, this would query the IBM Q API
        // For now, return simulated status
        Ok(HardwareStatus {
            online: true,
            utilization: 0.75,
            queue_length: 5,
            estimated_wait_sec: 300,
            uptime_sec: 86400,
            error_rates: {
                let mut rates = HashMap::new();
                rates.insert("single_qubit".to_string(), 0.001);
                rates.insert("two_qubit".to_string(), 0.01);
                rates.insert("readout".to_string(), 0.02);
                rates
            },
            coherence_times: {
                let mut times = HashMap::new();
                times.insert("t1".to_string(), 100.0);
                times.insert("t2".to_string(), 50.0);
                times
            },
            available_qubits: 27,
            details: {
                let mut details = HashMap::new();
                details.insert("backend".to_string(), "ibmq_fake_provider".to_string());
                details.insert("version".to_string(), "1.0.0".to_string());
                details
            },
        })
    }
    
    fn calibrate(&mut self) -> Result<CalibrationResult> {
        if !self.initialized {
            return Err(HardwareInterfaceError::NotInitialized(
                "Executor not initialized".to_string()
            ).into());
        }
        
        // In a real implementation, this might trigger calibration on IBM Q
        // For now, return simulated calibration results
        
        Err(HardwareInterfaceError::UnsupportedOperation(
            "Calibration not supported on remote IBM Q devices".to_string()
        ).into())
    }
    
    fn close(&mut self) -> Result<()> {
        self.initialized = false;
        self.api_token = None;
        Ok(())
    }
    
    fn provider_name(&self) -> &'static str {
        "ibmq"
    }
}

/// `IonQ` executor implementation
pub struct IonQExecutor {
    /// Whether the executor is initialized
    initialized: bool,
    
    /// Connection configuration
    config: Option<HardwareConnectionConfig>,
    
    /// Discovered capabilities
    capabilities: Option<HardwareCapability>,
}

impl IonQExecutor {
    /// Create a new `IonQ` executor
    #[must_use]
    pub fn new() -> Self {
        Self {
            initialized: false,
            config: None,
            capabilities: None,
        }
    }
}

impl Default for IonQExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareExecutor for IonQExecutor {
    fn initialize(&mut self, config: &HardwareConnectionConfig) -> Result<()> {
        self.config = Some(config.clone());
        self.initialized = true;
        Ok(())
    }
    
    fn is_initialized(&self) -> bool {
        self.initialized
    }
    
    fn discover_capabilities(&mut self) -> Result<HardwareCapability> {
        if !self.initialized {
            return Err(HardwareInterfaceError::NotInitialized(
                "Executor not initialized".to_string()
            ).into());
        }
        
        // In a real implementation, this would query the IonQ API
        // For now, we'll create a simulated capability set
        
        let mut capability = HardwareCapability {
            id: "ionq_simulated".to_string(),
            architecture: HardwareArchitecture::TrappedIon,
            instruction_sets: vec![
                InstructionSetType::GateLevel,
                InstructionSetType::CircuitLevel,
            ],
            qubit_count: 11,
            supported_gates: vec![
                "x".to_string(), "y".to_string(), "z".to_string(), 
                "h".to_string(), "cnot".to_string(), "s".to_string(),
                "t".to_string(), "rx".to_string(), "ry".to_string(), "rz".to_string(),
            ],
            error_rates: HashMap::new(),
            coherence_times: HashMap::new(),
            connectivity: HashMap::new(),
            additional_params: HashMap::new(),
        };
        
        // Add error rates
        capability.error_rates.insert("single_qubit".to_string(), 0.0005);
        capability.error_rates.insert("two_qubit".to_string(), 0.01);
        capability.error_rates.insert("readout".to_string(), 0.01);
        
        // Add coherence times
        capability.coherence_times.insert("t1".to_string(), 10000.0); // Very long coherence in trapped ion
        capability.coherence_times.insert("t2".to_string(), 1000.0);
        
        // Add all-to-all connectivity
        for i in 0..11 {
            let mut connections = Vec::new();
            for j in 0..11 {
                if i != j {
                    connections.push(j);
                }
            }
            capability.connectivity.insert(i, connections);
        }
        
        self.capabilities = Some(capability.clone());
        Ok(capability)
    }
    
    fn execute_circuit(&mut self, _circuit: &[String], qubit_mapping: &[usize]) -> Result<Vec<u8>> {
        if !self.is_initialized() {
            return Err(HardwareInterfaceError::NotInitialized(
                "Execute called before initialization".to_string()
            ).into());
        }
        
        // Simulate execution delay
        std::thread::sleep(Duration::from_millis(800));
        
        // In a real implementation, this would submit the circuit to IonQ
        // For simulation, return deterministic but "random" results
        let result_size = qubit_mapping.len().max(1);
        let mut result = Vec::with_capacity(result_size);
        
        // For trapped ions, create some results with patterns
        for (i, &q) in qubit_mapping.iter().enumerate() {
            // Cast safely using try_from with fallback for large numbers
            let remainder = (q * 3 + i * 7) % 256; 
            let byte = u8::try_from(remainder).unwrap_or_default();
            result.push(byte);
        }
        
        Ok(result)
    }
    
    fn get_status(&self) -> Result<HardwareStatus> {
        if !self.initialized {
            return Err(HardwareInterfaceError::NotInitialized(
                "Executor not initialized".to_string()
            ).into());
        }
        
        // In a real implementation, this would query the IonQ API
        // For now, return simulated status
        Ok(HardwareStatus {
            online: true,
            utilization: 0.85,
            queue_length: 8,
            estimated_wait_sec: 600,
            uptime_sec: 172_800,
            error_rates: {
                let mut rates = HashMap::new();
                rates.insert("single_qubit".to_string(), 0.0005);
                rates.insert("two_qubit".to_string(), 0.01);
                rates.insert("readout".to_string(), 0.01);
                rates
            },
            coherence_times: {
                let mut times = HashMap::new();
                times.insert("t1".to_string(), 10000.0);
                times.insert("t2".to_string(), 1000.0);
                times
            },
            available_qubits: 11,
            details: {
                let mut details = HashMap::new();
                details.insert("backend".to_string(), "ionq_harmony".to_string());
                details.insert("version".to_string(), "2.0.0".to_string());
                details
            },
        })
    }
    
    fn calibrate(&mut self) -> Result<CalibrationResult> {
        if !self.initialized {
            return Err(HardwareInterfaceError::NotInitialized(
                "Executor not initialized".to_string()
            ).into());
        }
        
        // In a real implementation, this might trigger calibration
        // For now, return simulated calibration results
        Ok(CalibrationResult {
            success: true,
            metrics: {
                let mut metrics = HashMap::new();
                metrics.insert("gate_fidelity".to_string(), 0.9995);
                metrics.insert("readout_fidelity".to_string(), 0.995);
                metrics
            },
            updated_error_rates: {
                let mut rates = HashMap::new();
                rates.insert("single_qubit".to_string(), 0.0004);
                rates.insert("two_qubit".to_string(), 0.009);
                rates.insert("readout".to_string(), 0.008);
                rates
            },
            updated_coherence_times: {
                let mut times = HashMap::new();
                times.insert("t1".to_string(), 10500.0);
                times.insert("t2".to_string(), 1100.0);
                times
            },
            calibration_time_ms: 15000,
            messages: vec![
                "Calibration completed successfully".to_string(),
                "Single-qubit gates improved by 20%".to_string(),
            ],
        })
    }
    
    fn close(&mut self) -> Result<()> {
        self.initialized = false;
        Ok(())
    }
    
    fn provider_name(&self) -> &'static str {
        "ionq"
    }
}

/// Rigetti executor implementation (simplified for this example)
pub struct RigettiExecutor {
    initialized: bool,
}

impl RigettiExecutor {
    /// Create a new Rigetti executor
    #[must_use]
    pub fn new() -> Self {
        Self {
            initialized: false,
        }
    }
}

impl Default for RigettiExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareExecutor for RigettiExecutor {
    fn initialize(&mut self, _config: &HardwareConnectionConfig) -> Result<()> {
        self.initialized = true;
        Ok(())
    }
    
    fn is_initialized(&self) -> bool {
        self.initialized
    }
    
    fn discover_capabilities(&mut self) -> Result<HardwareCapability> {
        // Simplified stub implementation
        let mut capability = HardwareCapability {
            id: "rigetti_simulated".to_string(),
            architecture: HardwareArchitecture::Superconducting,
            instruction_sets: vec![InstructionSetType::GateLevel],
            qubit_count: 40,
            supported_gates: vec!["x".to_string(), "h".to_string(), "cz".to_string()],
            error_rates: HashMap::new(),
            coherence_times: HashMap::new(),
            connectivity: HashMap::new(),
            additional_params: HashMap::new(),
        };
        
        capability.error_rates.insert("single_qubit".to_string(), 0.002);
        capability.error_rates.insert("two_qubit".to_string(), 0.02);
        
        Ok(capability)
    }
    
    fn execute_circuit(&mut self, _circuit: &[String], qubit_mapping: &[usize]) -> Result<Vec<u8>> {
        if !self.initialized {
            return Err(HardwareInterfaceError::NotInitialized(
                "Executor not initialized".to_string()
            ).into());
        }
        
        // In a real implementation, this would actually execute the circuit
        // For this simulation, just return a placeholder result
        
        // In practice, the result would depend on the circuit and qubit mapping
        // For the simulation, we'll return a deterministic result based on qubits
        let result_size = qubit_mapping.len().max(1);
        let mut result = Vec::with_capacity(result_size);
        
        for &q in qubit_mapping {
            // Cast safely using try_from with fallback for large numbers
            let remainder = q % 256;
            let value = u8::try_from(remainder).unwrap_or_default();
            result.push(value);
        }
        
        Ok(result)
    }
    
    fn get_status(&self) -> Result<HardwareStatus> {
        // Simplified status
        Ok(HardwareStatus {
            online: true,
            utilization: 0.5,
            queue_length: 2,
            estimated_wait_sec: 120,
            uptime_sec: 43200,
            error_rates: HashMap::new(),
            coherence_times: HashMap::new(),
            available_qubits: 40,
            details: HashMap::new(),
        })
    }
    
    fn calibrate(&mut self) -> Result<CalibrationResult> {
        if !self.initialized {
            return Err(HardwareInterfaceError::NotInitialized("Executor must be initialized before calibration".to_string()).into());
        }
        
        // Simulate calibration delay
        std::thread::sleep(Duration::from_millis(1500));
        
        // Create simulated calibration metrics
        let mut metrics = HashMap::new();
        metrics.insert("readout_fidelity".to_string(), 0.972);
        metrics.insert("two_qubit_fidelity".to_string(), 0.945);
        metrics.insert("single_qubit_fidelity".to_string(), 0.989);
        
        // Create simulated error rates
        let mut error_rates = HashMap::new();
        error_rates.insert("single_qubit".to_string(), 0.011);
        error_rates.insert("two_qubit".to_string(), 0.055);
        error_rates.insert("measurement".to_string(), 0.028);
        
        // Create simulated coherence times
        let mut coherence_times = HashMap::new();
        coherence_times.insert("t1".to_string(), 45.0);
        coherence_times.insert("t2".to_string(), 32.0);
        
        Ok(CalibrationResult {
            success: true,
            metrics,
            updated_error_rates: error_rates,
            updated_coherence_times: coherence_times,
            calibration_time_ms: 1500,
            messages: vec!["Calibration completed successfully".to_string()],
        })
    }
    
    fn close(&mut self) -> Result<()> {
        self.initialized = false;
        Ok(())
    }
    
    fn provider_name(&self) -> &'static str {
        "rigetti"
    }
}

/// Simulated quantum hardware executor
///
/// This executor provides a realistic simulation of quantum hardware behavior
/// without requiring access to actual quantum hardware. It's useful for:
///
/// - Development and testing without hardware access
/// - Simulating different noise models and error rates
/// - Educational purposes to understand quantum hardware behavior
/// - Fallback when real hardware is unavailable
///
/// The simulator mimics real hardware constraints like decoherence, gate errors,
/// and measurement noise to provide a realistic testing environment.
pub struct SimulatedExecutor {
    /// Whether the executor is initialized
    initialized: bool,
    
    /// Simulated hardware capabilities
    capabilities: Option<HardwareCapability>,
    
    /// Current noise model settings
    noise_model: NoiseModel,
    
    /// Simulated qubits count
    qubit_count: usize,
    
    /// Simulated device ID
    device_id: String,
    
    /// Simulated queue length
    queue_length: usize,
    
    /// Last calibration timestamp
    last_calibration: u64,
}

/// Noise model for quantum simulation
#[derive(Debug, Clone)]
pub struct NoiseModel {
    /// Gate error rates by gate type
    pub gate_errors: HashMap<String, f64>,
    
    /// Measurement error rate
    pub measurement_error: f64,
    
    /// T1 relaxation time (amplitude damping) in microseconds
    pub t1_us: f64,
    
    /// T2 dephasing time in microseconds
    pub t2_us: f64,
    
    /// Crosstalk error between adjacent qubits
    pub crosstalk: f64,
}

impl Default for NoiseModel {
    fn default() -> Self {
        let mut gate_errors = HashMap::new();
        gate_errors.insert("x".to_string(), 0.001);
        gate_errors.insert("h".to_string(), 0.002);
        gate_errors.insert("cx".to_string(), 0.01);
        gate_errors.insert("cz".to_string(), 0.008);
        gate_errors.insert("rz".to_string(), 0.001);
        
        Self {
            gate_errors,
            measurement_error: 0.02,
            t1_us: 50.0,
            t2_us: 30.0,
            crosstalk: 0.002,
        }
    }
}

impl SimulatedExecutor {
    /// Create a new simulated executor with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            initialized: false,
            capabilities: None,
            noise_model: NoiseModel::default(),
            qubit_count: 20, // Default to 20 qubits
            device_id: format!("simulated_device_{}", random::<u16>()),
            queue_length: 0,
            last_calibration: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
    
    /// Create a new simulated executor with custom noise model and qubit count
    #[must_use]
    pub fn new_with_noise(noise_model: NoiseModel, qubit_count: usize) -> Self {
        Self {
            initialized: false,
            capabilities: None,
            noise_model,
            qubit_count,
            device_id: format!("simulated_device_{}", random::<u16>()),
            queue_length: 0,
            last_calibration: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
    
    /// Simulate a quantum circuit execution with noise
    ///
    /// # Arguments
    ///
    /// * `circuit` - Quantum circuit as string instructions
    /// * `shots` - Number of circuit repetitions
    ///
    /// # Returns
    ///
    /// Simulated measurement results
    fn simulate_circuit_with_noise(&self, circuit: &[String], shots: usize) -> Vec<u8> {
        let mut rng = rand::thread_rng();
        
        // For each shot, simulate measurement results
        let mut results = Vec::with_capacity(shots);
        
        for _ in 0..shots {
            // In a real implementation, this would use a quantum circuit simulator
            // with the configured noise model
            
            // Generate a random measurement outcome based on the circuit depth
            // and our noise model
            #[allow(clippy::cast_precision_loss)]
            let circuit_depth = circuit.len() as f64;
            
            // Calculate effective error based on noise model and circuit
            let effective_error = self.noise_model.measurement_error + 
                (circuit_depth * self.noise_model.gate_errors.get("cx").unwrap_or(&0.01));
            
            // Simulate decoherence based on T1/T2 times
            let circuit_time_us = circuit_depth * 0.1; // assume 0.1us per gate
            let decoherence_factor = (-circuit_time_us / self.noise_model.t1_us).exp();
            
            // Probabilistic output based on error rates
            if rand::Rng::gen_bool(&mut rng, effective_error) || 
               rand::Rng::gen_bool(&mut rng, 1.0 - decoherence_factor) {
                // Error case - randomize result
                results.push(rand::Rng::gen_range(&mut rng, 0..=255));
            } else {
                // Deterministic result based on circuit
                // For simplicity, hash the circuit strings to get a consistent result
                let hash: u8 = circuit.iter()
                    .fold(0u8, |acc, s| acc.wrapping_add(s.bytes().next().unwrap_or(0)));
                results.push(hash);
            }
        }
        
        results
    }
}

impl Default for SimulatedExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareExecutor for SimulatedExecutor {
    fn initialize(&mut self, _config: &HardwareConnectionConfig) -> Result<()> {
        // Simulate hardware initialization delay
        std::thread::sleep(Duration::from_millis(500));
        
        self.initialized = true;
        Ok(())
    }
    
    fn is_initialized(&self) -> bool {
        self.initialized
    }
    
    fn discover_capabilities(&mut self) -> Result<HardwareCapability> {
        if !self.initialized {
            return Err(HardwareInterfaceError::NotInitialized(
                "Executor not initialized".to_string()
            ).into());
        }
        
        // Create a simulated capability profile
        let capability = HardwareCapability {
            id: self.device_id.clone(),
            architecture: HardwareArchitecture::Simulated,
            instruction_sets: vec![
                InstructionSetType::GateLevel,
                InstructionSetType::CircuitLevel,
            ],
            qubit_count: self.qubit_count,
            supported_gates: vec![
                "x".to_string(), "y".to_string(), "z".to_string(),
                "h".to_string(), "cx".to_string(), "cz".to_string(),
                "rx".to_string(), "ry".to_string(), "rz".to_string(),
            ],
            error_rates: {
                let mut rates = HashMap::new();
                rates.insert("single_qubit".to_string(), 0.001);
                rates.insert("two_qubit".to_string(), 0.01);
                rates.insert("measurement".to_string(), self.noise_model.measurement_error);
                rates
            },
            coherence_times: {
                let mut times = HashMap::new();
                times.insert("t1".to_string(), self.noise_model.t1_us);
                times.insert("t2".to_string(), self.noise_model.t2_us);
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
            additional_params: {
                let mut params = HashMap::new();
                params.insert("simulator_type".to_string(), "full_density_matrix".to_string());
                params.insert("device_id".to_string(), self.device_id.clone());
                params.insert("max_circuit_depth".to_string(), "100".to_string());
                params.insert("mid_circuit_measurement".to_string(), "true".to_string());
                params.insert("dynamic_decoupling".to_string(), "true".to_string());
                params.insert("error_mitigation".to_string(), "basic".to_string());
                params
            },
        };
        
        // Store and return capabilities
        self.capabilities = Some(capability.clone());
        
        Ok(capability)
    }
    
    fn execute_circuit(&mut self, circuit: &[String], qubit_mapping: &[usize]) -> Result<Vec<u8>> {
        if !self.initialized {
            return Err(HardwareInterfaceError::NotInitialized(
                "Simulated executor not initialized".to_string(),
            ).into());
        }

        // Random delay to simulate queue waiting and execution time
        let execution_time = rand::thread_rng().gen_range(100..1000);
        std::thread::sleep(Duration::from_millis(execution_time));
        
        // Increment queue length (max out at 10 for realism)
        self.queue_length = (self.queue_length + 1).min(10);
        
        // Validate qubit mapping
        if !qubit_mapping.is_empty() && qubit_mapping.iter().any(|&q| q >= self.qubit_count) {
            return Err(HardwareInterfaceError::InvalidConfiguration(
                format!("Qubit mapping exceeds available qubits ({})", self.qubit_count)
            ).into());
        }
        
        // Check if circuit is valid
        if circuit.is_empty() {
            return Err(HardwareInterfaceError::InvalidConfiguration(
                "Empty circuit provided".to_string()
            ).into());
        }
        
        // Simulate circuit execution with noise
        let results = self.simulate_circuit_with_noise(circuit, 8);
        
        // Decrement queue after execution
        if self.queue_length > 0 {
            self.queue_length -= 1;
        }
        
        Ok(results)
    }
    
    fn get_status(&self) -> Result<HardwareStatus> {
        if !self.initialized {
            return Err(HardwareInterfaceError::NotInitialized(
                "Executor not initialized".to_string()
            ).into());
        }
        
        // Calculate time since last calibration
        let now = util::timestamp_now();
        let calibration_age_sec = (now - self.last_calibration) / 1000;
        
        // Construct errors and coherence maps
        let mut error_rates = HashMap::new();
        for (gate, &base_error) in &self.noise_model.gate_errors {
            // Errors get slightly worse as time passes since calibration
            // Convert using f64::from for safer precision
            let time_factor = f64::from(
                u32::try_from(calibration_age_sec)
                    .unwrap_or(u32::MAX)
            ) / 3600.0;
            
            let bounded_time_factor = time_factor.min(2.0);
            error_rates.insert(gate.clone(), base_error * bounded_time_factor);
        }
        
        let mut coherence_times = HashMap::new();
        coherence_times.insert("T1".to_string(), self.noise_model.t1_us);
        coherence_times.insert("T2".to_string(), self.noise_model.t2_us);
        
        // Add simulated utilization based on queue length
        // Convert using f64::from for safer precision
        let utilization = f64::from(
            u32::try_from(self.queue_length)
                .unwrap_or(u32::MAX)
        ) / 10.0;
        
        let bounded_utilization = utilization.min(1.0);
        
        // Include some realistic device information
        let mut details = HashMap::new();
        details.insert("simulator_type".to_string(), "full_density_matrix".to_string());
        details.insert("device_id".to_string(), self.device_id.clone());
        
        // Format the calibration age with proper conversion
        details.insert("calibration_age_hrs".to_string(), 
            format!("{:.1}", f64::from(
                u32::try_from(calibration_age_sec)
                    .unwrap_or(u32::MAX)
            ) / 3600.0));
        
        // Return simulated status
        Ok(HardwareStatus {
            online: true,
            utilization: bounded_utilization,
            queue_length: self.queue_length,
            estimated_wait_sec: u64::from(
                u32::try_from(self.queue_length)
                    .unwrap_or(u32::MAX)
            ) * 60, // 1 minute per job in queue
            uptime_sec: 86400, // Simulated 24 hours uptime
            error_rates,
            coherence_times,
            available_qubits: self.qubit_count,
            details,
        })
    }
    
    fn calibrate(&mut self) -> Result<CalibrationResult> {
        if !self.initialized {
            return Err(HardwareInterfaceError::NotInitialized(
                "Executor not initialized".to_string()
            ).into());
        }
        
        // Simulate calibration delay
        std::thread::sleep(Duration::from_millis(3000));
        
        // Update last calibration timestamp
        self.last_calibration = util::timestamp_now();
        
        // Prepare calibration metrics
        let mut metrics = HashMap::new();
        metrics.insert("readout_fidelity".to_string(), 1.0 - self.noise_model.measurement_error);
        metrics.insert("average_gate_fidelity".to_string(), 0.996);
        metrics.insert("entanglement_fidelity".to_string(), 0.92);
        
        // Slightly improved error rates after calibration
        let mut updated_error_rates = HashMap::new();
        for (gate, &error) in &self.noise_model.gate_errors {
            updated_error_rates.insert(gate.clone(), error * 0.9); // 10% improvement
        }
        
        // T1/T2 times remain the same (hardware characteristic)
        let mut updated_coherence_times = HashMap::new();
        updated_coherence_times.insert("T1".to_string(), self.noise_model.t1_us);
        updated_coherence_times.insert("T2".to_string(), self.noise_model.t2_us);
        
        // Generate some realistic calibration messages
        let messages = vec![
            "Optimizing single-qubit gates".to_string(),
            "Calibrating readout discrimination".to_string(),
            "Measuring T1/T2 coherence times".to_string(),
            "Optimizing two-qubit gates".to_string(),
            "Compensating for crosstalk".to_string(),
            "Calibration completed successfully".to_string(),
        ];
        
        // Return calibration result
        Ok(CalibrationResult {
            success: true,
            metrics,
            updated_error_rates,
            updated_coherence_times,
            calibration_time_ms: 3000,
            messages,
        })
    }
    
    fn close(&mut self) -> Result<()> {
        // No real connection to close, just clear the state
        self.initialized = false;
        Ok(())
    }
    
    fn provider_name(&self) -> &'static str {
        "simulator"
    }
}

/// Configuration for fallback behavior
#[derive(Debug, Clone)]
pub struct FallbackConfig {
    /// Whether to enable automatic fallback to simulation
    pub enable_fallback: bool,
    
    /// Maximum number of failures before triggering fallback
    pub max_failures: usize,
    
    /// Initial retry delay in milliseconds
    pub initial_retry_delay_ms: u64,
    
    /// Maximum retry delay in milliseconds
    pub max_retry_delay_ms: u64,
    
    /// Backoff multiplier for retries (exponential backoff)
    pub backoff_multiplier: f64,
    
    /// Whether to log fallback events
    pub log_fallback_events: bool,
}

impl Default for FallbackConfig {
    fn default() -> Self {
        Self {
            enable_fallback: true,
            max_failures: 3,
            initial_retry_delay_ms: 1000,
            max_retry_delay_ms: 30000,
            backoff_multiplier: 2.0,
            log_fallback_events: true,
        }
    }
}

/// The `HardwareRegistry` provides a centralized system for managing connections
/// to quantum hardware providers and simulators.
pub struct HardwareRegistry {
    /// Connected hardware executors
    executors: HashMap<String, Box<dyn HardwareExecutor>>,
    
    /// Default executor ID
    default_executor: Option<String>,
    
    /// Fallback configuration
    fallback_config: FallbackConfig,
    
    /// Number of connection failures for each provider
    connection_failures: HashMap<String, usize>,
    
    /// Last retry delay for each provider (for exponential backoff)
    retry_delays: HashMap<String, u64>,
    
    /// Last error message for diagnostics
    last_error: Option<String>,
    
    /// Last successful fallback (for metrics)
    last_fallback: Option<(String, String, u64)>, // (provider, fallback_id, timestamp)
}

impl HardwareRegistry {
    /// Create a new `HardwareRegistry` with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            executors: HashMap::new(),
            default_executor: None,
            fallback_config: FallbackConfig::default(),
            connection_failures: HashMap::new(),
            retry_delays: HashMap::new(),
            last_error: None,
            last_fallback: None,
        }
    }
    
    /// Create a new `HardwareRegistry` with the specified fallback settings
    #[must_use]
    pub fn new_with_fallback_config(fallback_config: FallbackConfig) -> Self {
        Self {
            executors: HashMap::new(),
            default_executor: None,
            fallback_config,
            connection_failures: HashMap::new(),
            retry_delays: HashMap::new(),
            last_error: None,
            last_fallback: None,
        }
    }
    
    /// Configure the fallback behavior
    ///
    /// # Arguments
    ///
    /// * `config` - The new fallback configuration
    pub fn set_fallback_config(&mut self, config: FallbackConfig) {
        self.fallback_config = config;
    }
    
    /// Get the current fallback configuration
    ///
    /// # Returns
    ///
    /// A reference to the current fallback configuration
    #[must_use]
    pub fn fallback_config(&self) -> &FallbackConfig {
        &self.fallback_config
    }
    
    /// Check if automatic fallback is enabled
    ///
    /// # Returns
    ///
    /// true if fallback is enabled, false otherwise
    #[must_use]
    pub const fn is_fallback_enabled(&self) -> bool {
        self.fallback_config.enable_fallback
    }
    
    /// Enable or disable automatic fallback
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable fallback
    pub const fn set_fallback_enabled(&mut self, enable: bool) {
        self.fallback_config.enable_fallback = enable;
    }
    
    /// Get information about the last fallback that occurred
    ///
    /// An Option containing a tuple with (provider, `fallback_id`, timestamp) if a fallback occurred
    #[must_use]
    pub const fn last_fallback(&self) -> Option<&(String, String, u64)> {
        self.last_fallback.as_ref()
    }
    
    /// Reset failure count for a provider
    ///
    /// # Arguments
    ///
    /// * `provider` - The provider to reset failures for
    pub fn reset_failures(&mut self, provider: &str) {
        self.connection_failures.remove(provider);
        self.retry_delays.remove(provider);
    }
    
    /// Reset all failure counts
    pub fn reset_all_failures(&mut self) {
        self.connection_failures.clear();
        self.retry_delays.clear();
    }
    
    /// Connect to quantum hardware
    ///
    /// Attempts to connect to the specified hardware provider with the given
    /// configuration. If connection fails and fallback is enabled, will automatically
    /// create a simulator instead. Uses exponential backoff for retries.
    ///
    /// # Arguments
    ///
    /// * `provider` - Hardware provider name (e.g., "ibmq", "ionq", "rigetti")
    /// * `config` - Connection configuration
    ///
    /// # Returns
    ///
    /// A Result containing the ID of the connected hardware
    ///
    /// # Errors
    ///
    /// Returns an error if connection fails and fallback is disabled
    pub fn connect_hardware(&mut self, 
                          provider: &str, 
                          config: &HardwareConnectionConfig) -> Result<String> {
        // Attempt to connect to the specified hardware
        match self.try_connect_to_hardware(provider, config) {
            Ok(id) => {
                // Successfully connected
                // Reset failure count for this provider
                self.reset_failures(provider);
                
                // If this is our first executor, make it the default
                if self.default_executor.is_none() {
                    self.default_executor = Some(id.clone());
                }
                
                Ok(id)
            },
            Err(error) => {
                // Connection failed
                let error_msg = format!("Failed to connect to {provider}: {error}");
                self.last_error = Some(error_msg.clone());
                
                // Increment failure count for this provider
                let failures = self.connection_failures
                    .entry(provider.to_string())
                    .or_insert(0);
                *failures += 1;
                
                // Calculate retry delay with exponential backoff
                let retry_delay = self.retry_delays
                    .entry(provider.to_string())
                    .or_insert(self.fallback_config.initial_retry_delay_ms);
                
                // Update retry delay for next attempt (exponential backoff)
                *retry_delay = calculate_next_delay(
                    *retry_delay,
                    self.fallback_config.backoff_multiplier,
                    self.fallback_config.max_retry_delay_ms
                );
                
                // Check if fallback is enabled and needed
                if self.fallback_config.enable_fallback && *failures >= self.fallback_config.max_failures {
                    // Create simulator as fallback
                    self.create_simulator_fallback(provider, error_msg.as_str())
                } else {
                    // No fallback, return the error
                    Err(error)
                }
            }
        }
    }
    
    /// Try to connect to a hardware provider
    fn try_connect_to_hardware(&mut self, provider: &str, config: &HardwareConnectionConfig) -> Result<String> {
        let mut executor = HardwareExecutorFactory::create(provider)?;
        
        // Generate unique ID for this connection
        let connection_id = format!("{}_{}", provider, util::generate_id(provider));
        
        // Try to initialize
        if let Err(error) = executor.initialize(config) {
            // Record failure for this provider
            let failures = self.connection_failures.entry(provider.to_string())
                .or_insert(0);
            *failures += 1;
            
            // Record detailed error message
            let error_msg = format!("Failed to connect to {provider}: {error}");
            self.last_error = Some(error_msg.clone());
            
            // Return the error
            return Err(HardwareInterfaceError::ConnectionFailed(error_msg).into());
        }
        
        // Store the executor
        self.executors.insert(connection_id.clone(), executor);
        
        // Reset failure count on success
        self.connection_failures.remove(provider);
        self.retry_delays.remove(provider);
        
        // If no default is set, make this the default
        if self.default_executor.is_none() {
            self.default_executor = Some(connection_id.clone());
        }
        
        Ok(connection_id)
    }
    
    /// Create a simulator as a fallback
    ///
    /// Used when real hardware connection fails and fallback is enabled
    fn create_simulator_fallback(&mut self, original_provider: &str, error_msg: &str) -> Result<String> {
        // Log the fallback if enabled
        if self.fallback_config.log_fallback_events {
            println!("WARNING: Falling back to simulator due to hardware connection failure: {error_msg}");
        }
        
        // Create a simulator
        let mut executor: Box<dyn HardwareExecutor> = Box::new(SimulatedExecutor::new_with_noise(
            NoiseModel::default(),
            32, // Default to 32 qubits for fallback
        ));
        
        // Initialize the simulator with a default config
        executor.initialize(&HardwareConnectionConfig::default())?;
        
        // Create an ID that indicates this is a fallback
        let id = format!("fallback_sim_for_{original_provider}");
        
        // Add to executors map
        self.executors.insert(id.clone(), executor);
        
        // Store fallback information
        self.last_fallback = Some((
            original_provider.to_string(),
            id.clone(),
            util::timestamp_now(),
        ));
        
        // If we have no default, make this the default
        if self.default_executor.is_none() {
            self.default_executor = Some(id.clone());
        }
        
        Ok(id)
    }
    
    /// Get a hardware executor by ID
    ///
    /// # Arguments
    ///
    /// * `id` - The hardware executor ID
    ///
    /// # Returns
    ///
    /// A Result containing a mutable reference to the executor
    ///
    /// # Errors
    ///
    /// Returns an error if the ID doesn't exist
    pub fn get_executor(&mut self, id: &str) -> Result<&mut Box<dyn HardwareExecutor>> {
        self.executors.get_mut(id).ok_or_else(|| {
            let error_msg = format!("Hardware executor not found: {id}");
            QHEPError::HardwareDetectionFailed(error_msg).into()
        })
    }
    
    /// Get the default executor
    ///
    /// This function returns the default executor if one is set.
    /// If no default executor is set, it returns an error.
    ///
    /// # Errors
    ///
    /// Returns error if default executor is not set or cannot be found
    ///
    /// # Panics
    ///
    /// Will panic if default executor is set but not found in registry
    pub fn get_default_executor(&mut self) -> Result<&mut Box<dyn HardwareExecutor>> {
        if let Some(default_id) = &self.default_executor {
            self.executors.get_mut(default_id).ok_or_else(|| {
                let msg = format!("Default executor '{default_id}' not found");
                anyhow::anyhow!(msg)
            })
        } else {
            Err(anyhow::anyhow!("No default executor set"))
        }
    }
    
    /// Set the default hardware executor
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the executor to set as default
    ///
    /// # Returns
    ///
    /// A Result indicating success or failure
    ///
    /// # Errors
    ///
    /// Returns an error if the ID doesn't exist
    pub fn set_default_executor(&mut self, id: &str) -> Result<()> {
        if self.executors.contains_key(id) {
            self.default_executor = Some(id.to_string());
            Ok(())
        } else {
            Err(QHEPError::HardwareDetectionFailed(
                format!("Cannot set default executor - ID not found: {id}")
            ).into())
        }
    }
    
    /// Disconnect hardware by ID
    ///
    /// # Errors
    ///
    /// Returns an error if the hardware with the specified ID is not found or
    /// disconnection fails.
    pub fn disconnect_hardware(&mut self, id: &str) -> Result<()> {
        if let Some(executor) = self.executors.get_mut(id) {
            // Close the connection
            let provider = executor.provider_name();
            
            match executor.close() {
                Ok(()) => {
                    log_info(&format!("Successfully disconnected from {provider} hardware"));
                },
                Err(e) => {
                    log_error(&format!("Error disconnecting from {provider} hardware: {e}"));
                    // Continue with removal anyway
                }
            }
            
            // If it was the default executor, clear that reference
            if let Some(default_id) = &self.default_executor {
                if default_id == id {
                    self.default_executor = None;
                }
            }
            
            // Remove from executors map
            self.executors.remove(id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Hardware not found with ID: {id}"))
        }
    }
    
    /// Disconnect all hardware
    ///
    /// Closes all hardware connections and clears the registry
    ///
    /// # Returns
    ///
    /// A Result indicating success or failure
    ///
    /// # Errors
    ///
    /// Returns an error if any connection fails to close properly
    pub fn disconnect_all(&mut self) -> Result<()> {
        // Collect all IDs to avoid borrowing issues
        let executor_ids: Vec<String> = self.executors.keys().cloned().collect();
        
        // Close each executor
        for id in executor_ids {
            self.disconnect_hardware(&id)?;
        }
        
        // Clear all remaining state
        self.executors.clear();
        self.default_executor = None;
        self.connection_failures.clear();
        
        Ok(())
    }
    
    /// Get a list of connected hardware IDs
    ///
    /// # Returns
    ///
    /// A vector of connected hardware executor IDs
    #[must_use]
    pub fn get_connected_hardware(&self) -> Vec<String> {
        self.executors.keys().cloned().collect()
    }
    
    /// Check if any hardware is connected
    ///
    /// # Returns
    ///
    /// true if at least one hardware executor is connected, false otherwise
    #[must_use]
    pub fn has_connected_hardware(&self) -> bool {
        !self.executors.is_empty()
    }
    
    /// Get the last error message
    ///
    /// Useful for diagnostics when fallback occurs
    ///
    /// # Returns
    ///
    /// An Option containing the last error message, if any
    #[must_use]
    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    /// Connect to quantum hardware with automatic retry
    ///
    /// Attempts to connect to the specified hardware provider with retries
    /// using exponential backoff until successful or max retries reached.
    ///
    /// # Arguments
    ///
    /// * `provider` - Hardware provider name (e.g., "ibmq", "ionq", "rigetti")
    /// * `config` - Connection configuration
    /// * `max_retries` - Maximum number of retry attempts
    ///
    /// # Returns
    ///
    /// A Result containing the ID of the connected hardware
    ///
    /// # Errors
    ///
    /// Returns an error if connection fails after all retries
    /// 
    /// # Panics
    /// 
    /// This method will panic if the `last_error` unwrap fails, which should not 
    /// happen in practice as we always set `last_error` when an error occurs
    pub fn connect_hardware_with_retry(&mut self, 
                                     provider: &str, 
                                     config: &HardwareConnectionConfig,
                                     max_retries: usize) -> Result<String> {
        let mut attempts = 0;
        let mut last_error = None;
        
        while attempts < max_retries + 1 {
            match self.try_connect_to_hardware(provider, config) {
                Ok(id) => {
                    // Reset failure tracking on success
                    self.reset_failures(provider);
                    
                    // If this is our first executor, make it the default
                    if self.default_executor.is_none() {
                        self.default_executor = Some(id.clone());
                    }
                    
                    return Ok(id);
                },
                Err(error) => {
                    last_error = Some(error);
                    
                    // Calculate delay with exponential backoff
                    let retry_delay = self.retry_delays
                        .entry(provider.to_string())
                        .or_insert(self.fallback_config.initial_retry_delay_ms);
                    
                    // Only sleep if not the last attempt
                    if attempts < max_retries {
                        if self.fallback_config.log_fallback_events {
                            println!("Connection attempt {current} failed, retrying in {delay} ms", 
                                    current = attempts + 1, 
                                    delay = retry_delay);
                        }
                        
                        // Sleep with exponential backoff
                        std::thread::sleep(Duration::from_millis(*retry_delay));
                        
                        // Update delay for next attempt
                        *retry_delay = calculate_next_delay(
                            *retry_delay,
                            self.fallback_config.backoff_multiplier,
                            self.fallback_config.max_retry_delay_ms
                        );
                    }
                    
                    attempts += 1;
                }
            }
        }
        
        // All attempts failed, try fallback if enabled
        if self.fallback_config.enable_fallback {
            let total_attempts = max_retries + 1;
            let error_msg = format!("Failed to connect to {provider} after {total_attempts} attempts");
            self.last_error = Some(error_msg.clone());
            
            self.create_simulator_fallback(provider, &error_msg)
        } else {
            // No fallback, return the last error
            Err(last_error.unwrap())
        }
    }

    // Add a method to log errors that updates the last_error field
    #[allow(dead_code)]
    fn log_registry_error(&mut self, message: &str) {
        log_error(message);
        self.last_error = Some(message.to_string());
    }
}

/// Log an informational message
fn log_info(message: &str) {
    // In a real implementation, this would log to a file or monitoring system
    // For now, just print to stdout
    println!("INFO: {message}");
}

/// Log an error message
fn log_error(message: &str) {
    // In a real implementation, this would log to a file or monitoring system
    // For now, just print to stderr
    eprintln!("ERROR: {message}");
}

impl fmt::Debug for HardwareRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HardwareRegistry")
            .field("executor_count", &self.executors.len())
            .field("default_executor", &self.default_executor)
            .field("fallback_config", &self.fallback_config)
            .field("connection_failures", &self.connection_failures)
            .field("retry_delays", &self.retry_delays)
            .field("last_error", &self.last_error)
            .field("last_fallback", &self.last_fallback)
            .finish()
    }
}

impl Default for HardwareRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate next backoff delay using exponential backoff with jitter
fn calculate_next_delay(current_delay: u64, multiplier: f64, max_delay: u64) -> u64 {
    if current_delay == 0 {
        return 100; // Start with a small delay
    }
    
    // Calculate next delay (with multiplier)
    let next_delay_f64 = f64::from(u32::try_from(current_delay).unwrap_or(100)) * multiplier;
    
    // Add some jitter (±10%)
    let jitter = next_delay_f64 * 0.1 * (rand::random::<f64>() * 2.0 - 1.0);
    let next_delay_f64 = next_delay_f64 + jitter;
    
    // Convert to u64 safely and cap at maximum
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let next_delay = u64::from(next_delay_f64.round() as u32);
    
    std::cmp::min(next_delay, max_delay)
}

// Test functions for hardware integration
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hardware_executor_factory() {
        let executor = HardwareExecutorFactory::create("simulator").unwrap();
        assert_eq!(executor.provider_name(), "simulator");
        
        let executor = HardwareExecutorFactory::create("ibmq").unwrap();
        assert_eq!(executor.provider_name(), "ibmq");
    }
    
    #[test]
    fn test_hardware_registry() {
        // Create a registry with custom fallback config
        let fallback_config = FallbackConfig {
            enable_fallback: true,
            max_failures: 2,
            initial_retry_delay_ms: 100,
            max_retry_delay_ms: 1000,
            backoff_multiplier: 1.5,
            log_fallback_events: false,
        };
        
        let mut registry = HardwareRegistry::new_with_fallback_config(fallback_config);
        
        // Verify fallback configuration
        assert!(registry.is_fallback_enabled());
        assert_eq!(registry.fallback_config().max_failures, 2);
        
        // Connect to a simulator
        let config = HardwareConnectionConfig::default();
        let id = registry.connect_hardware("simulator", &config).unwrap();
        
        // Check that we have a connection
        assert!(registry.has_connected_hardware());
        assert_eq!(registry.get_connected_hardware().len(), 1);
        
        // Test getting the executor
        let executor = registry.get_executor(&id).unwrap();
        assert_eq!(executor.provider_name(), "simulator");
        
        // Test default executor
        let default = registry.get_default_executor().unwrap();
        assert_eq!(default.provider_name(), "simulator");
        
        // Test reset failures
        registry.reset_failures("ibmq");
        
        // Test resetting all failures
        registry.reset_all_failures();
        
        // Disconnect
        registry.disconnect_hardware(&id).unwrap();
        assert!(!registry.has_connected_hardware());
    }
    
    #[test]
    fn test_simulated_executor() {
        let mut executor = SimulatedExecutor::new();
        let config = HardwareConnectionConfig::default();
        
        // Initialize
        executor.initialize(&config).unwrap();
        assert!(executor.is_initialized());
        
        // Discover capabilities
        let capabilities = executor.discover_capabilities().unwrap();
        assert_eq!(capabilities.architecture, HardwareArchitecture::Simulated);
        assert_eq!(capabilities.qubit_count, 20);
        
        // Execute a simple circuit
        let circuit = vec![
            "h 0".to_string(),
            "cx 0 1".to_string(),
        ];
        let mapping = vec![0, 1];
        let result = executor.execute_circuit(&circuit, &mapping).unwrap();
        assert_eq!(result.len(), 8);
        
        // Check status
        let status = executor.get_status().unwrap();
        assert!(status.online);
        assert_eq!(status.available_qubits, 20);
        
        // Close
        executor.close().unwrap();
        assert!(!executor.is_initialized());
    }
} 