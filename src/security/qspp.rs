// Quantum Side-channel Protection Protocol (QSPP)
//
// This protocol protects quantum systems against side-channel attacks
// that attempt to extract information through timing, power analysis,
// photon emissions, and other non-direct means.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use thiserror::Error;
use rand::{Rng, thread_rng};

/// Errors specific to QSPP
#[derive(Error, Debug)]
pub enum QSPPError {
    /// Timing attack detected
    #[error("Timing attack detected: {0}")]
    TimingAttack(String),
    
    /// Power analysis attack detected
    #[error("Power analysis attack detected: {0}")]
    PowerAnalysis(String),
    
    /// Photon emission attack detected
    #[error("Photon emission attack detected: {0}")]
    PhotonEmission(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    /// Hardware interface error
    #[error("Hardware interface error: {0}")]
    HardwareError(String),
    
    /// General protection error
    #[error("Protection error: {0}")]
    ProtectionError(String),
    
    /// Entropy source error
    #[error("Entropy source error: {0}")]
    EntropyError(String),
    
    /// Entropy service error
    #[error("Entropy service error: {0}")]
    EntropyServiceError(String),
    
    /// Protocol error
    #[error("Protocol error: {0}")]
    ProtocolError(String),
    
    /// Randomness error
    #[error("Randomness error: {0}")]
    RandomnessError(String),
}

/// Types of side-channel attacks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum SideChannelAttackType {
    /// Timing analysis attacks
    Timing,
    
    /// Power consumption analysis attacks
    Power,
    
    /// Electromagnetic radiation analysis
    EM,
    
    /// Photon emission analysis
    Photon,
    
    /// Thermal analysis
    Thermal,
    
    /// Acoustic analysis
    Acoustic,
    
    /// Cache-based analysis
    Cache,
    
    /// Quantum measurement discrimination
    MeasurementDiscrimination,
    
    /// Entanglement leakage analysis
    EntanglementLeakage,
    
    /// Quantum state tomography attack
    StateTomography,
    
    /// Quantum predictable entropy attack
    PredictableEntropy,
}

/// Countermeasure technique for side-channel protection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CountermeasureTechnique {
    /// Constant-time implementation
    ConstantTime,
    
    /// Power balancing
    PowerBalancing,
    
    /// Random delays
    RandomDelays,
    
    /// Masking/blinding operations
    Masking,
    
    /// Noise generation
    NoiseGeneration,
    
    /// Shielding (physical)
    Shielding,
    
    /// Operation shuffling
    Shuffling,
    
    /// Quantum noise injection
    QuantumNoiseInjection,
    
    /// Superposition state masking
    SuperpositionMasking,
    
    /// Entanglement purification
    EntanglementPurification,
    
    /// Random basis switching
    RandomBasisSwitching,
    
    /// Decoy states
    DecoyStates,
    
    /// Quantum-resilient entropy mixing
    QuantumResilientEntropy,
}

/// Types of entropy sources for quantum-resilient randomness
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum EntropySource {
    /// Quantum random number generator
    QRNG,
    
    /// System entropy
    System,
    
    /// Hardware thermal noise
    Thermal,
    
    /// Radioactive decay
    Radioactive,
    
    /// Atmospheric noise
    Atmospheric,
    
    /// Quantum vacuum fluctuations
    QuantumVacuum,
}

/// Quantum-resilient entropy service for QSPP
#[derive(Debug)]
pub struct QuantumResilientEntropy {
    /// Active entropy sources
    sources: Vec<EntropySource>,
    
    /// Last collected entropy
    last_entropy: Vec<u8>,
    
    /// Entropy mixing rounds
    mixing_rounds: u8,
    
    /// Health check status for each source
    source_health: HashMap<EntropySource, bool>,
}

impl Default for QuantumResilientEntropy {
    fn default() -> Self {
        let sources = vec![EntropySource::System];
        
        let mut source_health = HashMap::new();
        source_health.insert(EntropySource::System, true);
        
        Self {
            sources,
            last_entropy: Vec::new(),
            mixing_rounds: 3,
            source_health,
        }
    }
}

impl QuantumResilientEntropy {
    /// Create a new entropy service with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Get active entropy sources
    #[must_use]
    pub fn sources(&self) -> &[EntropySource] {
        &self.sources
    }
    
    /// Add an entropy source
    pub fn add_source(&mut self, source: EntropySource) {
        if !self.sources.contains(&source) {
            self.sources.push(source);
            self.source_health.insert(source, true);
        }
    }
    
    /// Remove an entropy source
    pub fn remove_source(&mut self, source: EntropySource) {
        if let Some(pos) = self.sources.iter().position(|&s| s == source) {
            self.sources.remove(pos);
            self.source_health.remove(&source);
        }
    }
    
    /// Get entropy bytes
    ///
    /// # Arguments
    ///
    /// * `bytes` - Number of entropy bytes to get
    ///
    /// # Returns
    ///
    /// Random entropy bytes
    ///
    /// # Errors
    ///
    /// * `QSPPError::EntropyServiceError` - If the entropy service fails
    /// * `QSPPError::ProtocolError` - If protection is not active
    pub fn get_entropy(&mut self, bytes: usize) -> Result<Vec<u8>, QSPPError> {
        if self.sources.is_empty() {
            return Err(QSPPError::EntropyError("No entropy sources available".to_string()));
        }
        
        let mut result = Vec::with_capacity(bytes);
        let mut rng = thread_rng();
        
        // Collect entropy from all sources
        for _ in 0..bytes {
            let mut byte: u8 = 0;
            
            // Mix entropy from each source
            for (i, source) in self.sources.iter().enumerate() {
                if !self.source_health.get(source).unwrap_or(&false) {
                    continue;
                }
                
                let source_byte = match source {
                    EntropySource::QRNG => {
                        // In a real implementation, this would call a quantum RNG
                        rng.gen::<u8>()
                    }
                    EntropySource::System => {
                        // System entropy
                        rng.gen::<u8>()
                    }
                    EntropySource::Thermal => {
                        // Simulated thermal noise
                        rng.gen::<u8>()
                    }
                    EntropySource::Radioactive => {
                        // Simulated radioactive decay
                        rng.gen::<u8>()
                    }
                    EntropySource::Atmospheric => {
                        // Simulated atmospheric noise
                        rng.gen::<u8>()
                    }
                    EntropySource::QuantumVacuum => {
                        // Simulated quantum vacuum fluctuations
                        rng.gen::<u8>()
                    }
                };
                
                // XOR mixing
                if i == 0 {
                    byte = source_byte;
                } else {
                    byte ^= source_byte;
                }
            }
            
            result.push(byte);
        }
        
        // Apply multiple rounds of mixing for quantum resilience
        for _ in 0..self.mixing_rounds {
            result = Self::mix_entropy(&result);
        }
        
        self.last_entropy.clone_from(&result);
        Ok(result)
    }
    
    /// Mix entropy using a simple algorithm
    fn mix_entropy(data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }
        
        let mut result = Vec::with_capacity(data.len());
        let mut last = data[data.len() - 1];
        
        for &b in data {
            // Non-linear mixing
            let mixed = b.rotate_left(3) ^ last.rotate_right(2) ^ (b & last);
            result.push(mixed);
            last = mixed;
        }
        
        result
    }
    
    /// Check the health of entropy sources
    pub fn check_sources(&mut self) -> Vec<EntropySource> {
        let mut unhealthy = Vec::new();
        
        for &source in &self.sources {
            // In a real implementation, this would perform proper health checks
            let healthy = thread_rng().gen_bool(0.95); // 95% chance of being healthy
            
            self.source_health.insert(source, healthy);
            
            if !healthy {
                unhealthy.push(source);
            }
        }
        
        unhealthy
    }
}

/// Configuration for QSPP
#[derive(Debug, Clone)]
pub struct QSPPConfig {
    /// Enabled countermeasures for each attack type
    pub countermeasures: HashMap<SideChannelAttackType, Vec<CountermeasureTechnique>>,
    
    /// Protection level (higher = more protection but potential performance impact)
    pub protection_level: u8,
    
    /// Whether to detect attacks only or also mitigate them
    pub detection_only: bool,
    
    /// Attack detection sensitivity (0-100, higher = more sensitive)
    pub detection_sensitivity: u8,
    
    /// Maximum allowed timing variation (nanoseconds)
    pub max_timing_variation_ns: u64,
    
    /// Perform random operation shuffling
    pub enable_operation_shuffling: bool,
    
    /// Random delay amount (maximum milliseconds)
    pub max_random_delay_ms: u64,
    
    /// Enable quantum-resilient entropy service
    pub enable_entropy_service: bool,
    
    /// Minimum entropy sources required
    pub min_entropy_sources: u8,
    
    /// Entropy mixing rounds for additional security
    pub entropy_mixing_rounds: u8,
}

impl Default for QSPPConfig {
    fn default() -> Self {
        let mut countermeasures = HashMap::new();
        
        // Default countermeasures for each attack type
        countermeasures.insert(SideChannelAttackType::Timing, 
            vec![CountermeasureTechnique::ConstantTime, CountermeasureTechnique::RandomDelays]);
        countermeasures.insert(SideChannelAttackType::Power, 
            vec![CountermeasureTechnique::PowerBalancing, CountermeasureTechnique::Masking]);
        countermeasures.insert(SideChannelAttackType::EM, 
            vec![CountermeasureTechnique::Shielding]);
        countermeasures.insert(SideChannelAttackType::Photon, 
            vec![CountermeasureTechnique::Shielding]);
        countermeasures.insert(SideChannelAttackType::Thermal, 
            vec![CountermeasureTechnique::NoiseGeneration]);
        countermeasures.insert(SideChannelAttackType::Acoustic, 
            vec![CountermeasureTechnique::NoiseGeneration]);
        countermeasures.insert(SideChannelAttackType::Cache, 
            vec![CountermeasureTechnique::Shuffling]);
        // Add new quantum-specific countermeasures
        countermeasures.insert(SideChannelAttackType::MeasurementDiscrimination,
            vec![CountermeasureTechnique::RandomBasisSwitching, CountermeasureTechnique::DecoyStates]);
        countermeasures.insert(SideChannelAttackType::EntanglementLeakage,
            vec![CountermeasureTechnique::EntanglementPurification, CountermeasureTechnique::SuperpositionMasking]);
        countermeasures.insert(SideChannelAttackType::StateTomography,
            vec![CountermeasureTechnique::QuantumNoiseInjection, CountermeasureTechnique::RandomBasisSwitching]);
        countermeasures.insert(SideChannelAttackType::PredictableEntropy,
            vec![CountermeasureTechnique::QuantumResilientEntropy]);
            
        Self {
            countermeasures,
            protection_level: 2,  // Medium protection level
            detection_only: false,
            detection_sensitivity: 70,
            max_timing_variation_ns: 500,
            enable_operation_shuffling: true,
            max_random_delay_ms: 5,
            enable_entropy_service: true,
            min_entropy_sources: 1,
            entropy_mixing_rounds: 3,
        }
    }
}

/// Protection profile for a specific component
#[derive(Debug, Clone)]
pub struct ProtectionProfile {
    /// Component identifier
    pub component_id: String,
    
    /// Component type (e.g., "`quantum_gate`", "memory", "measurement")
    pub component_type: String,
    
    /// Specific attack vectors for this component
    pub attack_vectors: Vec<SideChannelAttackType>,
    
    /// Required countermeasures for this component
    pub required_countermeasures: Vec<CountermeasureTechnique>,
    
    /// Risk level (1-5, 5 being highest)
    pub risk_level: u8,
}

/// Detection event when a side-channel attack is detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionEvent {
    /// Time of the detection
    pub timestamp: u64,
    
    /// Type of attack detected
    pub attack_type: SideChannelAttackType,
    
    /// Confidence level of detection (0-100)
    pub confidence: u8,
    
    /// Source component
    pub component: String,
    
    /// Raw detection data
    pub detection_data: HashMap<String, f64>,
}

/// QSPP implementation
pub struct QSPP {
    /// Configuration
    config: QSPPConfig,
    
    /// Protected components
    protected_components: HashMap<String, ProtectionProfile>,
    
    /// Detection history
    detection_history: Vec<DetectionEvent>,
    
    /// Current protection state
    is_protection_active: bool,
    
    /// Hardware monitoring state
    #[allow(dead_code)]
    monitoring_active: bool,
    
    /// Last timing reference
    last_timing_reference: Instant,
    
    /// Randomization seed
    #[allow(dead_code)]
    randomization_seed: u64,
    
    /// Quantum-resilient entropy service
    entropy_service: Option<QuantumResilientEntropy>,
}

impl Default for QSPP {
    fn default() -> Self {
        Self::new()
    }
}

impl QSPP {
    /// Create a new QSPP instance with default configuration
    #[must_use]
    pub fn new() -> Self {
        let config = QSPPConfig::default();
        let entropy_service = if config.enable_entropy_service {
            Some(QuantumResilientEntropy::new())
        } else {
            None
        };
        
        Self {
            config,
            protected_components: HashMap::new(),
            detection_history: Vec::new(),
            is_protection_active: false,
            monitoring_active: false,
            last_timing_reference: Instant::now(),
            randomization_seed: thread_rng().gen(),
            entropy_service,
        }
    }
    
    /// Create a new QSPP instance with custom configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration to use
    ///
    /// # Returns
    ///
    /// A new QSPP instance
    #[must_use]
    pub fn with_config(config: QSPPConfig) -> Self {
        let entropy_service = if config.enable_entropy_service {
            let mut service = QuantumResilientEntropy::new();
            service.mixing_rounds = config.entropy_mixing_rounds;
            Some(service)
        } else {
            None
        };
        
        Self {
            config,
            protected_components: HashMap::new(),
            detection_history: Vec::new(),
            is_protection_active: false,
            monitoring_active: false,
            last_timing_reference: Instant::now(),
            randomization_seed: thread_rng().gen(),
            entropy_service,
        }
    }
    
    /// Get the current configuration
    #[must_use]
    pub fn config(&self) -> &QSPPConfig {
        &self.config
    }
    
    /// Modify the configuration
    pub fn config_mut(&mut self) -> &mut QSPPConfig {
        &mut self.config
    }
    
    /// Register a component for protection
    pub fn register_component(&mut self, profile: ProtectionProfile) {
        self.protected_components.insert(profile.component_id.clone(), profile);
    }
    
    /// Unregister a component
    pub fn unregister_component(&mut self, component_id: &str) -> Option<ProtectionProfile> {
        self.protected_components.remove(component_id)
    }
    
    /// Start protection
    ///
    /// # Returns
    ///
    /// Success or failure
    ///
    /// # Errors
    ///
    /// * `QSPPError::ProtocolError` - If protection is already active
    /// * `QSPPError::InitializationError` - If initialization fails
    pub fn start_protection(&mut self) -> Result<(), QSPPError> {
        if self.is_protection_active {
            return Ok(());
        }
        
        // Initialize protection mechanisms
        self.initialize_countermeasures()?;
        
        self.is_protection_active = true;
        self.last_timing_reference = Instant::now();
        
        Ok(())
    }
    
    /// Stop protection
    pub fn stop_protection(&mut self) {
        self.is_protection_active = false;
    }
    
    /// Get access to the entropy service
    pub fn entropy_service(&mut self) -> Option<&mut QuantumResilientEntropy> {
        self.entropy_service.as_mut()
    }
    
    /// Generate quantum-resilient random bytes
    ///
    /// # Arguments
    ///
    /// * `count` - Number of bytes to generate
    ///
    /// # Returns
    ///
    /// Random secure bytes
    ///
    /// # Errors
    ///
    /// * `QSPPError::ProtocolError` - If protection is not active
    /// * `QSPPError::RandomnessError` - If generation fails
    pub fn generate_secure_random_bytes(&mut self, count: usize) -> Result<Vec<u8>, QSPPError> {
        if let Some(entropy_service) = &mut self.entropy_service {
            entropy_service.get_entropy(count)
        } else {
            Err(QSPPError::EntropyServiceError("Entropy service not initialized".to_string()))
        }
    }
    
    /// Initialize countermeasures based on configuration
    fn initialize_countermeasures(&mut self) -> Result<(), QSPPError> {
        // This would contain hardware-specific initialization
        // For simulation purposes, we just return successful initialization
        
        if self.config.protection_level > 5 {
            return Err(QSPPError::ConfigError(
                "Protection level must be between 0 and 5".to_string()));
        }
        
        // Initialize the entropy service if it exists
        if let Some(entropy_service) = &mut self.entropy_service {
            // Set mixing rounds from config
            entropy_service.mixing_rounds = self.config.entropy_mixing_rounds;
            
            // Check entropy sources health
            let unhealthy = entropy_service.check_sources();
            if !unhealthy.is_empty() {
                // Log unhealthy sources but continue
                for source in unhealthy {
                    println!("Warning: Entropy source {source:?} is unhealthy");
                }
            }
            
            // Add more entropy sources based on protection level
            if self.config.protection_level >= 3 {
                entropy_service.add_source(EntropySource::QRNG);
            }
            
            if self.config.protection_level >= 4 {
                entropy_service.add_source(EntropySource::Thermal);
                entropy_service.add_source(EntropySource::Atmospheric);
            }
            
            if self.config.protection_level >= 5 {
                entropy_service.add_source(EntropySource::Radioactive);
                entropy_service.add_source(EntropySource::QuantumVacuum);
            }
            
            // Check if we meet the minimum required sources
            if entropy_service.sources.len() < self.config.min_entropy_sources as usize {
                return Err(QSPPError::EntropyError(format!(
                    "Insufficient entropy sources: have {}, need {}",
                    entropy_service.sources.len(),
                    self.config.min_entropy_sources
                )));
            }
        } else if self.has_countermeasure(
            SideChannelAttackType::PredictableEntropy,
            CountermeasureTechnique::QuantumResilientEntropy
        ) {
            // We need entropy service but it's not enabled
            return Err(QSPPError::ConfigError(
                "Entropy-based countermeasures require enabled entropy service".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Protection against predictable entropy attacks
    ///
    /// # Errors
    ///
    /// Returns a `QSPPError` if entropy generation fails
    pub fn protect_against_entropy_prediction<F, R>(&mut self, operation: F) -> Result<R, QSPPError>
    where
        F: FnOnce(Option<&[u8]>) -> R
    {
        if !self.is_protection_active {
            return Ok(operation(None));
        }
        
        // Check if we have the appropriate countermeasure
        if !self.has_countermeasure(
            SideChannelAttackType::PredictableEntropy, 
            CountermeasureTechnique::QuantumResilientEntropy) {
            return Ok(operation(None));
        }
        
        // Generate entropy to protect the operation
        let entropy = self.generate_secure_random_bytes(32)?;
        
        // Execute the operation with the entropy
        let result = operation(Some(&entropy));
        
        Ok(result)
    }
    
    /// Apply timing attack countermeasures to an operation
    ///
    /// # Errors
    ///
    /// Returns a `QSPPError::TimingAttack` if a timing anomaly is detected
    pub fn protect_timing<F, R>(&self, operation: F) -> Result<R, QSPPError> 
    where 
        F: FnOnce() -> R 
    {
        if !self.is_protection_active {
            return Ok(operation());
        }
        
        let start = Instant::now();
        
        // Apply random delay if configured
        if self.has_countermeasure(SideChannelAttackType::Timing, CountermeasureTechnique::RandomDelays) {
            let delay = thread_rng().gen_range(0..=self.config.max_random_delay_ms);
            std::thread::sleep(Duration::from_millis(delay));
        }
        
        // Execute the operation
        let result = operation();
        
        // If constant time is required, ensure minimum execution time
        if self.has_countermeasure(SideChannelAttackType::Timing, CountermeasureTechnique::ConstantTime) {
            let elapsed = start.elapsed();
            let target_duration = Duration::from_millis(10); // Example constant time
            
            if elapsed < target_duration {
                std::thread::sleep(target_duration - elapsed);
            }
        }
        
        // Check for timing anomalies
        self.detect_timing_anomalies(start.elapsed())?;
        
        Ok(result)
    }
    
    /// Apply power analysis countermeasures to an operation
    ///
    /// # Errors
    ///
    /// Returns a `QSPPError` if power analysis protection fails
    pub fn protect_power<F, R>(&self, operation: F) -> Result<R, QSPPError> 
    where 
        F: FnOnce() -> R 
    {
        if !self.is_protection_active {
            return Ok(operation());
        }
        
        // Here we would integrate with hardware power monitoring
        // For simulation, we just execute the operation
        
        Ok(operation())
    }
    
    /// Apply protection to a quantum operation with specific requirements
    ///
    /// # Arguments
    ///
    /// * `operation_type` - Type of quantum operation
    /// * `component_id` - ID of the component being protected
    /// * `operation` - The operation to protect
    ///
    /// # Errors
    ///
    /// Returns a `QSPPError` if protection fails
    pub fn protect_quantum_operation<F, R>(&self, 
        _operation_type: &str, 
        component_id: &str, 
        operation: F
    ) -> Result<R, QSPPError> 
    where 
        F: FnOnce() -> R 
    {
        if !self.is_protection_active {
            return Ok(operation());
        }
        
        // Check if component is registered
        if !self.protected_components.contains_key(component_id) {
            // Not registered, just run without protection
            return Ok(operation());
        }
        
        // For simulation, we just apply timing protection
        self.protect_timing(operation)
    }
    
    /// Add a detection event
    pub fn add_detection_event(&mut self, event: DetectionEvent) {
        // Store confidence before we move event
        let confidence = event.confidence;
        
        // Depending on config, we might take action here
        self.detection_history.push(event);
        
        // If not in detection-only mode, increase protection
        if !self.config.detection_only && confidence > 80 {
            self.increase_protection_level();
        }
    }
    
    /// Increase protection level in response to detected attack
    fn increase_protection_level(&mut self) {
        if self.config.protection_level < 5 {
            self.config.protection_level += 1;
        }
    }
    
    /// Check if a specific countermeasure is enabled for an attack type
    fn has_countermeasure(&self, attack: SideChannelAttackType, countermeasure: CountermeasureTechnique) -> bool {
        if let Some(measures) = self.config.countermeasures.get(&attack) {
            measures.contains(&countermeasure)
        } else {
            false
        }
    }
    
    /// Detect timing anomalies that could indicate an attack
    fn detect_timing_anomalies(&self, elapsed: Duration) -> Result<(), QSPPError> {
        // This would use statistical analysis in a real implementation
        // For simulation, we just check against maximum allowed variation
        
        let elapsed_ns = match u64::try_from(elapsed.as_nanos()) {
            Ok(value) => value,
            Err(_) => u64::MAX, // If value is too large, use maximum u64
        };
        
        if elapsed_ns > self.config.max_timing_variation_ns && 
           self.config.detection_sensitivity > 50 {
            
            // Only report as an attack if sensitivity is high enough
            if self.config.detection_sensitivity > 90 {
                return Err(QSPPError::TimingAttack(
                    format!("Operation took {elapsed_ns}ns, exceeding the maximum allowed {}", 
                            self.config.max_timing_variation_ns)));
            }
        }
        
        Ok(())
    }
    
    /// Get detection history
    #[must_use]
    pub fn get_detection_history(&self) -> &[DetectionEvent] {
        &self.detection_history
    }
    
    /// Clear detection history
    pub fn clear_detection_history(&mut self) {
        self.detection_history.clear();
    }
    
    /// Get current protection status
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.is_protection_active
    }
    
    /// Protect a quantum operation with context-specific protection
    ///
    /// # Arguments
    ///
    /// * `context` - Protection context
    /// * `operation` - The operation to protect
    ///
    /// # Errors
    ///
    /// Returns a `QSPPError` if protection fails based on the context
    pub fn protect_with_context<F, R>(&self, 
        context: &ProtectionContext,
        operation: F
    ) -> Result<R, QSPPError> 
    where 
        F: FnOnce() -> R 
    {
        if !self.is_protection_active {
            return Ok(operation());
        }
        
        // Check if component is registered
        if !self.protected_components.contains_key(&context.component_id) {
            // Not registered, just run without protection
            return Ok(operation());
        }
        
        // Apply protection based on context
        let result = if context.attack_vectors.contains(&SideChannelAttackType::Timing) {
            self.protect_timing(operation)?
        } else if context.attack_vectors.contains(&SideChannelAttackType::MeasurementDiscrimination) {
            self.protect_quantum_measurement(operation)?
        } else if context.attack_vectors.contains(&SideChannelAttackType::EntanglementLeakage) {
            self.protect_entanglement(operation)?
        } else {
            // Fall back to timing protection as a baseline
            self.protect_timing(operation)?
        };
        
        Ok(result)
    }
    
    /// Apply measurement discrimination countermeasures to quantum measurements
    ///
    /// # Errors
    ///
    /// Returns a `QSPPError` if measurement protection fails
    pub fn protect_quantum_measurement<F, R>(&self, operation: F) -> Result<R, QSPPError> 
    where 
        F: FnOnce() -> R 
    {
        if !self.is_protection_active {
            return Ok(operation());
        }
        
        // Apply quantum-specific countermeasures
        // For simulation, we apply random time delays before measurement 
        // to prevent timing-based measurement discrimination
        
        if self.has_countermeasure(SideChannelAttackType::MeasurementDiscrimination, 
                                    CountermeasureTechnique::RandomBasisSwitching) {
            // Simulate random basis switching with a timing variance
            let delay = thread_rng().gen_range(0..=10);
            std::thread::sleep(Duration::from_millis(delay));
        }
        
        let result = operation();
        
        Ok(result)
    }
    
    /// Apply entanglement leakage countermeasures
    ///
    /// # Errors
    ///
    /// Returns a `QSPPError` if entanglement protection fails
    pub fn protect_entanglement<F, R>(&self, operation: F) -> Result<R, QSPPError> 
    where 
        F: FnOnce() -> R 
    {
        if !self.is_protection_active {
            return Ok(operation());
        }
        
        // For entanglement protection, we would apply quantum-specific
        // techniques like entanglement purification or random state injections
        
        // For now, in simulation, we just execute with some timing variance
        if self.has_countermeasure(SideChannelAttackType::EntanglementLeakage, 
                                  CountermeasureTechnique::EntanglementPurification) {
            // Simulate purification with a time delay
            let delay = thread_rng().gen_range(1..=5);
            std::thread::sleep(Duration::from_millis(delay));
        }
        
        let result = operation();
        
        Ok(result)
    }
    
    /// Helper for comprehensive protection configuration
    pub fn configure_for_protocol<T: QSPPProtectable>(&mut self, protocol: &T) {
        let profile = protocol.protection_profile();
        self.register_component(profile);
        
        // Add recommended countermeasures
        for (attack, measures) in protocol.recommended_countermeasures() {
            if let Some(existing) = self.config.countermeasures.get_mut(&attack) {
                // Add any countermeasures that aren't already configured
                for measure in measures {
                    if !existing.contains(&measure) {
                        existing.push(measure);
                    }
                }
            } else {
                // Add new attack type with countermeasures
                self.config.countermeasures.insert(attack, measures);
            }
        }
    }
}

/// Interface for integrating QSPP with quantum protocols
pub trait QSPPProtectable {
    /// Get the protection profile for this protocol
    fn protection_profile(&self) -> ProtectionProfile;
    
    /// Register with a QSPP instance for protection
    fn register_with_qspp(&self, qspp: &mut QSPP);
    
    /// Check if a specific operation type is protected
    fn is_operation_protected(&self, operation_type: &str) -> bool;
    
    /// Get recommended countermeasures for this protocol
    fn recommended_countermeasures(&self) -> HashMap<SideChannelAttackType, Vec<CountermeasureTechnique>>;
}

/// Protection context containing all necessary information to protect an operation
#[derive(Debug, Clone)]
pub struct ProtectionContext {
    /// The component being protected
    pub component_id: String,
    
    /// The type of operation being performed
    pub operation_type: String,
    
    /// Specific attack vectors to defend against
    pub attack_vectors: Vec<SideChannelAttackType>,
    
    /// Current protection level (1-5)
    pub protection_level: u8,
}

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_qspp_initialization() {
        let qspp = QSPP::new();
        assert!(!qspp.is_active());
        assert_eq!(qspp.config().protection_level, 2);
        
        let custom_config = QSPPConfig {
            protection_level: 4,
            ..Default::default()
        };
        let qspp2 = QSPP::with_config(custom_config);
        assert_eq!(qspp2.config().protection_level, 4);
    }
    
    #[test]
    fn test_component_registration() {
        let mut qspp = QSPP::new();
        
        let profile = ProtectionProfile {
            component_id: "test_component".to_string(),
            component_type: "measurement".to_string(),
            attack_vectors: vec![SideChannelAttackType::Timing, SideChannelAttackType::Power],
            required_countermeasures: vec![CountermeasureTechnique::ConstantTime],
            risk_level: 3,
        };
        
        qspp.register_component(profile.clone());
        assert!(qspp.protected_components.contains_key("test_component"));
        
        let removed = qspp.unregister_component("test_component").unwrap();
        assert_eq!(removed.component_id, "test_component");
        assert!(!qspp.protected_components.contains_key("test_component"));
    }
    
    #[test]
    fn test_protection_activation() {
        let mut qspp = QSPP::new();
        assert!(!qspp.is_active());
        
        qspp.start_protection().expect("Failed to start protection");
        assert!(qspp.is_active());
        
        qspp.stop_protection();
        assert!(!qspp.is_active());
    }
    
    #[test]
    fn test_timing_protection() {
        let mut qspp = QSPP::new();
        qspp.start_protection().expect("Failed to start protection");
        
        // Test with a fast operation
        let result = qspp.protect_timing(|| {
            // Fast operation
            42
        }).expect("Protection failed");
        
        assert_eq!(result, 42);
        
        // Test with slow operation
        let result = qspp.protect_timing(|| {
            // Slow operation
            thread::sleep(Duration::from_millis(20));
            84
        }).expect("Protection failed");
        
        assert_eq!(result, 84);
    }
    
    #[test]
    fn test_detection_events() {
        let mut qspp = QSPP::new();
        
        let event = DetectionEvent {
            timestamp: 12345,
            attack_type: SideChannelAttackType::Timing,
            confidence: 85,
            component: "test_component".to_string(),
            detection_data: HashMap::new(),
        };
        
        qspp.add_detection_event(event);
        assert_eq!(qspp.get_detection_history().len(), 1);
        
        qspp.clear_detection_history();
        assert_eq!(qspp.get_detection_history().len(), 0);
    }
} 