// QSTP Simulation Test Hooks
//
// This file provides testing hooks and simulation capabilities for the
// Quantum State Transfer Protocol (QSTP), allowing controlled testing
// and simulation of various network conditions.
//
// ## Overview
//
// The QSTP simulation framework allows developers to test quantum state transfer
// functionality in controlled conditions without requiring real quantum hardware.
// It provides mechanisms to simulate:
//
// * Variable success rates for transfers
// * Network latency and jitter
// * Quantum decoherence effects
// * Node failures
// * Fidelity loss during transfers
//
// ## Usage Example
//
// ```rust
// use quantum_protocols::core::qstp::{QSTP, QSTPTransport};
// use quantum_protocols::core::quantum_state::QuantumState;
// use quantum_protocols::simulation::qstp_hooks::{
//     QSTPSimulationParams, activate_qstp_simulation, simulate_qstp_transfer
// };
//
// async fn example() {
//     // Create QSTP and a quantum state
//     let mut qstp = QSTP::new("source_node".to_string());
//     let state = QuantumState::new(2);
//     
//     // Configure simulation parameters
//     let params = QSTPSimulationParams {
//         success_rate: 0.8,
//         fidelity_loss: 0.1,
//         latency_ms: 100,
//         ..Default::default()
//     };
//     
//     // Activate simulation
//     activate_qstp_simulation(Some(params));
//     
//     // Perform simulated transfer
//     let result = simulate_qstp_transfer(&mut qstp, &state, "dest_node").await;
// }
// ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::sync::OnceLock;
use std::time::Duration;

use crate::core::quantum_state::QuantumState;
use crate::core::qstp::{QSTP, QSTPError, TransferResult, QSTPTransport};
use crate::error::Result;

/// Global counter for simulated transfers
static TRANSFER_COUNT: AtomicUsize = AtomicUsize::new(0);

/// A collection of simulation parameters that can be adjusted to test
/// different scenarios in QSTP transfers
///
/// These parameters allow fine-grained control over the behavior of simulated
/// quantum state transfers, enabling testing of success/failure scenarios,
/// performance characteristics, and error handling.
#[derive(Debug, Clone)]
pub struct QSTPSimulationParams {
    /// Probability of transfer success (0.0-1.0)
    ///
    /// Controls how often transfers succeed vs. fail. A value of 1.0 means
    /// all transfers succeed, while 0.0 means all transfers fail.
    pub success_rate: f64,
    
    /// Fixed fidelity loss per transfer (0.0-1.0)
    ///
    /// Simulates quantum noise that reduces the fidelity of the state
    /// during transfer. A value of 0.0 means perfect fidelity preservation.
    pub fidelity_loss: f64,
    
    /// Network latency in milliseconds
    ///
    /// Simulates the base network delay for quantum transfers.
    pub latency_ms: u64,
    
    /// Jitter in milliseconds (random variation in latency)
    ///
    /// Simulates variable network conditions with random latency fluctuations.
    pub jitter_ms: u64,
    
    /// Probability of complete decoherence (0.0-1.0)
    ///
    /// Simulates the chance of complete quantum state collapse during transfer.
    /// This is a more severe failure than simple fidelity loss.
    pub decoherence_probability: f64,
    
    /// Fixed duration to simulate for each transfer
    ///
    /// If set, overrides the latency calculation and uses this fixed duration.
    pub fixed_duration_ms: Option<u64>,
    
    /// Simulate node failures during transfer
    ///
    /// If true, the simulation will consider node availability during transfers.
    pub simulate_node_failures: bool,
    
    /// Probability of node failure (0.0-1.0)
    ///
    /// When node failures are simulated, this controls how often nodes are offline.
    pub node_failure_probability: f64,
    
    /// Inject specific error for certain transfers
    ///
    /// Allows deterministic error injection for specific transfer IDs,
    /// which is useful for testing error handling code.
    pub error_injection: HashMap<usize, QSTPError>,
}

impl Default for QSTPSimulationParams {
    fn default() -> Self {
        Self {
            success_rate: 0.95,
            fidelity_loss: 0.05,
            latency_ms: 50,
            jitter_ms: 20,
            decoherence_probability: 0.02,
            fixed_duration_ms: None,
            simulate_node_failures: false,
            node_failure_probability: 0.0,
            error_injection: HashMap::new(),
        }
    }
}

/// Captures events that occur during QSTP simulations for analysis
///
/// Events provide detailed information about each quantum state transfer,
/// allowing developers to debug and analyze the behavior of QSTP under
/// various conditions.
#[derive(Debug, Clone)]
pub struct QSTPSimulationEvent {
    /// Event ID
    pub id: usize,
    
    /// Source node
    pub source: String,
    
    /// Destination node
    pub destination: String,
    
    /// Event type
    pub event_type: QSTPEventType,
    
    /// Original quantum state quality (before transfer)
    pub original_fidelity: f64,
    
    /// Resulting quantum state quality (after transfer)
    pub result_fidelity: Option<f64>,
    
    /// Whether the transfer was successful
    pub success: bool,
    
    /// Error details if the transfer failed
    pub error: Option<String>,
    
    /// Transfer duration in milliseconds
    pub duration_ms: u64,
}

/// Types of events that can occur during QSTP simulation
///
/// These event types categorize different stages and outcomes of
/// the quantum state transfer process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QSTPEventType {
    /// Normal transfer start
    TransferStart,
    
    /// Normal transfer completion
    TransferComplete,
    
    /// Teleportation start
    TeleportStart,
    
    /// Teleportation complete
    TeleportComplete,
    
    /// Error during transfer
    TransferError,
    
    /// Node failure
    NodeFailure,
    
    /// Decoherence event
    Decoherence,
}

/// Global registry for QSTP simulation hooks 
///
/// The registry maintains simulation state, parameters, and event history.
/// It is implemented as a singleton with thread-safe access.
pub struct QSTPSimulationRegistry {
    /// Current simulation parameters
    params: QSTPSimulationParams,
    
    /// Event log
    events: Vec<QSTPSimulationEvent>,
    
    /// Map of node IDs to their online status
    node_status: HashMap<String, bool>,
    
    /// Whether the simulation is active
    active: bool,
}

impl QSTPSimulationRegistry {
    /// Create a new simulation registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            params: QSTPSimulationParams::default(),
            events: Vec::new(),
            node_status: HashMap::new(),
            active: false,
        }
    }
    
    /// Get the current simulation parameters
    ///
    /// # Returns
    ///
    /// A reference to the current simulation parameters
    #[must_use]
    pub const fn params(&self) -> &QSTPSimulationParams {
        &self.params
    }
    
    /// Set new simulation parameters
    ///
    /// # Arguments
    ///
    /// * `params` - The new parameters to use for simulation
    pub fn set_params(&mut self, params: QSTPSimulationParams) {
        self.params = params;
    }
    
    /// Get the event log
    ///
    /// # Returns
    ///
    /// A slice containing all recorded simulation events
    #[must_use]
    pub fn events(&self) -> &[QSTPSimulationEvent] {
        &self.events
    }
    
    /// Record a simulation event
    ///
    /// # Arguments
    ///
    /// * `event` - The event to record
    pub fn record_event(&mut self, event: QSTPSimulationEvent) {
        self.events.push(event);
    }
    
    /// Clear the event log
    ///
    /// Removes all recorded events from the registry
    pub fn clear_events(&mut self) {
        self.events.clear();
    }
    
    /// Activate the simulation
    ///
    /// Enables the simulation hooks
    pub const fn activate(&mut self) {
        self.active = true;
    }
    
    /// Deactivate the simulation
    ///
    /// Disables the simulation hooks, returning to normal QSTP behavior
    pub const fn deactivate(&mut self) {
        self.active = false;
    }
    
    /// Check if the simulation is active
    ///
    /// # Returns
    ///
    /// true if simulation is active, false otherwise
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active
    }
    
    /// Set a node's online status
    ///
    /// # Arguments
    ///
    /// * `node_id` - The ID of the node to set status for
    /// * `online` - Whether the node is online (true) or offline (false)
    pub fn set_node_status(&mut self, node_id: &str, online: bool) {
        self.node_status.insert(node_id.to_string(), online);
    }
    
    /// Get a node's online status
    ///
    /// # Arguments
    ///
    /// * `node_id` - The ID of the node to check
    ///
    /// # Returns
    ///
    /// true if the node is online, false if offline
    /// If the node is not in the registry, returns true (assumes online)
    #[must_use]
    pub fn get_node_status(&self, node_id: &str) -> bool {
        *self.node_status.get(node_id).unwrap_or(&true)
    }
}

impl Default for QSTPSimulationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Global registry for QSTP simulations
static QSTP_REGISTRY: OnceLock<Arc<Mutex<QSTPSimulationRegistry>>> = OnceLock::new();

/// Get a reference to the global QSTP simulation registry
///
/// This provides access to the simulation registry for advanced configuration
/// and event analysis.
///
/// # Returns
///
/// A thread-safe reference to the global registry
///
/// # Examples
///
/// ```
/// use quantum_protocols::simulation::qstp_hooks::get_qstp_registry;
///
/// let registry = get_qstp_registry();
/// let events = {
///     let registry = registry.lock().unwrap();
///     registry.events().to_vec()
/// };
/// ```
#[must_use]
pub fn get_qstp_registry() -> Arc<Mutex<QSTPSimulationRegistry>> {
    QSTP_REGISTRY.get_or_init(|| {
        Arc::new(Mutex::new(QSTPSimulationRegistry::new()))
    }).clone()
}

/// Activate QSTP simulation with the specified parameters
///
/// Enables the simulation hooks with either custom or default parameters.
///
/// # Arguments
///
/// * `params` - Optional simulation parameters. If None, uses defaults.
///
/// # Panics
///
/// Panics if unable to acquire lock on the global registry due to poisoning.
///
/// # Examples
///
/// ```
/// use quantum_protocols::simulation::qstp_hooks::{activate_qstp_simulation, QSTPSimulationParams};
///
/// // Activate with default parameters
/// activate_qstp_simulation(None);
///
/// // Or activate with custom parameters
/// let params = QSTPSimulationParams {
///     success_rate: 0.8,
///     latency_ms: 100,
///     ..Default::default()
/// };
/// activate_qstp_simulation(Some(params));
/// ```
pub fn activate_qstp_simulation(params: Option<QSTPSimulationParams>) {
    let registry = get_qstp_registry();
    let mut registry = registry.lock().unwrap();
    if let Some(p) = params {
        registry.set_params(p);
    }
    registry.activate();
}

/// Deactivate QSTP simulation
///
/// Disables the simulation hooks, returning to normal QSTP behavior.
///
/// # Panics
///
/// Panics if unable to acquire lock on the global registry due to poisoning.
///
/// # Examples
///
/// ```
/// use quantum_protocols::simulation::qstp_hooks::{activate_qstp_simulation, deactivate_qstp_simulation};
///
/// // First activate
/// activate_qstp_simulation(None);
///
/// // Do some testing...
///
/// // Then deactivate when done
/// deactivate_qstp_simulation();
/// ```
pub fn deactivate_qstp_simulation() {
    let registry = get_qstp_registry();
    let mut registry = registry.lock().unwrap();
    registry.deactivate();
}

/// Process and record simulation events
///
/// Helper function to handle recording events in the QSTP registry.
///
/// # Arguments
///
/// * `event` - The event to record
///
/// # Panics
///
/// Panics if unable to acquire lock on the global registry due to poisoning.
fn record_simulation_event(event: QSTPSimulationEvent) {
    let registry = get_qstp_registry();
    let mut registry = registry.lock().unwrap();
    registry.record_event(event);
}

/// Handle a successful transfer in simulation
///
/// Creates a successful transfer result and records the corresponding event.
///
/// # Arguments
///
/// * `transfer_id` - The ID of the transfer
/// * `source_node` - The source node ID
/// * `dest_node_id` - The destination node ID
/// * `state_fidelity` - The original state fidelity
/// * `result_fidelity` - The resulting state fidelity
/// * `duration_ms` - The duration of the transfer in milliseconds
///
/// # Returns
///
/// A successful `TransferResult`
#[must_use]
fn handle_successful_transfer(
    transfer_id: usize,
    source_node: String,
    dest_node_id: &str,
    state_fidelity: f64,
    result_fidelity: f64,
    duration_ms: u64,
) -> TransferResult {
    // Record successful transfer event
    let complete_event = QSTPSimulationEvent {
        id: transfer_id,
        source: source_node,
        destination: dest_node_id.to_string(),
        event_type: QSTPEventType::TransferComplete,
        original_fidelity: state_fidelity,
        result_fidelity: Some(result_fidelity),
        success: true,
        error: None,
        duration_ms,
    };
    
    // Record the completion event
    record_simulation_event(complete_event);
    
    // Return successful result
    TransferResult {
        success: true,
        fidelity: result_fidelity,
        duration_ms,
        state_id: format!("simulated_state_{transfer_id}"),
        error: None,
        transaction_id: format!("sim_transfer_{transfer_id}"),
    }
}

/// Handle a failed transfer in simulation
///
/// Creates an error and records the corresponding event.
///
/// # Arguments
///
/// * `transfer_id` - The ID of the transfer
/// * `source_node` - The source node ID
/// * `dest_node_id` - The destination node ID
/// * `state_fidelity` - The original state fidelity
/// * `error` - The error that occurred
/// * `duration_ms` - The duration of the transfer in milliseconds
///
/// # Returns
///
/// An error result
fn handle_failed_transfer(
    transfer_id: usize,
    source_node: String,
    dest_node_id: &str,
    state_fidelity: f64,
    error: QSTPError,
    duration_ms: u64,
) -> Result<TransferResult> {
    // Record error event
    let error_event = QSTPSimulationEvent {
        id: transfer_id,
        source: source_node,
        destination: dest_node_id.to_string(),
        event_type: QSTPEventType::TransferError,
        original_fidelity: state_fidelity,
        result_fidelity: None,
        success: false,
        error: Some(error.to_string()),
        duration_ms,
    };
    
    // Record the error event
    record_simulation_event(error_event);
    
    // Return error
    Err(error.into())
}

/// Simulates a QSTP transfer with given simulation parameters
///
/// This is the core simulation function that intercepts `QSTP.send_state` calls
/// and applies the configured simulation parameters to produce realistic 
/// simulated outcomes.
///
/// # Arguments
///
/// * `qstp` - The QSTP instance to use for the transfer
/// * `state` - The quantum state to transfer
/// * `dest_node_id` - The destination node ID
///
/// # Returns
///
/// A Result containing the transfer result or an error
///
/// # Errors
///
/// Can return various `QSTPError` types based on simulation parameters:
/// - `NodeNotFound` if the destination node is simulated as offline
/// - `NetworkError` for simulated network failures
/// - `Decoherence` for simulated quantum state collapse
/// - Any error explicitly injected via `error_injection`
///
/// # Panics
///
/// Panics if unable to acquire lock on the global registry due to poisoning.
///
/// # Examples
///
/// ```
/// use quantum_protocols::core::qstp::QSTP;
/// use quantum_protocols::simulation::qstp_hooks::{
///     QSTPSimulationParams, activate_qstp_simulation, simulate_qstp_transfer
/// };
/// use quantum_protocols::test::create_test_quantum_state;
///
/// async fn example() {
///     let mut qstp = QSTP::new("source_node".to_string());
///     let state = create_test_quantum_state(2, true);
///
///     // Activate simulation
///     activate_qstp_simulation(None);
///
///     // Simulate transfer
///     let result = simulate_qstp_transfer(&mut qstp, &state, "destination_node").await;
/// }
/// ```
pub async fn simulate_qstp_transfer(
    qstp: &mut QSTP,
    state: &QuantumState,
    dest_node_id: &str
) -> Result<TransferResult> {
    // Increment the global transfer counter
    let transfer_id = TRANSFER_COUNT.fetch_add(1, Ordering::SeqCst);
    
    // Check if simulation is active and get parameters
    let (is_active, params, node_online) = {
        let registry = get_qstp_registry();
        let registry = registry.lock().unwrap();
        let is_active = registry.is_active();
        let params = if is_active {
            Some(registry.params().clone())
        } else {
            None
        };
        let node_online = if is_active {
            registry.get_node_status(dest_node_id)
        } else {
            true
        };
        (is_active, params, node_online)
    };
    
    // If simulation is not active, perform the real transfer
    if !is_active {
        return match qstp.send_state(state, dest_node_id).await {
            Ok(result) => Ok(result),
            Err(err) => Err(err.into())
        };
    }
    
    let params = params.unwrap();
    let source_node = qstp.node_id().to_string();
    let state_fidelity = state.fidelity();
    
    // Check if there's a specific error to inject for this transfer
    if let Some(error) = params.error_injection.get(&transfer_id).cloned() {
        let event = QSTPSimulationEvent {
            id: transfer_id,
            source: source_node.clone(),
            destination: dest_node_id.to_string(),
            event_type: QSTPEventType::TransferError,
            original_fidelity: state_fidelity,
            result_fidelity: None,
            success: false,
            error: Some(error.to_string()),
            duration_ms: 0,
        };
        
        // Record the event
        record_simulation_event(event);
        return Err(error.into());
    }
    
    // Check if destination node is online
    if !node_online {
        let event = QSTPSimulationEvent {
            id: transfer_id,
            source: source_node.clone(),
            destination: dest_node_id.to_string(),
            event_type: QSTPEventType::NodeFailure,
            original_fidelity: state_fidelity,
            result_fidelity: None,
            success: false,
            error: Some(format!("Node {dest_node_id} is offline")),
            duration_ms: 0,
        };
        
        // Record the event
        record_simulation_event(event);
        return Err(QSTPError::NodeNotFound(dest_node_id.to_string()).into());
    }
    
    // Record transfer start event
    let start_event = QSTPSimulationEvent {
        id: transfer_id,
        source: source_node.clone(),
        destination: dest_node_id.to_string(),
        event_type: QSTPEventType::TransferStart,
        original_fidelity: state_fidelity,
        result_fidelity: None,
        success: true,
        error: None,
        duration_ms: 0,
    };
    record_simulation_event(start_event);
    
    // Simulate transfer delay
    let delay = params.latency_ms;
    tokio::time::sleep(Duration::from_millis(delay)).await;
    
    // Determine if the transfer succeeds based on success rate
    let mut rng = rand::thread_rng();
    let success = rand::Rng::gen_bool(&mut rng, params.success_rate);
    
    // Process result based on success
    if success {
        // Calculate result fidelity with loss
        let result_fidelity = (state_fidelity * (1.0 - params.fidelity_loss)).max(0.0);
        let duration_ms = params.fixed_duration_ms.unwrap_or(params.latency_ms);
        
        Ok(handle_successful_transfer(
            transfer_id, 
            source_node, 
            dest_node_id,
            state_fidelity, 
            result_fidelity, 
            duration_ms
        ))
    } else {
        // Choose a random error type
        let error = if rand::Rng::gen_bool(&mut rng, params.decoherence_probability) {
            QSTPError::Decoherence
        } else {
            QSTPError::NetworkError("Simulated network failure".to_string())
        };
        
        handle_failed_transfer(
            transfer_id,
            source_node,
            dest_node_id,
            state_fidelity,
            error,
            params.latency_ms
        )
    }
}

/// Test functions for QSTP simulation capabilities
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::create_test_quantum_state;
    
    #[tokio::test]
    async fn test_qstp_simulation_hooks() {
        // Create test QSTP instance
        let mut qstp = QSTP::new("test_node".to_string());
        
        // Create test quantum state
        let state = create_test_quantum_state(2, true);
        
        // Activate simulation with custom parameters
        let params = QSTPSimulationParams {
            success_rate: 1.0, // Ensure success
            latency_ms: 10,    // Minimal latency
            fidelity_loss: 0.1, // 10% fidelity loss
            ..Default::default()
        };
        
        activate_qstp_simulation(Some(params));
        
        // Simulate a transfer
        let result = simulate_qstp_transfer(&mut qstp, &state, "dest_node")
            .await
            .expect("Transfer should succeed");
        
        // Verify expected results
        assert!(result.success);
        // Check that fidelity is reduced by 10%
        let expected_fidelity = state.fidelity() * (1.0 - 0.1);
        assert!((result.fidelity - expected_fidelity).abs() < 0.001);
        
        // Check event log
        let registry = get_qstp_registry();
        let registry = registry.lock().unwrap();
        let events = registry.events();
        
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, QSTPEventType::TransferStart);
        assert_eq!(events[1].event_type, QSTPEventType::TransferComplete);
        
        // Deactivate simulation
        drop(registry);
        deactivate_qstp_simulation();
    }
} 