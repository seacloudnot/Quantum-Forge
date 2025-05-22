// Simulation Module
//
// This module provides simulation capabilities for quantum systems.

use crate::network::topology::NetworkTopology;
use crate::error::{Error, NetworkError, Result};
use std::time::{Duration, Instant};

// Export sub-modules
pub mod qstp_hooks;

/// Configuration for quantum network simulation
///
/// This struct contains all parameters needed to configure how the quantum network
/// simulation behaves, including noise levels, timing, and logging preferences.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Duration of each simulation step
    pub step_duration: Duration,
    
    /// Noise level (0.0 - 1.0)
    pub noise_level: f64,
    
    /// Whether to simulate decoherence
    pub simulate_decoherence: bool,
    
    /// Whether to simulate communication delays
    pub simulate_delays: bool,
    
    /// Number of qubits to simulate
    pub qubit_count: usize,
    
    /// Whether to log detailed simulation steps
    pub detailed_logging: bool,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            step_duration: Duration::from_millis(100),
            noise_level: 0.01,
            simulate_decoherence: true,
            simulate_delays: true,
            qubit_count: 8,
            detailed_logging: false,
        }
    }
}

/// Simulator for quantum networks
///
/// This simulator allows for testing quantum protocols without actual quantum hardware
/// by simulating the behavior of quantum networks including entanglement, decoherence,
/// and communication.
///
/// # Example
///
/// ```
/// use quantum_protocols::simulation::{QuantumNetworkSimulator, SimulationConfig};
/// use quantum_protocols::network::topology::NetworkTopology;
///
/// // Create a simulator with default config
/// let mut simulator = QuantumNetworkSimulator::new(SimulationConfig::default());
///
/// // Set up a simple network topology
/// let mut topology = NetworkTopology::new();
/// topology.add_node_by_id("node1".to_string());
/// topology.add_node_by_id("node2".to_string());
/// topology.add_connection("node1", "node2");
///
/// simulator.setup_topology(topology);
/// simulator.start();
///
/// // Run 10 simulation steps
/// for _ in 0..10 {
///     simulator.step();
/// }
///
/// simulator.stop();
/// ```
pub struct QuantumNetworkSimulator {
    /// Configuration for this simulator
    config: SimulationConfig,
    
    /// Network topology being simulated
    topology: NetworkTopology,
    
    /// Current simulation time
    current_time: Instant,
    
    /// Event log
    events: Vec<SimulationEvent>,
    
    /// Whether the simulation is running
    running: bool,
}

/// An event in the simulation
///
/// Events track all activities that happen during simulation, including
/// protocol operations, error conditions, and state changes.
#[derive(Debug, Clone)]
pub struct SimulationEvent {
    /// When the event occurred
    pub time: Instant,
    
    /// Type of event
    pub event_type: String,
    
    /// Source node, if any
    pub source: Option<String>,
    
    /// Destination node, if any
    pub destination: Option<String>,
    
    /// Event details
    pub details: String,
    
    /// Success or failure
    pub success: bool,
}

impl QuantumNetworkSimulator {
    /// Create a new simulator with the specified configuration
    #[must_use]
    pub fn new(config: SimulationConfig) -> Self {
        Self {
            config,
            topology: NetworkTopology::new(),
            current_time: Instant::now(),
            events: Vec::new(),
            running: false,
        }
    }
    
    /// Set up a network topology for simulation
    pub fn setup_topology(&mut self, topology: NetworkTopology) {
        self.topology = topology;
    }
    
    /// Start the simulation
    pub fn start(&mut self) {
        self.running = true;
        self.current_time = Instant::now();
        self.log_event("Simulation started", None, None, "Initialized simulator", true);
    }
    
    /// Run a single simulation step
    ///
    /// Each step advances the simulation time by the configured step duration
    /// and applies environmental effects like decoherence.
    pub fn step(&mut self) {
        if !self.running {
            return;
        }
        
        // Apply decoherence to all nodes
        if self.config.simulate_decoherence {
            self.apply_decoherence();
        }
        
        // Advance simulation time
        self.current_time += self.config.step_duration;
    }
    
    /// Apply decoherence to all quantum states in the network
    fn apply_decoherence(&mut self) {
        let noise = self.config.noise_level;
        
        // Get a mutable copy of all nodes
        let node_ids: Vec<String> = self.topology.nodes().keys().cloned().collect();
        
        for node_id in node_ids {
            if let Some(node) = self.topology.get_node_mut(&node_id) {
                // Apply decoherence to all stored states
                node.apply_decoherence(noise);
                
                // Log decoherence if detailed logging is enabled
                if self.config.detailed_logging {
                    self.log_event(
                        "Decoherence", 
                        Some(node_id.clone()), 
                        None, 
                        &format!("Applied noise factor {noise}"),
                        true
                    );
                }
            }
        }
    }
    
    /// Stop the simulation
    pub fn stop(&mut self) {
        self.running = false;
        self.log_event("Simulation stopped", None, None, "Terminated simulator", true);
    }
    
    /// Log a simulation event
    fn log_event(&mut self, event_type: &str, source: Option<String>, destination: Option<String>, details: &str, success: bool) {
        self.events.push(SimulationEvent {
            time: self.current_time,
            event_type: event_type.to_string(),
            source,
            destination,
            details: details.to_string(),
            success,
        });
    }
    
    /// Add a custom event to the simulation log
    ///
    /// This allows protocols and applications to record important events
    /// that occur during simulation.
    ///
    /// # Arguments
    ///
    /// * `event_type` - The type of event (e.g., "Entanglement", "Key Exchange")
    /// * `source` - Optional source node ID
    /// * `destination` - Optional destination node ID
    /// * `details` - Details about the event
    /// * `success` - Whether the event was successful
    ///
    /// # Example
    ///
    /// ```
    /// # use quantum_protocols::simulation::{QuantumNetworkSimulator, SimulationConfig};
    /// # let mut simulator = QuantumNetworkSimulator::new(SimulationConfig::default());
    /// simulator.add_custom_event(
    ///     "QKD",
    ///     Some("alice".to_string()),
    ///     Some("bob".to_string()),
    ///     "Key exchange completed with 128 bits",
    ///     true
    /// );
    /// ```
    pub fn add_custom_event(&mut self, event_type: &str, source: Option<String>, destination: Option<String>, details: &str, success: bool) {
        self.log_event(event_type, source, destination, details, success);
    }
    
    /// Get the simulation events
    #[must_use]
    pub fn events(&self) -> &[SimulationEvent] {
        &self.events
    }
    
    /// Get the current simulation time
    #[must_use]
    pub const fn current_time(&self) -> Instant {
        self.current_time
    }
    
    /// Check if the simulation is running
    #[must_use]
    pub const fn is_running(&self) -> bool {
        self.running
    }
    
    /// Get the network topology
    #[must_use]
    pub const fn topology(&self) -> &NetworkTopology {
        &self.topology
    }
    
    /// Get a mutable reference to the network topology
    pub const fn topology_mut(&mut self) -> &mut NetworkTopology {
        &mut self.topology
    }
    
    /// Reset the simulation
    ///
    /// Clears all events and resets the simulation time. The network topology
    /// remains unchanged.
    pub fn reset(&mut self) {
        self.events.clear();
        self.current_time = Instant::now();
        self.running = false;
    }

    /// Apply noise to a specific node
    ///
    /// # Arguments
    ///
    /// * `node_id` - ID of the node to apply noise to
    /// * `noise` - Noise factor (0.0-1.0)
    ///
    /// # Returns
    ///
    /// A Result indicating success or failure
    ///
    /// # Errors
    ///
    /// Returns an error if the node is not found or if noise application fails
    pub fn apply_node_noise(&mut self, node_id: &str, noise: f64) -> Result<()> {
        let result = self.topology.get_node_mut(node_id).map_or_else(
            || Err(Error::Network(NetworkError::NodeNotFound(node_id.to_string()))),
            |_node| {
                // This is a placeholder for actual noise application
                // In a real implementation, we would iterate through quantum states and apply noise
                Ok(())
            });
        
        // If successful, record the event
        if result.is_ok() {
            self.log_event(
                "NoiseEvent",
                Some(node_id.to_string()),
                None,
                &format!("Applied noise factor {noise}"),
                true
            );
        }
        
        result
    }
} 