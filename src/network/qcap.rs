// Quantum Capability Announcement Protocol (QCAP)
//
// This protocol allows quantum network nodes to announce and discover
// their quantum capabilities to enable efficient resource allocation and
// capability-aware routing in a quantum network.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;

use crate::network::Node;
use crate::util;

/// Type alias for parameter requirement functions
pub type ParamRequirementFn = Box<dyn Fn(&str) -> bool>;

/// Type alias for parameter requirements map
pub type ParamRequirements = HashMap<String, ParamRequirementFn>;

/// Errors specific to the QCAP protocol
#[derive(Error, Debug)]
pub enum QCAPError {
    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    /// Unknown capability
    #[error("Unknown capability: {0}")]
    UnknownCapability(String),
    
    /// Capability mismatch
    #[error("Capability mismatch: {0}")]
    CapabilityMismatch(String),
    
    /// Protocol timeout
    #[error("Protocol timeout after {0:?}")]
    Timeout(Duration),
    
    /// Protocol error
    #[error("Protocol error: {0}")]
    ProtocolError(String),
}

/// Level of quantum capability
///
/// This represents the maturity and performance level of a quantum capability.
/// Higher levels generally indicate better performance, reliability, and features.
/// The order is from least to most capable: None < Basic < Standard < Advanced < Experimental
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CapabilityLevel {
    /// No capability
    None,
    
    /// Basic capability (limited functionality, lower fidelity)
    Basic,
    
    /// Standard capability (normal operation parameters)
    Standard,
    
    /// Advanced capability (high performance, high fidelity)
    Advanced,
    
    /// Experimental (cutting-edge, highest performance but may not be stable)
    Experimental,
}

impl Default for CapabilityLevel {
    fn default() -> Self {
        Self::None
    }
}

/// Quantum capability description
///
/// Describes a specific quantum capability that a node possesses, including
/// its performance characteristics and other metadata.
///
/// # Examples
///
/// ```
/// use quantum_protocols::network::qcap::{QuantumCapability, CapabilityLevel};
///
/// // Create a capability for quantum error correction
/// let mut qec_capability = QuantumCapability::new(
///     "error_correction".to_string(),
///     "Surface code quantum error correction".to_string(),
///     CapabilityLevel::Advanced,
///     "2.0".to_string()
/// );
///
/// // Add detailed parameters
/// qec_capability.set_parameter("code_distance".to_string(), "7".to_string());
/// qec_capability.set_parameter("logical_error_rate".to_string(), "1e-8".to_string());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCapability {
    /// Name of the capability
    pub name: String,
    
    /// Description of the capability
    pub description: String,
    
    /// Level of capability
    pub level: CapabilityLevel,
    
    /// Version of the capability
    pub version: String,
    
    /// When the capability was last updated
    pub last_updated: u64,
    
    /// Additional parameters for this capability
    pub parameters: HashMap<String, String>,
}

impl QuantumCapability {
    /// Create a new quantum capability
    #[must_use]
    pub fn new(name: String, description: String, level: CapabilityLevel, version: String) -> Self {
        Self {
            name,
            description,
            level,
            version,
            parameters: HashMap::new(),
            last_updated: util::timestamp_now(),
        }
    }
    
    /// Get a parameter value by key
    #[must_use]
    pub fn get_parameter(&self, key: &str) -> Option<&String> {
        self.parameters.get(key)
    }
    
    /// Add or update a parameter
    /// 
    /// Also updates the `last_updated` timestamp.
    pub fn set_parameter(&mut self, key: String, value: String) -> &mut Self {
        self.parameters.insert(key, value);
        self.last_updated = util::timestamp_now();
        self
    }
    
    /// Check if this capability is at least at the specified level
    #[must_use]
    pub fn is_at_least(&self, level: CapabilityLevel) -> bool {
        self.level >= level
    }
}

/// Capability announcement message
///
/// This message is broadcast by nodes to announce their capabilities to the network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityAnnouncement {
    /// Node ID announcing capabilities
    pub node_id: String,
    
    /// List of capabilities
    pub capabilities: Vec<QuantumCapability>,
    
    /// Timestamp of announcement
    pub timestamp: u64,
    
    /// Expiration of this announcement (0 = never)
    pub expiration: u64,
    
    /// Network address for capability requests
    pub address: String,
}

/// Capability request message
///
/// Sent by a node to request specific capability information from another node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityRequest {
    /// Request ID
    pub request_id: String,
    
    /// Node ID requesting capabilities
    pub node_id: String,
    
    /// Specific capabilities requested (empty = all)
    pub requested_capabilities: Vec<String>,
    
    /// Timestamp of request
    pub timestamp: u64,
}

/// Capability response message
///
/// Sent in response to a capability request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityResponse {
    /// Request ID this is responding to
    pub request_id: String,
    
    /// Node ID providing capabilities
    pub node_id: String,
    
    /// List of capabilities
    pub capabilities: Vec<QuantumCapability>,
    
    /// Timestamp of response
    pub timestamp: u64,
}

/// Configuration for QCAP
///
/// Controls the behavior of the QCAP protocol.
#[derive(Debug, Clone)]
pub struct QCAPConfig {
    /// How often to announce capabilities (in milliseconds)
    pub announcement_interval_ms: u64,
    
    /// Capability cache TTL (in milliseconds)
    pub cache_ttl_ms: u64,
    
    /// Maximum announcements to retain per node
    pub max_announcements_per_node: usize,
    
    /// Whether to automatically respond to requests
    pub auto_respond: bool,
    
    /// Timeout for network operations (in milliseconds)
    pub timeout_ms: u64,
    
    /// Whether to broadcast on the network
    pub enable_broadcast: bool,
}

impl Default for QCAPConfig {
    fn default() -> Self {
        Self {
            announcement_interval_ms: 60_000, // 1 minute
            cache_ttl_ms: 300_000,           // 5 minutes
            max_announcements_per_node: 5,
            auto_respond: true,
            timeout_ms: 5_000,               // 5 seconds
            enable_broadcast: true,
        }
    }
}

/// Extension trait for Option to check if None or satisfies predicate
#[allow(dead_code)]
trait OptionExt<T> {
    /// Returns true if the option is None or if the predicate returns true for the value
    fn is_none_or<F>(&self, predicate: F) -> bool
    where
        F: FnOnce(&T) -> bool;
}

impl<T> OptionExt<T> for Option<T> {
    fn is_none_or<F>(&self, predicate: F) -> bool
    where
        F: FnOnce(&T) -> bool,
    {
        match self {
            None => true,
            Some(value) => predicate(value),
        }
    }
}

/// Implementation of Quantum Capability Announcement Protocol
///
/// QCAP enables nodes in a quantum network to announce their quantum capabilities
/// (such as entanglement generation, teleportation, error correction) and to
/// discover the capabilities of other nodes in the network.
///
/// This protocol is essential for:
/// - Capability-aware routing in quantum networks
/// - Resource allocation and quality-of-service management
/// - Service discovery in quantum networks
/// - Network topology optimization
///
/// # Examples
///
/// ```
/// use quantum_protocols::network::qcap::{QCAP, QuantumCapability, CapabilityLevel};
///
/// // Create a new QCAP instance for a node
/// let mut qcap = QCAP::new("node-alpha".to_string());
///
/// // Register a teleportation capability
/// let teleport_cap = QuantumCapability::new(
///     "teleportation".to_string(),
///     "Quantum teleportation of states".to_string(),
///     CapabilityLevel::Standard,
///     "1.5".to_string()
/// );
/// qcap.register_capability(teleport_cap);
///
/// // Create an announcement to broadcast
/// let announcement = qcap.create_announcement();
/// ```
pub struct QCAP {
    /// Node ID
    node_id: String,
    
    /// Configuration
    config: QCAPConfig,
    
    /// Reference to the node
    node: Option<Arc<RwLock<Node>>>,
    
    /// Local capabilities
    local_capabilities: HashMap<String, QuantumCapability>,
    
    /// Known capabilities of other nodes
    remote_capabilities: HashMap<String, Vec<QuantumCapability>>,
    
    /// Last announcement sent
    last_announcement: Option<CapabilityAnnouncement>,
    
    /// Last announcement time
    last_announcement_time: Instant,
    
    /// Pending capability requests
    pending_requests: HashMap<String, CapabilityRequest>,
    
    /// Target nodes for pending requests
    pending_target_nodes: HashMap<String, String>,
}

impl QCAP {
    /// Create a new QCAP instance
    #[must_use]
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            config: QCAPConfig::default(),
            node: None,
            local_capabilities: HashMap::new(),
            remote_capabilities: HashMap::new(),
            last_announcement: None,
            last_announcement_time: Instant::now(),
            pending_requests: HashMap::new(),
            pending_target_nodes: HashMap::new(),
        }
    }
    
    /// Create a new QCAP instance with custom configuration
    #[must_use]
    pub fn with_config(node_id: String, config: QCAPConfig) -> Self {
        Self {
            node_id,
            config,
            node: None,
            local_capabilities: HashMap::new(),
            remote_capabilities: HashMap::new(),
            last_announcement: None,
            last_announcement_time: Instant::now(),
            pending_requests: HashMap::new(),
            pending_target_nodes: HashMap::new(),
        }
    }
    
    /// Set the node reference
    pub fn set_node(&mut self, node: Arc<RwLock<Node>>) {
        self.node = Some(node);
    }
    
    /// Register a local capability
    pub fn register_capability(&mut self, capability: QuantumCapability) {
        self.local_capabilities.insert(capability.name.clone(), capability);
    }
    
    /// Unregister a local capability
    pub fn unregister_capability(&mut self, name: &str) -> Option<QuantumCapability> {
        self.local_capabilities.remove(name)
    }
    
    /// Get a local capability
    #[must_use]
    pub fn get_capability(&self, name: &str) -> Option<&QuantumCapability> {
        self.local_capabilities.get(name)
    }
    
    /// Get all local capabilities
    pub fn get_all_capabilities(&self) -> impl Iterator<Item = &QuantumCapability> {
        self.local_capabilities.values()
    }
    
    /// Update a local capability
    ///
    /// # Errors
    ///
    /// Returns `QCAPError::UnknownCapability` if capability is not found
    pub fn update_capability(&mut self, name: &str, update_fn: impl FnOnce(&mut QuantumCapability)) -> Result<(), QCAPError> {
        if let Some(capability) = self.local_capabilities.get_mut(name) {
            update_fn(capability);
            Ok(())
        } else {
            Err(QCAPError::UnknownCapability(name.to_string()))
        }
    }
    
    /// Create capability announcement
    #[must_use]
    pub fn create_announcement(&self) -> CapabilityAnnouncement {
        let capabilities = self.local_capabilities.values().cloned().collect();
        
        CapabilityAnnouncement {
            node_id: self.node_id.clone(),
            capabilities,
            timestamp: util::timestamp_now(),
            expiration: 0, // Never expire by default
            address: format!("node://{}", self.node_id), // Placeholder
        }
    }
    
    /// Announce capabilities to the network
    ///
    /// # Errors
    ///
    /// Returns `QCAPError` if announcement fails
    pub async fn announce_capabilities(&mut self) -> Result<(), QCAPError> {
        let announcement = self.create_announcement();
        self.last_announcement = Some(announcement.clone());
        self.last_announcement_time = Instant::now();
        
        // In a real implementation, broadcast to the network
        if let Some(node) = &self.node {
            if self.config.enable_broadcast {
                // Placeholder for network broadcast
                // In a real system, serialize and broadcast the announcement
                let _node_guard = node.read().await;
                // node_guard.broadcast_message(...);
            }
        }
        
        Ok(())
    }
    
    /// Receive an announcement from another node
    pub fn receive_announcement(&mut self, announcement: &CapabilityAnnouncement) {
        // Store capabilities for the remote node
        self.remote_capabilities.insert(
            announcement.node_id.clone(),
            announcement.capabilities.clone()
        );
    }
    
    /// Create a capability request for a specific node
    ///
    /// This method creates a request to fetch specific capabilities from a target node.
    /// If the capabilities vector is empty, the request will fetch all capabilities
    /// from the target node.
    ///
    /// # Arguments
    ///
    /// * `target_node` - ID of the node from which to request capabilities
    /// * `capabilities` - Vector of capability names to request (empty = all)
    ///
    /// # Returns
    ///
    /// A `CapabilityRequest` struct that can be sent to the target node
    ///
    /// # Examples
    ///
    /// ```
    /// use quantum_protocols::network::qcap::QCAP;
    ///
    /// let mut qcap = QCAP::new("node-alpha".to_string());
    ///
    /// // Request specific capabilities from node-beta
    /// let request = qcap.create_request(
    ///     "node-beta",
    ///     vec!["entanglement".to_string(), "teleportation".to_string()]
    /// );
    ///
    /// // Request all capabilities
    /// let all_caps_request = qcap.create_request("node-beta", vec![]);
    /// ```
    pub fn create_request(&mut self, target_node: &str, capabilities: Vec<String>) -> CapabilityRequest {
        let request_id = util::generate_id("qcap-req");
        
        let request = CapabilityRequest {
            request_id: request_id.clone(),
            node_id: self.node_id.clone(),
            requested_capabilities: capabilities,
            timestamp: util::timestamp_now(),
        };
        
        // Store in pending requests
        self.pending_requests.insert(request_id.clone(), request.clone());
        
        // Store target node for this request
        self.pending_target_nodes.insert(request_id, target_node.to_string());
        
        request
    }
    
    /// Send a capability request to a node
    ///
    /// # Errors
    ///
    /// Returns `QCAPError` if the request fails
    pub async fn request_capabilities(&mut self, target_node: &str, capabilities: Vec<String>) -> Result<String, QCAPError> {
        let request = self.create_request(target_node, capabilities);
        let request_id = request.request_id.clone();
        
        // In a real implementation, send to the specific node
        if let Some(node) = &self.node {
            // Placeholder for sending request
            let _node_guard = node.read().await;
            // node_guard.send_message(target_node, request);
        }
        
        Ok(request_id)
    }
    
    /// Process a capability request
    pub fn process_request(&mut self, request: CapabilityRequest) -> CapabilityResponse {
        let mut response_capabilities = Vec::new();
        
        // If no specific capabilities requested, return all
        if request.requested_capabilities.is_empty() {
            response_capabilities = self.local_capabilities.values().cloned().collect();
        } else {
            // Otherwise, return only requested capabilities
            for cap_name in &request.requested_capabilities {
                if let Some(cap) = self.local_capabilities.get(cap_name) {
                    response_capabilities.push(cap.clone());
                }
            }
        }
        
        CapabilityResponse {
            request_id: request.request_id,
            node_id: self.node_id.clone(),
            capabilities: response_capabilities,
            timestamp: util::timestamp_now(),
        }
    }
    
    /// Receive a capability response
    pub fn receive_response(&mut self, response: &CapabilityResponse) {
        // Remove from pending requests
        self.pending_requests.remove(&response.request_id);
        
        // Also remove from pending target nodes
        self.pending_target_nodes.remove(&response.request_id);
        
        // Store capabilities for the remote node
        self.remote_capabilities.insert(
            response.node_id.clone(),
            response.capabilities.clone()
        );
    }
    
    /// Get node capabilities
    #[must_use]
    pub fn get_node_capabilities(&self, node_id: &str) -> Option<&Vec<QuantumCapability>> {
        self.remote_capabilities.get(node_id)
    }
    
    /// Check if a node has a capability at specified level
    #[must_use]
    pub fn node_has_capability(&self, node_id: &str, capability_name: &str, min_level: CapabilityLevel) -> bool {
        if let Some(capabilities) = self.remote_capabilities.get(node_id) {
            for cap in capabilities {
                if cap.name == capability_name && cap.level >= min_level {
                    return true;
                }
            }
        }
        false
    }
    
    /// Find nodes with a specific capability
    #[must_use]
    pub fn find_nodes_with_capability(&self, capability_name: &str, min_level: CapabilityLevel) -> Vec<String> {
        let mut matching_nodes = Vec::new();
        
        for (node_id, capabilities) in &self.remote_capabilities {
            for cap in capabilities {
                if cap.name == capability_name && cap.level >= min_level {
                    matching_nodes.push(node_id.clone());
                    break;
                }
            }
        }
        
        matching_nodes
    }
    
    /// Run periodic maintenance tasks
    ///
    /// # Errors
    ///
    /// Returns `QCAPError` if maintenance fails
    pub async fn periodic_maintenance(&mut self) -> Result<(), QCAPError> {
        let now = Instant::now();
        
        // Check if it's time to send a new announcement
        if now.duration_since(self.last_announcement_time).as_millis() >= u128::from(self.config.announcement_interval_ms) {
            self.announce_capabilities().await?;
        }
        
        // Expire old data (To be implemented)
        
        Ok(())
    }
    
    /// Get current configuration
    #[must_use]
    pub fn config(&self) -> &QCAPConfig {
        &self.config
    }
    
    /// Modify configuration
    pub fn config_mut(&mut self) -> &mut QCAPConfig {
        &mut self.config
    }
    
    /// Find capabilities by minimum level
    #[must_use]
    pub fn find_capabilities_by_level(&self, min_level: CapabilityLevel) -> HashMap<String, Vec<String>> {
        let mut result: HashMap<String, Vec<String>> = HashMap::new();
        
        // Iterate through all known remote capabilities
        for (node_id, capabilities) in &self.remote_capabilities {
            for capability in capabilities {
                if capability.level >= min_level {
                    // Add the node to the list for this capability
                    result.entry(capability.name.clone())
                        .or_default()
                        .push(node_id.clone());
                }
            }
        }
        
        // Also check local capabilities
        for (name, capability) in &self.local_capabilities {
            if capability.level >= min_level {
                result.entry(name.clone())
                    .or_default()
                    .push(self.node_id.clone());
            }
        }
        
        result
    }

    /// Find nodes that have all specified capabilities
    #[must_use]
    pub fn find_nodes_with_all_capabilities(&self, requirements: &HashMap<String, CapabilityLevel>) -> Vec<String> {
        if requirements.is_empty() {
            return Vec::new();
        }
        
        let mut result = Vec::new();
        
        // Check remote nodes
        'node_loop: for (node_id, capabilities) in &self.remote_capabilities {
            // For each requirement, check if the node has it
            for (req_name, req_level) in requirements {
                // Find a matching capability
                let has_matching_capability = capabilities.iter().any(|cap| {
                    cap.name == *req_name && cap.level >= *req_level
                });
                
                // If a requirement is not met, skip this node
                if !has_matching_capability {
                    continue 'node_loop;
                }
            }
            
            // If we get here, the node meets all requirements
            result.push(node_id.clone());
        }
        
        // Check if local node meets all requirements
        let mut local_qualifies = true;
        for (req_name, req_level) in requirements {
            if let Some(cap) = self.local_capabilities.get(req_name) {
                if cap.level < *req_level {
                    local_qualifies = false;
                    break;
                }
            } else {
                local_qualifies = false;
                break;
            }
        }
        
        if local_qualifies {
            result.push(self.node_id.clone());
        }
        
        result
    }
    
    /// Find capabilities with specific parameters
    #[must_use]
    pub fn find_capabilities_with_parameters(
        &self,
        capability_name: &str,
        min_level: Option<CapabilityLevel>,
        param_requirements: &ParamRequirements
    ) -> HashMap<String, QuantumCapability>
    {
        let mut result = HashMap::new();
        
        // Check remote node capabilities
        for (node_id, capabilities) in &self.remote_capabilities {
            for cap in capabilities {
                // Check if this is the capability we're looking for
                if cap.name != capability_name {
                    continue;
                }
                
                // Check if it meets the minimum level requirement
                if let Some(level) = min_level {
                    if cap.level < level {
                        continue;
                    }
                }
                
                // Check all parameter requirements
                let all_params_match = param_requirements.iter()
                    .all(|(param_name, check_fn)| {
                        cap.get_parameter(param_name)
                            .is_some_and(|param_value| check_fn(param_value))
                    });
                
                if all_params_match {
                    result.insert(node_id.clone(), cap.clone());
                }
            }
        }
        
        // Check local capabilities
        if let Some(cap) = self.local_capabilities.get(capability_name) {
            // Check level requirement
            let level_meets_requirement = min_level.is_none_or(|level| cap.level >= level);
            
            if level_meets_requirement {
                // Check parameter requirements using the same functional style
                let all_params_match = param_requirements.iter()
                    .all(|(param_name, check_fn)| {
                        cap.get_parameter(param_name)
                            .is_some_and(|param_value| check_fn(param_value))
                    });
                
                if all_params_match {
                    result.insert(self.node_id.clone(), cap.clone());
                }
            }
        }
        
        result
    }
    
    /// Discover capabilities dynamically from the network
    ///
    /// This method broadcasts a capability discovery request to all known nodes
    /// and waits for responses up to the timeout duration.
    ///
    /// # Arguments
    ///
    /// * `capability_names` - Optional list of specific capabilities to discover. If empty, all capabilities are requested.
    /// * `timeout_ms` - Timeout for discovery in milliseconds. If None, uses the config timeout.
    ///
    /// # Returns
    ///
    /// A map of node IDs to their capabilities matching the discovery request
    ///
    /// # Errors
    ///
    /// Returns an error if discovery fails or times out
    pub async fn discover_capabilities(
        &mut self,
        capability_names: Vec<String>,
        timeout_ms: Option<u64>
    ) -> Result<HashMap<String, Vec<QuantumCapability>>, QCAPError> {
        // If not connected to a node, we cannot discover capabilities
        if self.node.is_none() {
            return Err(QCAPError::NetworkError("Not connected to network node".to_string()));
        }
        
        // Remember the nodes we had before discovery
        let previously_known: HashSet<String> = self.remote_capabilities.keys().cloned().collect();
        
        // Prepare the discovery timeout
        let timeout = timeout_ms.unwrap_or(self.config.timeout_ms);
        
        // Create a broadcast request if broadcast is enabled
        if self.config.enable_broadcast {
            // Create broadcast request ID
            let broadcast_request_id = format!("discovery_{}", util::generate_id("qcap"));
            
            // Create the request
            let request = CapabilityRequest {
                request_id: broadcast_request_id.clone(),
                node_id: self.node_id.clone(),
                requested_capabilities: capability_names.clone(),
                timestamp: util::timestamp_now(),
            };
            
            // Remember this pending request
            self.pending_requests.insert(broadcast_request_id.clone(), request.clone());
            
            // Send the broadcast request through our node
            if let Some(node) = &self.node {
                let node_locked = node.read().await;
                
                // Broadcast request - in a real implementation, this would use the network
                // For now, simulate broadcasting by just logging
                if let Err(e) = node_locked.broadcast("qcap_discovery", &request) {
                    return Err(QCAPError::NetworkError(format!("Broadcast error: {e}")));
                }
            }
        }
        
        // Also send direct requests to any known nodes
        let known_nodes: Vec<String> = self.remote_capabilities.keys().cloned().collect();
        
        for node_id in known_nodes {
            // Send a request to this specific node
            let _ = self.request_capabilities(&node_id, capability_names.clone()).await?;
        }
        
        // Wait for responses up to the timeout
        let start_time = Instant::now();
        
        // In a real implementation, we would use a proper async timeout mechanism
        // For this simulation, we'll check every 100ms up to timeout
        let sleep_interval = Duration::from_millis(100);
        
        while start_time.elapsed().as_millis() < u128::from(timeout) {
            // See if we've received any new capabilities
            let newly_discovered: Vec<String> = self.remote_capabilities.keys()
                .filter(|node_id| !previously_known.contains(*node_id))
                .cloned()
                .collect();
            
            // If we've discovered capabilities and have some from all known nodes,
            // we can return early
            if !newly_discovered.is_empty() && 
               self.pending_requests.is_empty() {
                break;
            }
            
            // Wait a bit before checking again
            tokio::time::sleep(sleep_interval).await;
        }
        
        // Check if we timed out with pending requests
        if !self.pending_requests.is_empty() && start_time.elapsed().as_millis() >= u128::from(timeout) {
            // We can still return partial results, but generate a warning
            eprintln!("Warning: Capability discovery timed out with {} pending requests", 
                     self.pending_requests.len());
        }
        
        // Filter capabilities based on the requested types
        let mut results = HashMap::new();
        
        for (node_id, capabilities) in &self.remote_capabilities {
            // If specific capabilities were requested, filter the results
            if capability_names.is_empty() {
                // Include all capabilities when no specific ones requested
                results.insert(node_id.clone(), capabilities.clone());
            } else {
                // Filter to just the requested capabilities
                let filtered_capabilities: Vec<QuantumCapability> = capabilities.iter()
                    .filter(|cap| capability_names.contains(&cap.name))
                    .cloned()
                    .collect();
                
                if !filtered_capabilities.is_empty() {
                    results.insert(node_id.clone(), filtered_capabilities);
                }
            }
        }
        
        Ok(results)
    }
    
    /// Wait for a response to a specific request
    ///
    /// # Arguments
    ///
    /// * `request_id` - The ID of the request to wait for
    /// * `timeout_ms` - Timeout in milliseconds
    ///
    /// # Returns
    ///
    /// The capabilities from the response, or an error if timeout
    ///
    /// # Errors
    ///
    /// Returns an error if waiting times out
    pub async fn wait_for_response(&mut self, request_id: &str, timeout_ms: u64) 
        -> Result<Vec<QuantumCapability>, QCAPError> 
    {
        // Check if this is a valid request
        if !self.pending_requests.contains_key(request_id) {
            return Err(QCAPError::ProtocolError(format!("Unknown request ID: {request_id}")));
        }
        
        // Get the target node for this request
        let target_node = match self.pending_target_nodes.get(request_id) {
            Some(node) => node.clone(),
            None => return Err(QCAPError::ProtocolError(format!("No target node for request: {request_id}"))),
        };
        
        // Start waiting from this moment
        let start_time = Instant::now();
        let timeout = Duration::from_millis(timeout_ms);
        
        // In a real implementation, we'd use a proper async notification mechanism
        // For simulation, poll every 100ms
        let interval = Duration::from_millis(100);
        
        while start_time.elapsed() < timeout {
            // Check if request is still pending (i.e., no response yet)
            if !self.pending_requests.contains_key(request_id) {
                // Request is no longer pending, so we got a response
                
                // Get the capabilities for the target node
                if let Some(capabilities) = self.remote_capabilities.get(&target_node) {
                    return Ok(capabilities.clone());
                }
                
                return Err(QCAPError::ProtocolError(
                    format!("Response processed but no capabilities found for node: {target_node}")
                ));
            }
            
            // Wait a bit before checking again
            tokio::time::sleep(interval).await;
        }
        
        // If we get here, we timed out waiting for a response
        Err(QCAPError::Timeout(timeout))
    }
    
    /// Prune expired capabilities from the cache
    ///
    /// This method removes any capabilities that have expired
    /// based on the cache TTL from the configuration.
    pub fn prune_expired_capabilities(&mut self) {
        let _now = Instant::now();
        let cache_ttl = Duration::from_millis(self.config.cache_ttl_ms);
        
        // This would be implemented in a real system to remove stale entries
        // For simulation, we'll just log that pruning would happen here
        eprintln!("Pruning capabilities older than {cache_ttl:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::RwLock;
    
    #[test]
    fn test_capability_creation() {
        let mut cap = QuantumCapability::new(
            "entanglement".to_string(),
            "Ability to create and maintain entangled qubits".to_string(),
            CapabilityLevel::Standard,
            "1.0".to_string()
        );
        
        assert_eq!(cap.name, "entanglement");
        assert_eq!(cap.level, CapabilityLevel::Standard);
        
        // Add parameters
        cap.set_parameter("max_qubits".to_string(), "10".to_string());
        cap.set_parameter("fidelity".to_string(), "0.98".to_string());
        
        assert_eq!(cap.get_parameter("max_qubits"), Some(&"10".to_string()));
        assert_eq!(cap.get_parameter("fidelity"), Some(&"0.98".to_string()));
        assert_eq!(cap.get_parameter("non_existent"), None);
        
        // Test level comparison
        assert!(cap.is_at_least(CapabilityLevel::Basic));
        assert!(cap.is_at_least(CapabilityLevel::Standard));
        assert!(!cap.is_at_least(CapabilityLevel::Advanced));
    }
    
    #[test]
    fn test_capability_registration() {
        let mut qcap = QCAP::new("test_node".to_string());
        
        // Register capabilities
        let cap1 = QuantumCapability::new(
            "entanglement".to_string(),
            "Entanglement capability".to_string(),
            CapabilityLevel::Standard,
            "1.0".to_string()
        );
        
        let cap2 = QuantumCapability::new(
            "teleportation".to_string(),
            "Teleportation capability".to_string(),
            CapabilityLevel::Advanced,
            "2.0".to_string()
        );
        
        qcap.register_capability(cap1);
        qcap.register_capability(cap2);
        
        // Check registration
        assert!(qcap.get_capability("entanglement").is_some());
        assert!(qcap.get_capability("teleportation").is_some());
        assert!(qcap.get_capability("non_existent").is_none());
        
        // Unregister a capability
        let removed = qcap.unregister_capability("entanglement");
        assert!(removed.is_some());
        assert!(qcap.get_capability("entanglement").is_none());
        
        // Count remaining capabilities
        let all_caps: Vec<_> = qcap.get_all_capabilities().collect();
        assert_eq!(all_caps.len(), 1);
        assert_eq!(all_caps[0].name, "teleportation");
    }
    
    #[test]
    fn test_announcement_creation() {
        let mut qcap = QCAP::new("test_node".to_string());
        
        // Register capabilities
        let cap1 = QuantumCapability::new(
            "entanglement".to_string(),
            "Entanglement capability".to_string(),
            CapabilityLevel::Standard,
            "1.0".to_string()
        );
        
        qcap.register_capability(cap1);
        
        // Create an announcement
        let announcement = qcap.create_announcement();
        
        assert_eq!(announcement.node_id, "test_node");
        assert_eq!(announcement.capabilities.len(), 1);
        assert_eq!(announcement.capabilities[0].name, "entanglement");
    }
    
    #[tokio::test]
    async fn test_request_response() {
        let mut qcap1 = QCAP::new("node1".to_string());
        let mut qcap2 = QCAP::new("node2".to_string());
        
        // Register capabilities for node1
        let cap1 = QuantumCapability::new(
            "entanglement".to_string(),
            "Entanglement capability".to_string(),
            CapabilityLevel::Standard,
            "1.0".to_string()
        );
        
        qcap1.register_capability(cap1);
        
        // Create a request from node2 to node1
        let request = qcap2.create_request("node1", vec!["entanglement".to_string()]);
        
        // Process the request at node1
        let response = qcap1.process_request(request);
        
        // Verify response
        assert_eq!(response.node_id, "node1");
        assert_eq!(response.capabilities.len(), 1);
        assert_eq!(response.capabilities[0].name, "entanglement");
        
        // Receive the response at node2
        qcap2.receive_response(&response);
        
        // Check that node2 now knows about node1's capabilities
        let node1_caps = qcap2.get_node_capabilities("node1").unwrap();
        assert_eq!(node1_caps.len(), 1);
        assert_eq!(node1_caps[0].name, "entanglement");
        
        // Check capability lookup
        assert!(qcap2.node_has_capability("node1", "entanglement", CapabilityLevel::Basic));
        assert!(qcap2.node_has_capability("node1", "entanglement", CapabilityLevel::Standard));
        assert!(!qcap2.node_has_capability("node1", "entanglement", CapabilityLevel::Advanced));
        assert!(!qcap2.node_has_capability("node1", "teleportation", CapabilityLevel::Basic));
    }
    
    #[test]
    fn test_find_capabilities_by_level() {
        let mut qcap = QCAP::new("test_node".to_string());
        
        // Register local capabilities
        let cap1 = QuantumCapability::new(
            "teleportation".to_string(),
            "Teleportation capability".to_string(),
            CapabilityLevel::Advanced,
            "2.0".to_string()
        );
        
        let cap2 = QuantumCapability::new(
            "error_correction".to_string(),
            "Error correction capability".to_string(),
            CapabilityLevel::Standard,
            "1.5".to_string()
        );
        
        let cap3 = QuantumCapability::new(
            "quantum_memory".to_string(),
            "Quantum memory with long coherence".to_string(),
            CapabilityLevel::Experimental,
            "0.9".to_string()
        );
        
        qcap.register_capability(cap1);
        qcap.register_capability(cap2);
        qcap.register_capability(cap3);
        
        // Add remote node capabilities
        let remote_node = "remote_node".to_string();
        let remote_caps = vec![
            QuantumCapability::new(
                "teleportation".to_string(),
                "Basic teleportation".to_string(),
                CapabilityLevel::Basic,
                "1.0".to_string()
            ),
            QuantumCapability::new(
                "entanglement".to_string(),
                "Advanced entanglement".to_string(),
                CapabilityLevel::Advanced,
                "3.0".to_string()
            )
        ];
        
        qcap.remote_capabilities.insert(remote_node.clone(), remote_caps);
        
        // Test filtering by different levels
        let basic_caps = qcap.find_capabilities_by_level(CapabilityLevel::Basic);
        assert_eq!(basic_caps.len(), 4); // teleportation, error_correction, entanglement, quantum_memory
        assert!(basic_caps.contains_key("teleportation"));
        assert!(basic_caps.contains_key("error_correction"));
        assert!(basic_caps.contains_key("entanglement"));
        assert!(basic_caps.contains_key("quantum_memory"));
        
        // Check that both nodes are included for teleportation
        assert_eq!(basic_caps.get("teleportation").unwrap().len(), 2);
        assert!(basic_caps.get("teleportation").unwrap().contains(&"test_node".to_string()));
        assert!(basic_caps.get("teleportation").unwrap().contains(&remote_node));
        
        // Test advanced level
        let advanced_caps = qcap.find_capabilities_by_level(CapabilityLevel::Advanced);
        assert_eq!(advanced_caps.len(), 3); // teleportation, entanglement, quantum_memory
        
        // Only local node should have advanced teleportation
        assert_eq!(advanced_caps.get("teleportation").unwrap().len(), 1);
        assert!(advanced_caps.get("teleportation").unwrap().contains(&"test_node".to_string()));
        
        // Only remote node should have entanglement
        assert_eq!(advanced_caps.get("entanglement").unwrap().len(), 1);
        assert!(advanced_caps.get("entanglement").unwrap().contains(&remote_node));
        
        // Test experimental level
        let experimental_caps = qcap.find_capabilities_by_level(CapabilityLevel::Experimental);
        assert_eq!(experimental_caps.len(), 1); // only quantum_memory
        assert!(experimental_caps.contains_key("quantum_memory"));
        assert_eq!(experimental_caps.get("quantum_memory").unwrap().len(), 1);
        assert!(experimental_caps.get("quantum_memory").unwrap().contains(&"test_node".to_string()));
        
        // Verify ordering of CapabilityLevel enum
        assert!(CapabilityLevel::None < CapabilityLevel::Basic);
        assert!(CapabilityLevel::Basic < CapabilityLevel::Standard);
        assert!(CapabilityLevel::Standard < CapabilityLevel::Advanced);
        assert!(CapabilityLevel::Advanced < CapabilityLevel::Experimental);
    }
    
    #[test]
    fn test_find_nodes_with_all_capabilities() {
        let mut qcap = QCAP::new("test_node".to_string());
        
        // Register local capabilities
        qcap.register_capability(QuantumCapability::new(
            "teleportation".to_string(),
            "Advanced teleportation".to_string(),
            CapabilityLevel::Advanced,
            "2.0".to_string()
        ));
        qcap.register_capability(QuantumCapability::new(
            "entanglement".to_string(),
            "Standard entanglement".to_string(),
            CapabilityLevel::Standard,
            "1.5".to_string()
        ));
        
        // Set up remote node A with multiple capabilities
        let node_a = "node_a".to_string();
        let node_a_caps = vec![
            QuantumCapability::new(
                "teleportation".to_string(),
                "Standard teleportation".to_string(),
                CapabilityLevel::Standard,
                "1.5".to_string()
            ),
            QuantumCapability::new(
                "entanglement".to_string(),
                "Advanced entanglement".to_string(),
                CapabilityLevel::Advanced,
                "2.0".to_string()
            ),
            QuantumCapability::new(
                "error_correction".to_string(),
                "Basic error correction".to_string(),
                CapabilityLevel::Basic,
                "1.0".to_string()
            )
        ];
        qcap.remote_capabilities.insert(node_a.clone(), node_a_caps);
        
        // Set up remote node B with some capabilities
        let node_b = "node_b".to_string();
        let node_b_caps = vec![
            QuantumCapability::new(
                "teleportation".to_string(),
                "Basic teleportation".to_string(),
                CapabilityLevel::Basic,
                "1.0".to_string()
            ),
            QuantumCapability::new(
                "qubits".to_string(),
                "Standard qubits".to_string(),
                CapabilityLevel::Standard,
                "1.5".to_string()
            )
        ];
        qcap.remote_capabilities.insert(node_b.clone(), node_b_caps);
        
        // Test 1: Find nodes with both teleportation (Standard) and entanglement (Standard)
        let mut requirements = HashMap::new();
        requirements.insert("teleportation".to_string(), CapabilityLevel::Standard);
        requirements.insert("entanglement".to_string(), CapabilityLevel::Standard);
        
        let matching_nodes = qcap.find_nodes_with_all_capabilities(&requirements);
        assert_eq!(matching_nodes.len(), 2);  // Should find test_node and node_a
        assert!(matching_nodes.contains(&"test_node".to_string()));
        assert!(matching_nodes.contains(&node_a));
        
        // Test 2: Find nodes with all three capabilities
        requirements.insert("error_correction".to_string(), CapabilityLevel::Basic);
        
        let matching_nodes = qcap.find_nodes_with_all_capabilities(&requirements);
        assert_eq!(matching_nodes.len(), 1);  // Only node_a has all three
        assert!(matching_nodes.contains(&node_a));
        
        // Test 3: Find nodes with an Advanced requirement
        let mut requirements = HashMap::new();
        requirements.insert("teleportation".to_string(), CapabilityLevel::Advanced);
        
        let matching_nodes = qcap.find_nodes_with_all_capabilities(&requirements);
        assert_eq!(matching_nodes.len(), 1);  // Only test_node has advanced teleportation
        assert!(matching_nodes.contains(&"test_node".to_string()));
        
        // Test 4: Empty requirements should return empty result
        let empty_requirements = HashMap::new();
        let matching_nodes = qcap.find_nodes_with_all_capabilities(&empty_requirements);
        assert!(matching_nodes.is_empty());
    }
    
    #[test]
    fn test_find_capabilities_with_parameters() {
        let mut qcap = QCAP::new("test_node".to_string());
        
        // Register local capability with parameters
        let mut local_cap = QuantumCapability::new(
            "qubits".to_string(),
            "Large qubit array".to_string(),
            CapabilityLevel::Advanced,
            "2.0".to_string()
        );
        local_cap.set_parameter("count".to_string(), "100".to_string());
        local_cap.set_parameter("coherence_time_us".to_string(), "200".to_string());
        local_cap.set_parameter("error_rate".to_string(), "0.001".to_string());
        qcap.register_capability(local_cap);
        
        // Set up remote node with parameters
        let node_a = "node_a".to_string();
        let mut remote_cap = QuantumCapability::new(
            "qubits".to_string(),
            "Medium qubit array".to_string(),
            CapabilityLevel::Standard,
            "1.5".to_string()
        );
        remote_cap.set_parameter("count".to_string(), "50".to_string());
        remote_cap.set_parameter("coherence_time_us".to_string(), "150".to_string());
        remote_cap.set_parameter("error_rate".to_string(), "0.005".to_string());
        
        qcap.remote_capabilities.insert(node_a.clone(), vec![remote_cap]);
        
        // Test 1: Find capabilities with minimum count of 50
        let mut param_checks: ParamRequirements = HashMap::new();
        param_checks.insert("count".to_string(), Box::new(|value: &str| {
            value.parse::<u32>().map(|n| n >= 50).unwrap_or(false)
        }));
        
        let matching = qcap.find_capabilities_with_parameters(
            "qubits",
            None,
            &param_checks
        );
        
        assert_eq!(matching.len(), 2);  // Both nodes have qubits with count >= 50
        assert!(matching.contains_key("test_node"));
        assert!(matching.contains_key(&node_a));
        
        // Test 2: Find capabilities with minimum count AND coherence time
        let mut param_checks: ParamRequirements = HashMap::new();
        param_checks.insert("count".to_string(), Box::new(|value: &str| {
            value.parse::<u32>().map(|n| n >= 75).unwrap_or(false)
        }));
        param_checks.insert("coherence_time_us".to_string(), Box::new(|value: &str| {
            value.parse::<u32>().map(|n| n >= 180).unwrap_or(false)
        }));
        
        let matching = qcap.find_capabilities_with_parameters(
            "qubits",
            None,
            &param_checks
        );
        
        assert_eq!(matching.len(), 1);  // Only test_node meets both requirements
        assert!(matching.contains_key("test_node"));
        
        // Test 3: Find capabilities with level filter
        let mut param_checks: ParamRequirements = HashMap::new();
        param_checks.insert("count".to_string(), Box::new(|value: &str| {
            value.parse::<u32>().map(|n| n >= 50).unwrap_or(false)
        }));
        
        let matching = qcap.find_capabilities_with_parameters(
            "qubits",
            Some(CapabilityLevel::Advanced),
            &param_checks
        );
        
        assert_eq!(matching.len(), 1);  // Only test_node has Advanced level
        assert!(matching.contains_key("test_node"));
        
        // Test 4: Test with a parameter that doesn't exist
        let mut param_checks: ParamRequirements = HashMap::new();
        param_checks.insert("non_existent_param".to_string(), Box::new(|_: &str| true));
        
        let matching = qcap.find_capabilities_with_parameters(
            "qubits",
            None,
            &param_checks
        );
        
        assert_eq!(matching.len(), 0);  // No capabilities have this parameter
    }
    
    #[tokio::test]
    async fn test_capability_discovery() {
        // Create two nodes
        let mut qcap1 = QCAP::new("node1".to_string());
        let mut qcap2 = QCAP::new("node2".to_string());
        
        // Add a node to make async behavior testable
        let node_mock = Arc::new(RwLock::new(Node::new("node1")));
        qcap1.set_node(node_mock.clone());
        
        // Register some capabilities with each node
        let cap1 = QuantumCapability::new(
            "entanglement".to_string(),
            "Entanglement capability".to_string(),
            CapabilityLevel::Standard,
            "1.0".to_string()
        );
        
        let cap2 = QuantumCapability::new(
            "error_correction".to_string(),
            "Error correction capability".to_string(),
            CapabilityLevel::Advanced,
            "2.0".to_string()
        );
        
        let cap3 = QuantumCapability::new(
            "measurement".to_string(),
            "Measurement capability".to_string(),
            CapabilityLevel::Basic,
            "1.5".to_string()
        );
        
        // Register capabilities
        qcap1.register_capability(cap1.clone());
        qcap1.register_capability(cap3.clone());
        qcap2.register_capability(cap2.clone());
        qcap2.register_capability(cap3.clone());
        
        // Skip actual network interactions in the test by directly handling
        // requests and responses
        
        // 1. Create a request from node1 to node2
        let request = qcap1.create_request("node2", vec![
            "error_correction".to_string(),
            "measurement".to_string()
        ]);
        
        // 2. Simulate node2 processing the request
        let response = qcap2.process_request(request.clone());
        
        // 3. Directly handle the response in node1
        qcap1.receive_response(&response);
        
        // Now node1 should have node2's capabilities
        assert!(qcap1.remote_capabilities.contains_key("node2"));
        
        let node2_caps = qcap1.remote_capabilities.get("node2").unwrap();
        assert_eq!(node2_caps.len(), 2); // Should have received 2 capabilities
        
        // Verify capability discovery works - this tests the client-side behavior
        // with no actual network but using already received capabilities
        let discovery_result = qcap1.discover_capabilities(
            vec!["measurement".to_string()],
            Some(1000) // Short timeout for test
        ).await.unwrap();
        
        // Should discover node2 with measurement capability
        assert!(discovery_result.contains_key("node2"));
        
        let discovered_capabilities = &discovery_result["node2"];
        assert_eq!(discovered_capabilities.len(), 1);
        assert_eq!(discovered_capabilities[0].name, "measurement");
        
        // Test pruning capabilities
        qcap1.prune_expired_capabilities();
        
        // Test waiting for response
        // This will return error since request doesn't exist anymore
        let wait_result = qcap1.wait_for_response(&request.request_id, 100).await;
        assert!(wait_result.is_err()); // Should error out since request is no longer pending
    }
} 