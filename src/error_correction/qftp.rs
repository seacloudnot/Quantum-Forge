// Quantum Fault Tolerance Protocol (QFTP)
//
// This protocol provides fault tolerance for quantum systems, including
// failure detection, recovery procedures, and redundant entanglement paths.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use thiserror::Error;
use crate::network::entanglement::EntanglementPair;
use crate::core::QuantumState;
use crate::util;

// Add this constant at the top of the file
const FLOAT_EPSILON: f64 = 1e-10;

/// Errors specific to QFTP
#[derive(Error, Debug)]
pub enum QFTPError {
    /// Node failure detected
    #[error("Node failure detected: {0}")]
    NodeFailure(String),
    
    /// Path failure
    #[error("Path failure between {0} and {1}")]
    PathFailure(String, String),
    
    /// Insufficient redundancy
    #[error("Insufficient redundancy: have {0}, need {1}")]
    InsufficientRedundancy(usize, usize),
    
    /// Recovery error
    #[error("Recovery error: {0}")]
    RecoveryError(String),
    
    /// Operation timeout
    #[error("Operation timed out after {0:?}")]
    Timeout(Duration),
    
    /// Resource exhaustion
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
}

/// Failure detection method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureDetection {
    /// Heartbeat-based detection
    Heartbeat,
    
    /// Entanglement verification
    EntanglementVerification,
    
    /// Distributed consensus detection
    ConsensusBasedDetection,
    
    /// Quantum error correction syndrome detection
    SyndromeDetection,
}

/// Recovery strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryStrategy {
    /// Immediate rerouting through alternate paths
    ImmediateRerouting,
    
    /// Entanglement distillation
    EntanglementDistillation,
    
    /// State teleportation to working nodes
    StateTeleportation,
    
    /// Distributed error correction
    DistributedErrorCorrection,
}

/// Status of a node
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    /// Node is fully operational
    Operational,
    
    /// Node is degraded but functioning
    Degraded,
    
    /// Node has partially failed
    PartialFailure,
    
    /// Node has completely failed
    CompleteFailure,
    
    /// Node status is unknown
    Unknown,
}

/// Configuration for QFTP
#[derive(Debug, Clone)]
pub struct QFTPConfig {
    /// Required redundancy level
    pub redundancy_level: usize,
    
    /// Failure detection method
    pub detection_method: FailureDetection,
    
    /// Recovery strategy
    pub recovery_strategy: RecoveryStrategy,
    
    /// Heartbeat interval in milliseconds
    pub heartbeat_interval_ms: u64,
    
    /// Failure threshold for degraded operation
    pub failure_threshold: f64,
    
    /// Whether to automatically attempt recovery
    pub auto_recovery: bool,
    
    /// Maximum recovery attempts
    pub max_recovery_attempts: usize,
}

impl Default for QFTPConfig {
    fn default() -> Self {
        Self {
            redundancy_level: 3,
            detection_method: FailureDetection::EntanglementVerification,
            recovery_strategy: RecoveryStrategy::ImmediateRerouting,
            heartbeat_interval_ms: 1000,
            failure_threshold: 0.3, // 30% failure threshold
            auto_recovery: true,
            max_recovery_attempts: 5,
        }
    }
}

/// Path redundancy information
#[derive(Debug, Clone)]
pub struct RedundantPath {
    /// Path identifier
    pub id: String,
    
    /// Source node ID
    pub source: String,
    
    /// Destination node ID
    pub destination: String,
    
    /// Alternative paths (in order of preference)
    pub alternatives: Vec<Vec<String>>,
    
    /// Current active path index
    pub active_path_index: usize,
    
    /// Path status
    pub status: PathStatus,
    
    /// Last verification time
    pub last_verified: Instant,
    
    /// Number of failures since establishment
    pub failure_count: usize,
}

/// Path status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathStatus {
    /// Path is active and verified
    Active,
    
    /// Path is degraded but usable
    Degraded,
    
    /// Path has failed
    Failed,
    
    /// Path status is unknown
    Unknown,
}

/// Main implementation of QFTP
pub struct QFTP {
    /// Configuration
    config: QFTPConfig,
    
    /// Network nodes
    nodes: HashMap<String, NodeStatus>,
    
    /// Redundant paths
    paths: HashMap<String, RedundantPath>,
    
    /// Entanglement pairs
    entanglement_pairs: HashMap<String, EntanglementPair>,
    
    /// Last failure detection time
    last_detection: Instant,
    
    /// Last recovery attempt time
    last_recovery: Instant,
    
    /// Recovery attempt counter
    recovery_attempts: usize,
    
    /// Protected quantum states
    protected_states: HashMap<String, QuantumState>,
}

impl Default for QFTP {
    fn default() -> Self {
        Self::new()
    }
}

impl QFTP {
    /// Create a new QFTP instance with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: QFTPConfig::default(),
            nodes: HashMap::new(),
            paths: HashMap::new(),
            entanglement_pairs: HashMap::new(),
            last_detection: Instant::now(),
            last_recovery: Instant::now(),
            recovery_attempts: 0,
            protected_states: HashMap::new(),
        }
    }
    
    /// Create a new QFTP instance with custom configuration
    #[must_use]
    pub fn with_config(config: QFTPConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            paths: HashMap::new(),
            entanglement_pairs: HashMap::new(),
            last_detection: Instant::now(),
            last_recovery: Instant::now(),
            recovery_attempts: 0,
            protected_states: HashMap::new(),
        }
    }
    
    /// Register a node with the fault tolerance system
    pub fn register_node(&mut self, node_id: String, status: NodeStatus) {
        self.nodes.insert(node_id, status);
    }
    
    /// Register multiple nodes
    pub fn register_nodes(&mut self, nodes: Vec<(String, NodeStatus)>) {
        for (node_id, status) in nodes {
            self.register_node(node_id, status);
        }
    }
    
    /// Check if a node is registered
    #[must_use]
    pub fn has_node(&self, node_id: &str) -> bool {
        self.nodes.contains_key(node_id)
    }
    
    /// Get node status
    #[must_use]
    pub fn node_status(&self, node_id: &str) -> Option<NodeStatus> {
        self.nodes.get(node_id).copied()
    }
    
    /// Update node status
    pub fn update_node_status(&mut self, node_id: &str, status: NodeStatus) -> bool {
        if let Some(current_status) = self.nodes.get_mut(node_id) {
            *current_status = status;
            
            // If a node has failed, mark all paths through it as failed
            if status == NodeStatus::CompleteFailure {
                self.mark_paths_through_node_as_failed(node_id);
            }
            
            true
        } else {
            false
        }
    }
    
    /// Mark all paths through a failed node as failed
    fn mark_paths_through_node_as_failed(&mut self, node_id: &str) {
        for path in self.paths.values_mut() {
            // Check if node is in any of the alternative paths
            let affected = path.alternatives.iter().any(|route| route.contains(&node_id.to_string()));
            
            if affected && (path.status == PathStatus::Active || path.status == PathStatus::Degraded) {
                path.status = PathStatus::Failed;
                path.failure_count += 1;
            }
        }
    }
    
    /// Create a redundant path between nodes
    ///
    /// # Arguments
    ///
    /// * `source` - Source node ID
    /// * `destination` - Destination node ID
    /// * `alternatives` - Alternative paths between source and destination
    ///
    /// # Returns
    ///
    /// Result containing the path ID
    ///
    /// # Errors
    ///
    /// * `QFTPError::InsufficientRedundancy` - If there aren't enough alternative paths
    /// * `QFTPError::NodeFailure` - If one of the nodes isn't registered
    pub fn create_redundant_path(&mut self, source: String, destination: String, alternatives: Vec<Vec<String>>) -> Result<String, QFTPError> {
        // Check if we have enough alternative paths
        if alternatives.len() < self.config.redundancy_level {
            return Err(QFTPError::InsufficientRedundancy(
                alternatives.len(),
                self.config.redundancy_level,
            ));
        }
        
        // Check if all nodes in the paths exist
        let mut all_nodes = HashSet::new();
        all_nodes.insert(source.clone());
        all_nodes.insert(destination.clone());
        
        for path in &alternatives {
            for node in path {
                all_nodes.insert(node.clone());
            }
        }
        
        for node in &all_nodes {
            if !self.has_node(node) {
                return Err(QFTPError::NodeFailure(format!("Node not registered: {node}")));
            }
        }
        
        // Create a unique ID for the path
        let path_id = util::generate_id("path");
        
        // Create the redundant path
        let redundant_path = RedundantPath {
            id: path_id.clone(),
            source,
            destination,
            alternatives,
            active_path_index: 0,
            status: PathStatus::Active,
            last_verified: Instant::now(),
            failure_count: 0,
        };
        
        // Store the path
        self.paths.insert(path_id.clone(), redundant_path);
        
        Ok(path_id)
    }
    
    /// Get a redundant path by ID
    #[must_use]
    pub fn get_path(&self, path_id: &str) -> Option<&RedundantPath> {
        self.paths.get(path_id)
    }
    
    /// Verify a path
    ///
    /// # Arguments
    ///
    /// * `path_id` - ID of the path to verify
    ///
    /// # Returns
    ///
    /// `true` if the path is verified as valid, `false` if verification failed
    /// 
    /// # Errors
    /// 
    /// * `QFTPError::PathFailure` - If path cannot be found or verification fails
    pub fn verify_path(&mut self, path_id: &str) -> Result<bool, QFTPError> {
        // First get the path and check if it exists
        let Some(path) = self.paths.get(path_id) else {
            return Err(QFTPError::PathFailure(
                "Unknown path ID".to_string(),
                path_id.to_string(),
            ))
        };
        
        // Cache the path data we need
        let source = path.source.clone();
        let destination = path.destination.clone();
        let active_path_index = path.active_path_index;
        let alternatives = path.alternatives.clone();
        
        // Get the current active path
        if active_path_index >= alternatives.len() {
            // Update the path status
            if let Some(path) = self.paths.get_mut(path_id) {
                path.status = PathStatus::Failed;
            }
            
            return Err(QFTPError::PathFailure(
                source,
                destination,
            ));
        }
        
        let current_path = &alternatives[active_path_index];
        
        // Check each node in the path
        let mut all_nodes_ok = true;
        
        for node_id in current_path {
            if let Some(NodeStatus::Operational | NodeStatus::Degraded) = self.node_status(node_id) {
                // Node is OK
            } else {
                // Node has failed or is unknown
                all_nodes_ok = false;
                break;
            }
        }
        
        // Update the path with results
        if let Some(path) = self.paths.get_mut(path_id) {
            path.last_verified = Instant::now();
            
            if all_nodes_ok {
                if path.status != PathStatus::Active {
                    path.status = PathStatus::Active;
                }
                Ok(true)
            } else {
                // Path has failed, try to switch to an alternative path
                path.status = PathStatus::Failed;
                path.failure_count += 1;
                
                // If auto-recovery is enabled, attempt recovery
                if self.config.auto_recovery {
                    let _ = path; // Release the mutable borrow before calling recover_path
                    self.recover_path(path_id)?;
                    Ok(false)
                } else {
                    Err(QFTPError::PathFailure(
                        source,
                        destination,
                    ))
                }
            }
        } else {
            // This shouldn't happen but handle it just in case
            Err(QFTPError::PathFailure(
                "Path disappeared during verification".to_string(),
                path_id.to_string(),
            ))
        }
    }
    
    /// Attempt to recover a failed path
    ///
    /// # Arguments
    ///
    /// * `path_id` - ID of the path to recover
    ///
    /// # Returns
    ///
    /// Success or failure of recovery attempt
    /// 
    /// # Errors
    /// 
    /// * `QFTPError::PathFailure` - If path cannot be found or recovery fails
    /// * `QFTPError::InvalidNodeID` - If node IDs in the path are invalid
    pub fn recover_path(&mut self, path_id: &str) -> Result<bool, QFTPError> {
        if self.recovery_attempts >= self.config.max_recovery_attempts {
            return Err(QFTPError::RecoveryError(
                "Maximum recovery attempts reached".to_string(),
            ));
        }
        
        // First get the path and check if it exists
        let Some(path) = self.paths.get(path_id) else {
            return Err(QFTPError::PathFailure(
                "Unknown path ID".to_string(),
                path_id.to_string(),
            ))
        };
        
        // Get information we need without borrowing self
        let alternatives = path.alternatives.clone();
        let current_index = path.active_path_index;
        
        // Try to find an alternative path
        let mut found_alternative = false;
        let mut new_index = current_index;
        
        // Check each alternative path
        for (idx, alternative) in alternatives.iter().enumerate() {
            if idx == current_index {
                continue; // Skip current failed path
            }
            
            // Check if all nodes in this alternative are operational
            let all_nodes_ok = alternative.iter().all(|node_id| {
                matches!(self.node_status(node_id), Some(NodeStatus::Operational | NodeStatus::Degraded))
            });
            
            if all_nodes_ok {
                new_index = idx;
                found_alternative = true;
                break;
            }
        }
        
        // Now update path with the new index if we found an alternative
        if found_alternative {
            if let Some(path) = self.paths.get_mut(path_id) {
                path.active_path_index = new_index;
                path.status = PathStatus::Active;
            }
        } else {
            // If no alternative was found, update status to Failed
            if let Some(path) = self.paths.get_mut(path_id) {
                path.status = PathStatus::Failed;
            }
        }
        
        self.last_recovery = Instant::now();
        self.recovery_attempts += 1;
        
        if found_alternative {
            Ok(true)
        } else {
            Err(QFTPError::RecoveryError(
                "No viable alternative paths found".to_string(),
            ))
        }
    }
    
    /// Register an entanglement pair
    pub fn register_entanglement(&mut self, pair_id: String, pair: EntanglementPair) {
        self.entanglement_pairs.insert(pair_id, pair);
    }
    
    /// Get an entanglement pair by ID
    ///
    /// # Arguments
    ///
    /// * `pair_id` - ID of the entanglement pair
    ///
    /// # Returns
    ///
    /// The entanglement pair if found, or None if not found
    #[must_use]
    pub fn get_entanglement(&self, pair_id: &str) -> Option<&EntanglementPair> {
        self.entanglement_pairs.get(pair_id)
    }
    
    /// Detect failures in the system
    pub fn detect_failures(&mut self) -> Vec<(String, NodeStatus)> {
        self.detect_failures_internal(false)
    }
    
    /// Detect failures in the system with option to force detection
    fn detect_failures_internal(&mut self, force: bool) -> Vec<(String, NodeStatus)> {
        let mut detected_failures = Vec::new();
        
        // Skip detection if we're not due for another check and not forcing
        if !force && self.last_detection.elapsed().as_millis() < u128::from(self.config.heartbeat_interval_ms) {
            return detected_failures;
        }
        
        // Check each node based on the detection method
        match self.config.detection_method {
            FailureDetection::Heartbeat => {
                // In a real system, we would check for heartbeat responses
                // For simulation, we'll just check current status
                for (node_id, status) in &self.nodes {
                    if *status == NodeStatus::PartialFailure || *status == NodeStatus::CompleteFailure {
                        detected_failures.push((node_id.clone(), *status));
                    }
                }
            },
            
            FailureDetection::EntanglementVerification => {
                // Check entanglement pairs to detect failures
                let mut node_failure_counts = HashMap::new();
                
                for pair in self.entanglement_pairs.values() {
                    if pair.fidelity < 0.5 {
                        // Increment failure count for both nodes
                        *node_failure_counts.entry(pair.node_a_id.clone()).or_insert(0) += 1;
                        *node_failure_counts.entry(pair.node_b_id.clone()).or_insert(0) += 1;
                    }
                }
                
                // Determine node status based on failure counts
                for (node_id, failure_count) in node_failure_counts {
                    if let Some(status) = self.nodes.get(&node_id) {
                        let total_pairs = self.entanglement_pairs.values()
                            .filter(|p| p.node_a_id == node_id || p.node_b_id == node_id)
                            .count();
                        
                        if total_pairs == 0 {
                            continue;
                        }
                        
                        #[allow(clippy::cast_precision_loss)]
                        let failure_rate = f64::from(failure_count) / total_pairs as f64;
                        
                        let new_status = if (failure_rate - self.config.failure_threshold).abs() > FLOAT_EPSILON && failure_rate > self.config.failure_threshold {
                            if (failure_rate - 0.8).abs() > FLOAT_EPSILON && failure_rate > 0.8 {
                                NodeStatus::CompleteFailure
                            } else {
                                NodeStatus::PartialFailure
                            }
                        } else if (failure_rate - 0.1).abs() > FLOAT_EPSILON && failure_rate > 0.1 {
                            NodeStatus::Degraded
                        } else {
                            NodeStatus::Operational
                        };
                        
                        if *status != new_status {
                            self.update_node_status(&node_id, new_status);
                            detected_failures.push((node_id.to_string(), new_status));
                        }
                    }
                }
            },
            
            FailureDetection::ConsensusBasedDetection => {
                // In a real system, this would involve communication with other nodes
                // For simulation, we'll just report known failures
                for (node_id, status) in &self.nodes {
                    if *status == NodeStatus::PartialFailure || *status == NodeStatus::CompleteFailure {
                        detected_failures.push((node_id.clone(), *status));
                    }
                }
            },
            
            FailureDetection::SyndromeDetection => {
                // In a real system, this would involve error correction syndromes
                // For simulation, we'll just report known failures
                for (node_id, status) in &self.nodes {
                    if *status == NodeStatus::PartialFailure || *status == NodeStatus::CompleteFailure {
                        detected_failures.push((node_id.clone(), *status));
                    }
                }
            },
        }
        
        self.last_detection = Instant::now();
        
        // For each failed node, mark paths as failed
        for (node_id, _) in &detected_failures {
            self.mark_paths_through_node_as_failed(node_id);
        }
        
        detected_failures
    }
    
    /// Register a quantum state for protection
    pub fn protect_state(&mut self, state_id: String, state: QuantumState) {
        self.protected_states.insert(state_id, state);
    }
    
    /// Get a protected quantum state by ID
    ///
    /// # Arguments
    ///
    /// * `state_id` - ID of the protected state
    ///
    /// # Returns
    ///
    /// The protected quantum state if found, or None if not found
    #[must_use]
    pub fn get_protected_state(&self, state_id: &str) -> Option<&QuantumState> {
        self.protected_states.get(state_id)
    }
    
    /// Take a protected quantum state (removes it from protection)
    pub fn take_protected_state(&mut self, state_id: &str) -> Option<QuantumState> {
        self.protected_states.remove(state_id)
    }
    
    /// Monitor system health and attempt to recover from failures
    ///
    /// # Returns
    ///
    /// List of recovered paths, if any
    /// 
    /// # Errors
    /// 
    /// * `QFTPError::PathFailure` - If path recovery fails
    /// * `QFTPError::InvalidNodeID` - If node IDs are invalid
    /// * `QFTPError::ProtocolError` - Protocol-level errors
    pub fn monitor_health(&mut self) -> Result<Vec<String>, QFTPError> {
        // Detect any failures with force=true to bypass the time check
        let failures = self.detect_failures_internal(true);
        
        // Check if any failures require recovery
        let mut recovered_paths = Vec::new();
        
        if !failures.is_empty() && self.config.auto_recovery {
            // Find paths affected by failed nodes
            let affected_paths: Vec<String> = self.paths.iter()
                .filter(|(_, path)| path.status == PathStatus::Failed)
                .map(|(id, _)| id.clone())
                .collect();
            
            // Attempt recovery on each affected path
            for path_id in affected_paths {
                if let Ok(true) = self.recover_path(&path_id) {
                    recovered_paths.push(path_id);
                } else {
                    // Recovery failed, path remains in failed state
                }
            }
        }
        
        Ok(recovered_paths)
    }
    
    /// Get system health metrics
    ///
    /// # Returns
    ///
    /// System health metrics
    #[must_use]
    pub fn health_metrics(&self) -> SystemHealth {
        let total_nodes = self.nodes.len();
        let operational_nodes = self.nodes.values()
            .filter(|&status| *status == NodeStatus::Operational)
            .count();
        
        let degraded_nodes = self.nodes.values()
            .filter(|&status| *status == NodeStatus::Degraded)
            .count();
        
        let failed_nodes = self.nodes.values()
            .filter(|&status| *status == NodeStatus::CompleteFailure || *status == NodeStatus::PartialFailure)
            .count();
        
        let total_paths = self.paths.len();
        let active_paths = self.paths.values()
            .filter(|path| path.status == PathStatus::Active)
            .count();
        
        let degraded_paths = self.paths.values()
            .filter(|path| path.status == PathStatus::Degraded)
            .count();
        
        let failed_paths = self.paths.values()
            .filter(|path| path.status == PathStatus::Failed)
            .count();
        
        SystemHealth {
            total_nodes,
            operational_nodes,
            degraded_nodes,
            failed_nodes,
            total_paths,
            active_paths,
            degraded_paths,
            failed_paths,
            recovery_attempts: self.recovery_attempts,
        }
    }
    
    /// Reset recovery attempts counter
    pub fn reset_recovery_counter(&mut self) {
        self.recovery_attempts = 0;
    }
}

/// System health metrics
#[derive(Debug, Clone, Copy)]
pub struct SystemHealth {
    /// Total number of nodes
    pub total_nodes: usize,
    
    /// Number of fully operational nodes
    pub operational_nodes: usize,
    
    /// Number of degraded nodes
    pub degraded_nodes: usize,
    
    /// Number of failed nodes
    pub failed_nodes: usize,
    
    /// Total number of paths
    pub total_paths: usize,
    
    /// Number of active paths
    pub active_paths: usize,
    
    /// Number of degraded paths
    pub degraded_paths: usize,
    
    /// Number of failed paths
    pub failed_paths: usize,
    
    /// Number of recovery attempts
    pub recovery_attempts: usize,
}

impl SystemHealth {
    /// Returns the percentage of operational nodes
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn node_health_percentage(&self) -> f64 {
        if self.total_nodes == 0 {
            return 100.0;
        }
        
        // Use safer approach to avoid precision loss when converting to f64
        let operational = u32::try_from(self.operational_nodes).unwrap_or(u32::MAX);
        let total = u32::try_from(self.total_nodes).unwrap_or(u32::MAX);
        
        f64::from(operational) / f64::from(total) * 100.0
    }
    
    /// Returns the percentage of active paths
    #[must_use]
    pub fn active_path_percentage(&self) -> f64 {
        if self.total_paths == 0 {
            return 0.0;
        }
        
        self.active_paths as f64 / self.total_paths as f64 * 100.0
    }
    
    /// Returns true if the system meets health thresholds
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        // System is healthy if at least 70% of nodes are operational
        // and at least 80% of paths are active
        self.node_health_percentage() >= 70.0 &&
        self.active_path_percentage() >= 80.0
    }

    /// Get the percentage of active paths (active paths divided by total paths)
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn path_health_percentage(&self) -> f64 {
        if self.total_paths == 0 {
            return 100.0;
        }
        
        self.active_paths as f64 / self.total_paths as f64 * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_node_registration() {
        let mut qftp = QFTP::new();
        
        // Register a node
        qftp.register_node("node1".to_string(), NodeStatus::Operational);
        
        // Check if node exists
        assert!(qftp.has_node("node1"));
        
        // Check node status
        assert_eq!(qftp.node_status("node1"), Some(NodeStatus::Operational));
        
        // Update node status
        qftp.update_node_status("node1", NodeStatus::Degraded);
        assert_eq!(qftp.node_status("node1"), Some(NodeStatus::Degraded));
        
        // Register multiple nodes
        qftp.register_nodes(vec![
            ("node2".to_string(), NodeStatus::Operational),
            ("node3".to_string(), NodeStatus::Operational),
        ]);
        
        assert!(qftp.has_node("node2"));
        assert!(qftp.has_node("node3"));
    }
    
    #[test]
    fn test_path_creation_and_verification() {
        let mut qftp = QFTP::with_config(QFTPConfig {
            redundancy_level: 2,
            ..Default::default()
        });
        
        // Register nodes
        qftp.register_nodes(vec![
            ("A".to_string(), NodeStatus::Operational),
            ("B".to_string(), NodeStatus::Operational),
            ("C".to_string(), NodeStatus::Operational),
            ("D".to_string(), NodeStatus::Operational),
        ]);
        
        // Create a redundant path
        let path_id = qftp.create_redundant_path(
            "A".to_string(),
            "D".to_string(),
            vec![
                vec!["B".to_string()],
                vec!["C".to_string()],
            ],
        ).unwrap();
        
        // Verify the path exists
        let path = qftp.get_path(&path_id).unwrap();
        assert_eq!(path.source, "A");
        assert_eq!(path.destination, "D");
        assert_eq!(path.alternatives.len(), 2);
        
        // Verify path status
        assert_eq!(path.status, PathStatus::Active);
        
        // Verify the path
        let result = qftp.verify_path(&path_id).unwrap();
        assert!(result);
    }
    
    #[test]
    fn test_failure_detection_and_recovery() {
        let mut qftp = QFTP::with_config(QFTPConfig {
            redundancy_level: 2,
            detection_method: FailureDetection::Heartbeat,
            auto_recovery: true,
            heartbeat_interval_ms: 0, // Use 0 to ensure detection always runs
            ..Default::default()
        });
        
        // Register nodes
        qftp.register_nodes(vec![
            ("A".to_string(), NodeStatus::Operational),
            ("B".to_string(), NodeStatus::Operational),
            ("C".to_string(), NodeStatus::Operational),
            ("D".to_string(), NodeStatus::Operational),
        ]);
        
        // Create a redundant path
        let path_id = qftp.create_redundant_path(
            "A".to_string(),
            "D".to_string(),
            vec![
                vec!["B".to_string()],
                vec!["C".to_string()],
            ],
        ).unwrap();
        
        // Mark a node as failed
        qftp.update_node_status("B", NodeStatus::CompleteFailure);
        
        // Detect failures (this should mark the path as failed)
        let failures = qftp.detect_failures();
        println!("Detected failures: {failures:?}");
        
        // Check path status after failure
        let path = qftp.get_path(&path_id).unwrap();
        println!("Path status after failure detection: {:?}", path.status);
        
        // Monitor health (this should recover the path using the alternative)
        let recovered = qftp.monitor_health().unwrap();
        println!("Recovered paths: {recovered:?}");
        
        // Check that the path was recovered
        let path = qftp.get_path(&path_id).unwrap();
        println!("Path status after recovery: {:?}", path.status);
        println!("Path active_path_index: {}", path.active_path_index);
        
        assert_eq!(path.status, PathStatus::Active);
        assert_eq!(path.active_path_index, 1);  // Second alternative
    }
    
    #[test]
    fn test_system_health_metrics() {
        let mut qftp = QFTP::new();
        
        // Register nodes with different statuses
        qftp.register_nodes(vec![
            ("A".to_string(), NodeStatus::Operational),
            ("B".to_string(), NodeStatus::Operational),
            ("C".to_string(), NodeStatus::Degraded),
            ("D".to_string(), NodeStatus::PartialFailure),
            ("E".to_string(), NodeStatus::CompleteFailure),
        ]);
        
        // Get health metrics
        let health = qftp.health_metrics();
        
        assert_eq!(health.total_nodes, 5);
        assert_eq!(health.operational_nodes, 2);
        assert_eq!(health.degraded_nodes, 1);
        assert_eq!(health.failed_nodes, 2);
        
        // Calculate percentages
        assert!((health.node_health_percentage() - 40.0).abs() < FLOAT_EPSILON);
        
        // Check system health
        assert!(!health.is_healthy());
    }
} 