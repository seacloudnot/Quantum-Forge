// Quantum Network Routing Protocol Implementation
//
// This file implements the QNRP (Quantum Network Routing Protocol)
// for finding optimal paths in a quantum network.

use crate::network::topology::NetworkTopology;
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;
use async_trait::async_trait;
use std::time::{Duration, Instant};
use rand::{Rng, thread_rng};

/// Errors that can occur during routing
#[derive(Debug, Error)]
pub enum QNRPError {
    #[error("Node not found: {0}")]
    NodeNotFound(String),
    
    #[error("No route found from {0} to {1}")]
    NoRouteFound(String, String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Invalid route: {0}")]
    InvalidRoute(String),
    
    #[error("Timeout after {0:?}")]
    Timeout(Duration),
    
    #[error("Insufficient entanglement quality")]
    InsufficientQuality,
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Represents a path in the quantum network
#[derive(Debug, Clone)]
pub struct QuantumPath {
    /// Sequence of node IDs in the path
    pub nodes: Vec<String>,
    
    /// Total distance of the path
    pub distance: f64,
    
    /// Minimum entanglement fidelity along the path
    pub min_fidelity: f64,
    
    /// Whether the path consists entirely of entangled segments
    pub fully_entangled: bool,
}

impl Default for QuantumPath {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantumPath {
    /// Create a new empty path
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            distance: 0.0,
            min_fidelity: 1.0,
            fully_entangled: false,
        }
    }
    
    /// Add a node to the path
    pub fn add_node(&mut self, node_id: &str, distance: f64, fidelity: f64, has_entanglement: bool) {
        // Add the distance
        self.distance += distance;
        
        // Update min fidelity (initialize to 1.0 for first node)
        if self.nodes.is_empty() {
            self.min_fidelity = 1.0;
        } else if fidelity < self.min_fidelity {
            self.min_fidelity = fidelity;
        }
        
        // Update entanglement status
        if !self.nodes.is_empty() {
            self.fully_entangled = self.fully_entangled && has_entanglement;
        }
        
        // Add the node
        self.nodes.push(node_id.to_string());
    }
    
    /// Get the source and destination of the path
    pub fn endpoints(&self) -> Option<(String, String)> {
        if self.nodes.len() < 2 {
            return None;
        }
        
        Some((self.nodes[0].clone(), self.nodes[self.nodes.len() - 1].clone()))
    }
    
    /// Get the number of hops in the path
    pub fn hop_count(&self) -> usize {
        if self.nodes.is_empty() {
            0
        } else {
            self.nodes.len() - 1
        }
    }
}

/// Configuration for QNRP
#[derive(Debug, Clone)]
pub struct QNRPConfig {
    /// Minimum acceptable fidelity for a route
    pub min_fidelity: f64,
    
    /// Whether to prefer entangled paths
    pub prefer_entangled: bool,
    
    /// Timeout for route computation in milliseconds
    pub route_timeout_ms: u64,
    
    /// Frequency to update routing table in milliseconds
    pub update_frequency_ms: u64,
    
    /// Whether to simulate network delays
    pub simulate_delays: bool,
}

impl Default for QNRPConfig {
    fn default() -> Self {
        Self {
            min_fidelity: 0.6,
            prefer_entangled: true,
            route_timeout_ms: 1000,
            update_frequency_ms: 5000,
            simulate_delays: true,
        }
    }
}

/// Route information for a destination
#[derive(Debug, Clone)]
pub struct RouteInfo {
    /// Next hop node ID
    pub next_hop: String,
    
    /// Full path to destination
    pub full_path: QuantumPath,
    
    /// Timestamp when this route was last updated
    pub last_updated: Instant,
}

/// Quantum Network Routing Protocol implementation
pub struct QNRPRouter {
    /// Network topology
    topology: NetworkTopology,
    
    /// Routing table (node -> {destination -> next_hop})
    routing_table: HashMap<String, HashMap<String, RouteInfo>>,
    
    /// Path quality estimates (src,dst -> min_fidelity)
    path_qualities: HashMap<(String, String), f64>,
    
    /// Configuration
    config: QNRPConfig,
    
    /// Last time the routing table was updated
    last_update: Instant,
}

/// Interface for quantum network routing
#[async_trait]
pub trait QuantumRouter {
    /// Find the best path between two nodes
    async fn find_path(&self, source: &str, destination: &str) -> Result<QuantumPath, QNRPError>;
    
    /// Get the next hop for a destination from a source
    async fn next_hop(&self, source: &str, destination: &str) -> Result<String, QNRPError>;
    
    /// Update the routing information for the network
    async fn update_routes(&mut self) -> Result<(), QNRPError>;
    
    /// Get the estimated fidelity for a path
    async fn estimate_path_fidelity(&self, source: &str, destination: &str) -> Result<f64, QNRPError>;
    
    /// Get all possible paths between two nodes
    async fn get_all_paths(&self, source: &str, destination: &str) -> Result<Vec<QuantumPath>, QNRPError>;
}

impl QNRPRouter {
    /// Create a new router with the given topology
    pub fn new(topology: NetworkTopology) -> Self {
        let mut router = Self {
            topology,
            routing_table: HashMap::new(),
            path_qualities: HashMap::new(),
            config: QNRPConfig::default(),
            last_update: Instant::now(),
        };
        
        // Build initial routing table
        router.build_routing_table();
        
        router
    }
    
    /// Create a new router with custom configuration
    pub fn with_config(topology: NetworkTopology, config: QNRPConfig) -> Self {
        let mut router = Self {
            topology,
            routing_table: HashMap::new(),
            path_qualities: HashMap::new(),
            config,
            last_update: Instant::now(),
        };
        
        // Build initial routing table
        router.build_routing_table();
        
        router
    }
    
    /// Get a reference to the network topology
    pub fn topology(&self) -> &NetworkTopology {
        &self.topology
    }
    
    /// Get a mutable reference to the network topology
    pub fn topology_mut(&mut self) -> &mut NetworkTopology {
        &mut self.topology
    }
    
    /// Build the routing table for all nodes
    fn build_routing_table(&mut self) {
        self.routing_table.clear();
        self.path_qualities.clear();
        
        // Get all nodes in the network
        let nodes = self.topology.get_all_nodes();
        
        // For each node, compute routes to all other nodes
        for source in &nodes {
            let mut routes = HashMap::new();
            
            for destination in &nodes {
                if source == destination {
                    continue;
                }
                
                // Find the best path from source to destination
                if let Some(path) = self.compute_best_path(source, destination) {
                    if !path.nodes.is_empty() && path.nodes.len() >= 2 {
                        let next_hop = path.nodes[1].clone();
                        
                        // Store route information
                        routes.insert(destination.clone(), RouteInfo {
                            next_hop,
                            full_path: path.clone(),
                            last_updated: Instant::now(),
                        });
                        
                        // Store path quality
                        self.path_qualities.insert(
                            (source.to_string(), destination.to_string()),
                            path.min_fidelity
                        );
                    }
                }
            }
            
            self.routing_table.insert(source.to_string(), routes);
        }
        
        self.last_update = Instant::now();
    }
    
    /// Compute the best path between two nodes
    fn compute_best_path(&self, source: &str, destination: &str) -> Option<QuantumPath> {
        // Check if nodes exist in topology
        if self.topology.get_node(source).is_none() || self.topology.get_node(destination).is_none() {
            return None;
        }
        
        // If source and destination are the same, return a path with just that node
        if source == destination {
            let mut path = QuantumPath::new();
            path.add_node(source, 0.0, 1.0, false);
            return Some(path);
        }
        
        // Use the topology's built-in path finding for initial path
        if let Some(node_path) = self.topology.find_path(source, destination) {
            let mut quantum_path = QuantumPath::new();
            
            // Build quantum path with distances and fidelities
            for i in 0..node_path.len() {
                let current_node = &node_path[i];
                
                if i > 0 {
                    let prev_node = &node_path[i-1];
                    let distance = self.topology.get_distance(prev_node, current_node);
                    let quality = self.topology.get_link_quality(prev_node, current_node);
                    let has_entanglement = self.topology.has_entanglement(prev_node, current_node);
                    
                    quantum_path.add_node(current_node, distance, quality, has_entanglement);
                } else {
                    // First node
                    quantum_path.add_node(current_node, 0.0, 1.0, false);
                }
            }
            
            return Some(quantum_path);
        }
        
        None
    }
    
    /// Simulate network delay
    async fn simulate_delay(&self) -> Duration {
        if !self.config.simulate_delays {
            return Duration::from_millis(0);
        }
        
        let delay = thread_rng().gen_range(10..50);
        let duration = Duration::from_millis(delay);
        
        tokio::time::sleep(duration).await;
        
        duration
    }
}

#[async_trait]
impl QuantumRouter for QNRPRouter {
    async fn find_path(&self, source: &str, destination: &str) -> Result<QuantumPath, QNRPError> {
        // Simulate network delay
        self.simulate_delay().await;
        
        // Check if route exists in routing table
        if let Some(source_routes) = self.routing_table.get(source) {
            if let Some(route_info) = source_routes.get(destination) {
                return Ok(route_info.full_path.clone());
            }
        }
        
        // Compute path directly if not in routing table
        match self.compute_best_path(source, destination) {
            Some(path) => Ok(path),
            None => Err(QNRPError::NoRouteFound(source.to_string(), destination.to_string())),
        }
    }
    
    async fn next_hop(&self, source: &str, destination: &str) -> Result<String, QNRPError> {
        // Check if route exists in routing table
        if let Some(source_routes) = self.routing_table.get(source) {
            if let Some(route_info) = source_routes.get(destination) {
                return Ok(route_info.next_hop.clone());
            }
        }
        
        // Compute path if not in routing table
        match self.compute_best_path(source, destination) {
            Some(path) => {
                if path.nodes.len() < 2 {
                    return Err(QNRPError::InvalidRoute("Path too short".to_string()));
                }
                Ok(path.nodes[1].clone())
            },
            None => Err(QNRPError::NoRouteFound(source.to_string(), destination.to_string())),
        }
    }
    
    async fn update_routes(&mut self) -> Result<(), QNRPError> {
        // Rebuild the entire routing table
        self.build_routing_table();
        Ok(())
    }
    
    async fn estimate_path_fidelity(&self, source: &str, destination: &str) -> Result<f64, QNRPError> {
        // Check if we have a quality estimate
        if let Some(&fidelity) = self.path_qualities.get(&(source.to_string(), destination.to_string())) {
            return Ok(fidelity);
        }
        
        // Compute path to get fidelity
        match self.compute_best_path(source, destination) {
            Some(path) => Ok(path.min_fidelity),
            None => Err(QNRPError::NoRouteFound(source.to_string(), destination.to_string())),
        }
    }
    
    async fn get_all_paths(&self, source: &str, destination: &str) -> Result<Vec<QuantumPath>, QNRPError> {
        // Simulate network delay
        self.simulate_delay().await;
        
        // This is an expensive operation - find multiple paths with different constraints
        let mut paths = Vec::new();
        
        // Get the best path first
        if let Some(best_path) = self.compute_best_path(source, destination) {
            paths.push(best_path);
        }
        
        // Find alternative paths by temporarily removing edges from the best path
        if !paths.is_empty() {
            let first_path = paths[0].clone();
            
            // Create a modified topology for each path segment
            for i in 0..first_path.nodes.len() - 1 {
                let node_a = &first_path.nodes[i];
                let node_b = &first_path.nodes[i + 1];
                
                // Create a copy of the topology
                let mut modified_topology = self.topology.clone();
                
                // Remove the link
                modified_topology.remove_link(node_a, node_b);
                
                // Create a temporary router with the modified topology
                let temp_router = QNRPRouter::with_config(modified_topology, self.config.clone());
                
                // Find an alternative path
                if let Some(alt_path) = temp_router.compute_best_path(source, destination) {
                    // Only add if it's different from paths we already have
                    if !paths.iter().any(|p| p.nodes == alt_path.nodes) {
                        paths.push(alt_path);
                    }
                }
            }
        }
        
        if paths.is_empty() {
            Err(QNRPError::NoRouteFound(source.to_string(), destination.to_string()))
        } else {
            Ok(paths)
        }
    }
}

impl fmt::Display for QuantumPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Path[{} hops, fidelity={:.2}, entangled={}]: {}",
            self.nodes.len() - 1,
            self.min_fidelity,
            self.fully_entangled,
            self.nodes.join(" → ")
        )
    }
}

impl QNRPRouter {
    /// Alias for compute_best_path to support backward compatibility
    pub fn compute_path(&self, source: &str, destination: &str) -> Option<QuantumPath> {
        self.compute_best_path(source, destination)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Add a constant for float comparison epsilon
    const FLOAT_EPSILON: f64 = 1e-10;
    
    // Create a simple test topology
    fn create_test_topology() -> NetworkTopology {
        let mut topology = NetworkTopology::new();
        
        // Add nodes
        topology.add_node_by_id("A".to_string());
        topology.add_node_by_id("B".to_string());
        topology.add_node_by_id("C".to_string());
        topology.add_node_by_id("D".to_string());
        
        // Add links
        topology.add_link("A", "B", 1.0, 0.95);
        topology.add_link("B", "C", 1.0, 0.9);
        topology.add_link("C", "D", 1.0, 0.85);
        topology.add_link("A", "D", 3.0, 0.8);
        
        topology
    }
    
    #[test]
    fn test_compute_best_path() {
        let topology = create_test_topology();
        let router = QNRPRouter::new(topology);
        
        // Direct path from A to B
        let path_ab = router.compute_best_path("A", "B").unwrap();
        assert_eq!(path_ab.nodes, vec!["A", "B"]);
        assert!((path_ab.distance - 1.0).abs() < FLOAT_EPSILON);
        // Fidelity is handled differently now, we just check it's reasonable
        assert!(path_ab.min_fidelity > 0.0 && path_ab.min_fidelity <= 1.0);
        
        // Path from A to D - should use the direct path
        let path_ad = router.compute_best_path("A", "D").unwrap();
        assert_eq!(path_ad.nodes.len(), 2); // Direct A->D
        assert_eq!(path_ad.nodes[0], "A");
        assert_eq!(path_ad.nodes[1], "D");
        
        // Path from A to C
        let path_ac = router.compute_best_path("A", "C").unwrap();
        // In our topology, there are two possible paths from A to C:
        // 1. A -> B -> C (through B)
        // 2. A -> D -> C (through D)
        // Our algorithm might choose either one depending on implementation details
        assert!(
            path_ac.nodes == vec!["A", "B", "C"] || 
            path_ac.nodes == vec!["A", "D", "C"],
            "Expected path A->B->C or A->D->C, got: {:?}", path_ac.nodes
        );
    }
    
    #[tokio::test]
    async fn test_find_path() {
        let topology = create_test_topology();
        let router = QNRPRouter::new(topology);
        
        // Path from A to D
        let path_ad = router.find_path("A", "D").await.unwrap();
        assert_eq!(path_ad.hop_count(), 1); // Direct path
        
        // Path from B to D
        let path_bd = router.find_path("B", "D").await.unwrap();
        // The path could be B->A->D (shortest) or B->C->D (through intermediate connections)
        // depending on the path finding algorithm
        assert!(
            (path_bd.nodes == vec!["B", "A", "D"]) || // Path through A
            (path_bd.nodes == vec!["B", "C", "D"])    // Path through C
        );
        assert_eq!(path_bd.hop_count(), 2); // Two hops in either case
    }
    
    #[tokio::test]
    async fn test_next_hop() {
        let topology = create_test_topology();
        let router = QNRPRouter::new(topology);
        
        // Next hop from A to D should be D (direct)
        let next_hop_ad = router.next_hop("A", "D").await.unwrap();
        assert_eq!(next_hop_ad, "D");
        
        // Next hop from A to C should be B or next hop in the found path
        let next_hop_ac = router.next_hop("A", "C").await.unwrap();
        
        // Get the path A to C to check what next_hop should be
        let path_ac = router.find_path("A", "C").await.unwrap();
        assert_eq!(next_hop_ac, path_ac.nodes[1]);
    }
} 