//! # Test Utilities Module
//!
//! This module provides common test utilities for quantum protocols, helping:
//!
//! - Set up test network topologies
//! - Create test nodes with predefined configurations
//! - Generate test quantum states
//! - Create simulated entanglement
//! - Verify protocol behavior
//!
//! These utilities are used in test suites throughout the codebase to ensure
//! protocol implementations behave as expected.

use std::sync::Arc;
use tokio::sync::RwLock;

use crate::network::{Node, NetworkTopology};
use crate::network::entanglement::{EntanglementPair, EntanglementPurpose};
use crate::core::QuantumState;

/// Create a basic test network with the specified number of nodes
///
/// This creates a simple linear network topology with nodes named "node1", "node2", etc.
/// and connections between adjacent nodes.
///
/// # Arguments
///
/// * `node_count` - Number of nodes to create (minimum 2)
///
/// # Returns
///
/// A configured `NetworkTopology`
///
/// # Panics
///
/// This function will panic if `node_count` is less than 2.
#[must_use]
pub fn create_test_network(node_count: usize) -> NetworkTopology {
    assert!(node_count >= 2, "Network must have at least 2 nodes");
    
    let mut topology = NetworkTopology::new();
    
    // Create nodes
    for i in 1..=node_count {
        let node_id = format!("node{i}");
        topology.add_node_by_id(node_id);
    }
    
    // Create connections between adjacent nodes
    for i in 1..node_count {
        let node_a = format!("node{i}");
        let node_b = format!("node{}", i + 1);
        topology.add_connection(&node_a, &node_b);
        
        // Set default link quality
        topology.set_link_quality(&node_a, &node_b, 0.95);
    }
    
    topology
}

/// Create a ring network topology with the specified number of nodes
///
/// This creates a network where each node is connected to its adjacent nodes,
/// forming a ring.
///
/// # Arguments
///
/// * `node_count` - Number of nodes in the ring (minimum 3)
///
/// # Returns
///
/// A configured `NetworkTopology` with a ring structure
///
/// # Panics
///
/// This function will panic if `node_count` is less than 3.
#[must_use]
pub fn create_ring_network(node_count: usize) -> NetworkTopology {
    assert!(node_count >= 3, "Ring network must have at least 3 nodes");
    
    let mut topology = create_test_network(node_count);
    
    // Connect the last node to the first to form a ring
    let first_node = "node1".to_string();
    let last_node = format!("node{node_count}");
    topology.add_connection(&first_node, &last_node);
    topology.set_link_quality(&first_node, &last_node, 0.9);
    
    topology
}

/// Create a fully connected network where every node is connected to every other node
///
/// # Arguments
///
/// * `node_count` - Number of nodes to create
///
/// # Returns
///
/// A fully connected `NetworkTopology`
///
/// # Panics
///
/// This function will panic if `node_count` is less than 2.
#[must_use]
pub fn create_fully_connected_network(node_count: usize) -> NetworkTopology {
    assert!(node_count >= 2, "Network must have at least 2 nodes");
    
    let mut topology = NetworkTopology::new();
    
    // Create nodes
    for i in 1..=node_count {
        let node_id = format!("node{i}");
        topology.add_node_by_id(node_id);
    }
    
    // Connect every node to every other node
    for i in 1..=node_count {
        for j in (i+1)..=node_count {
            let node_a = format!("node{i}");
            let node_b = format!("node{j}");
            topology.add_connection(&node_a, &node_b);
            
            // Set a quality based on "distance"
            // Use u32::try_from to safely handle the conversion
            let distance = j - i;
            #[allow(clippy::cast_precision_loss)]
            let distance_factor = match u32::try_from(distance) {
                Ok(dist) => f64::from(dist) * 0.05,
                Err(_) => 0.05 * distance as f64, // Fallback with explicit cast for unrealistically large networks
            };
            let quality = (1.0 - distance_factor).max(0.7);
            topology.set_link_quality(&node_a, &node_b, quality);
        }
    }
    
    topology
}

/// Create a test Node with pre-configured entanglement capabilities
///
/// # Arguments
///
/// * `node_id` - ID for the node
/// * `entanglement_partners` - Node IDs this node can entangle with
///
/// # Returns
///
/// A test Node configured for the specified entanglement partners
#[must_use]
pub fn create_test_node(node_id: &str, entanglement_partners: &[&str]) -> Node {
    let mut node = Node::new(node_id);
    
    // Configure QEP capabilities
    for partner in entanglement_partners {
        node.qep_mut().add_entanglement_capability(partner, 0.9);
    }
    
    node
}

/// Create a wrapped test node for async tests
///
/// # Arguments
///
/// * `node_id` - ID for the node
/// * `entanglement_partners` - Node IDs this node can entangle with
///
/// # Returns
///
/// A test Node wrapped in Arc<`RwLock`<>> for async tests
#[must_use]
pub fn create_wrapped_test_node(node_id: &str, entanglement_partners: &[&str]) -> Arc<RwLock<Node>> {
    let node = create_test_node(node_id, entanglement_partners);
    Arc::new(RwLock::new(node))
}

/// Create simulated entanglement between two nodes
///
/// # Arguments
///
/// * `node_a` - First node
/// * `node_b` - Second node
/// * `purpose` - Purpose of the entanglement
/// * `fidelity` - Fidelity of the entanglement
///
/// # Returns
///
/// An `EntanglementPair` representing the entanglement
#[must_use]
pub fn create_test_entanglement(
    node_a: &str,
    node_b: &str,
    purpose: EntanglementPurpose,
    fidelity: f64
) -> EntanglementPair {
    let mut pair = EntanglementPair::new(node_a, node_b, purpose);
    pair.fidelity = fidelity;
    pair
}

/// Create a test quantum state with specific properties
///
/// # Arguments
///
/// * `qubit_count` - Number of qubits in the state
/// * `initialize_hadamard` - Whether to put qubits in superposition
///
/// # Returns
///
/// A test `QuantumState`
#[must_use]
pub fn create_test_quantum_state(qubit_count: usize, initialize_hadamard: bool) -> QuantumState {
    let mut state = QuantumState::new(qubit_count);
    
    if initialize_hadamard {
        for i in 0..qubit_count {
            state.apply_gate("H", i);
        }
    }
    
    state
}

/// Create a Bell pair state for testing
///
/// Creates an entangled two-qubit state in the Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
///
/// # Returns
///
/// A `QuantumState` representing a Bell pair
#[must_use]
pub fn create_bell_pair() -> QuantumState {
    let mut state = QuantumState::new(2);
    state.apply_gate("H", 0);
    state.apply_gate("CNOT", 0); // Apply CNOT with control qubit 0 and target qubit 1
    state
}

/// Verify entanglement capabilities between test nodes
///
/// # Arguments
///
/// * `node_a` - First node
/// * `node_b` - Second node
///
/// # Returns
///
/// true if the nodes can potentially entangle with each other, false otherwise
#[must_use]
pub fn verify_entanglement_capability(node_a: &Node, node_b: &Node) -> bool {
    let qep = node_a.qep();
    
    // Use the available method to check if nodes can entangle
    qep.can_create_direct_entanglement(node_a.id(), node_b.id())
}

/// Test data structure for quantum network simulation
pub struct TestQNetwork {
    /// Network topology
    pub topology: NetworkTopology,
    
    /// Node instances
    pub nodes: Vec<Arc<RwLock<Node>>>,
}

impl TestQNetwork {
    /// Create a new test quantum network with specified node count
    #[must_use]
    pub fn new(node_count: usize) -> Self {
        let topology = create_test_network(node_count);
        let mut nodes = Vec::new();
        
        // Create wrapped nodes
        for i in 1..=node_count {
            let node_id = format!("node{i}");
            
            // Determine entanglement partners (adjacent nodes)
            let mut partners = Vec::new();
            if i > 1 {
                partners.push(format!("node{}", i-1));
            }
            if i < node_count {
                partners.push(format!("node{}", i+1));
            }
            
            // Convert to &str for create_test_node
            let partners_str: Vec<&str> = partners.iter().map(String::as_str).collect();
            
            // Create node and add to list
            let node = create_test_node(&node_id, &partners_str);
            nodes.push(Arc::new(RwLock::new(node)));
        }
        
        Self { topology, nodes }
    }
    
    /// Create a new test quantum network with a ring topology
    #[must_use]
    pub fn new_ring(node_count: usize) -> Self {
        let topology = create_ring_network(node_count);
        let mut nodes = Vec::new();
        
        // Create wrapped nodes
        for i in 1..=node_count {
            let node_id = format!("node{i}");
            
            // Determine entanglement partners (adjacent nodes in ring)
            let prev = if i > 1 { i-1 } else { node_count };
            let next = if i < node_count { i+1 } else { 1 };
            let partners = [
                format!("node{prev}"),
                format!("node{next}"),
            ];
            
            // Convert to &str for create_test_node
            let partners_str: Vec<&str> = partners.iter().map(String::as_str).collect();
            
            // Create node and add to list
            let node = create_test_node(&node_id, &partners_str);
            nodes.push(Arc::new(RwLock::new(node)));
        }
        
        Self { topology, nodes }
    }
    
    /// Get a node by index
    #[must_use]
    pub fn get_node(&self, index: usize) -> Option<Arc<RwLock<Node>>> {
        if index < self.nodes.len() {
            Some(self.nodes[index].clone())
        } else {
            None
        }
    }
    
    /// Get a node by ID
    #[must_use]
    pub fn get_node_by_id(&self, id: &str) -> Option<Arc<RwLock<Node>>> {
        for node in &self.nodes {
            // Fix the lifetime issue with try_read
            let node_lock = node.try_read().ok()?;
            let node_id = node_lock.id();
            if node_id == id {
                return Some(node.clone());
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_network_topology() {
        let network = create_test_network(5);
        assert_eq!(network.get_all_nodes().len(), 5);
        
        // Test connections
        assert!(network.are_connected("node1", "node2"));
        assert!(network.are_connected("node2", "node3"));
        assert!(!network.are_connected("node1", "node3"));
        
        // Test ring network
        let ring = create_ring_network(4);
        assert!(ring.are_connected("node1", "node4"));
    }
    
    #[test]
    fn test_quantum_state_operations() {
        let state = create_test_quantum_state(2, true);
        
        // Just check if state is valid
        assert_eq!(state.num_qubits(), 2);
    }
    
    #[test]
    fn test_entanglement_operations() {
        let pair = create_test_entanglement("nodeA", "nodeB", EntanglementPurpose::General, 0.95);
        assert_eq!(pair.node_a_id, "nodeA");
        assert_eq!(pair.node_b_id, "nodeB");
        assert!((pair.fidelity - 0.95).abs() < 1e-10, "Expected fidelity to be approximately 0.95");
    }
    
    #[tokio::test]
    async fn test_qstp_transfer() {
        let node_a = create_test_node("nodeA", &["nodeB"]);
        let node_b = create_test_node("nodeB", &["nodeA"]);
        
        // Verify entanglement capability
        assert!(verify_entanglement_capability(&node_a, &node_b));
    }
} 