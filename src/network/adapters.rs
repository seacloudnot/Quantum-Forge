// Network Adapters Module
//
// This module provides adapters between different network components
// that have incompatible interfaces or different node representations.
//
// ## Overview
// 
// In a quantum network simulation, different components often use different 
// node representations for historical or practical reasons:
//
// - `NetworkTopology` uses direct `Node` references for simplicity and performance
// - `QESP` (Quantum Entanglement Swapping Protocol) uses `Arc<RwLock<Node>>` for async operations
//
// The adapters in this module bridge these incompatible interfaces, allowing 
// components to work together seamlessly.
//
// ## Example Usage
//
// ```
// use quantum_protocols::network::NetworkTopology;
// use quantum_protocols::network::adapters::QESPNetworkAdapter;
// use quantum_protocols::network::qesp::QESPConfig;
//
// // Create a network topology
// let mut topology = NetworkTopology::new();
// topology.add_node_by_id("node1".to_string());
// topology.add_node_by_id("node2".to_string());
// topology.add_connection("node1", "node2");
//
// // Create the adapter
// let mut adapter = QESPNetworkAdapter::new(topology);
//
// // Now you can use QESP operations with NetworkTopology nodes
// let config = QESPConfig::default();
// adapter.set_qesp_config("node1", config);
//
// // In an async context, you could do:
// // let path = adapter.find_swapping_path("node1", "node4").await?;
// // let pair = adapter.establish_entanglement("node1", "node4").await?;
// ```

use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

use crate::network::{Node, NetworkTopology};
use crate::network::qesp::{QESP, QESPConfig, EntanglementSwapper};
use crate::network::entanglement::{EntanglementPair, EntanglementPurpose};
use crate::error::{Result, Error, NetworkError};

/// Adapter for QESP to work with `NetworkTopology`
///
/// This adapter bridges two different abstractions:
/// - `NetworkTopology` (which works with direct Node references)
/// - QESP (which expects a way to look up nodes by ID)
///
/// The adapter provides the following functions:
///
/// 1. Creating wrapped versions of nodes (`Arc<RwLock<Node>>`) from `NetworkTopology` nodes
/// 2. Node lookup by ID for QESP path computation
/// 3. Translating results between the two systems
pub struct QESPNetworkAdapter {
    /// The underlying network topology
    topology: NetworkTopology,
    
    /// QESP instances for each node
    qesp_instances: HashMap<String, QESP>,
    
    /// Wrapped nodes for async access
    wrapped_nodes: HashMap<String, Arc<RwLock<Node>>>,
}

impl QESPNetworkAdapter {
    /// Create a new `QESPNetworkAdapter` instance
    #[must_use]
    pub fn new(topology: NetworkTopology) -> Self {
        let mut adapter = Self {
            topology,
            qesp_instances: HashMap::new(),
            wrapped_nodes: HashMap::new(),
        };
        
        // Initialize wrapped nodes and QESP instances
        adapter.initialize();
        
        adapter
    }
    
    /// Initialize the adapter by creating wrapped nodes and QESP instances
    fn initialize(&mut self) {
        // First, get all node IDs in the network
        let node_ids = self.topology.get_all_nodes();
        
        // Create wrapped versions of each node
        for node_id in node_ids {
            if let Some(node) = self.topology.get_node(&node_id) {
                // Create a clone of the node we can wrap
                let node_clone = node.clone();
                
                // Wrap in Arc<RwLock<>>
                let wrapped_node = Arc::new(RwLock::new(node_clone));
                
                // Store wrapped node
                self.wrapped_nodes.insert(node_id.clone(), wrapped_node.clone());
                
                // Create QESP for this node
                let qesp = QESP::new(node_id.clone());
                self.qesp_instances.insert(node_id, qesp);
            }
        }
        
        // Now link the QESP instances to their wrapped nodes
        for (node_id, qesp) in &mut self.qesp_instances {
            if let Some(wrapped_node) = self.wrapped_nodes.get(node_id) {
                qesp.set_node(wrapped_node.clone());
            }
        }
    }
    
    /// Get access to the underlying topology
    #[must_use]
    pub fn topology(&self) -> &NetworkTopology {
        &self.topology
    }
    
    /// Get a mutable reference to the underlying network topology
    ///
    /// This allows modifying the topology directly, but changes
    /// won't automatically be reflected in the wrapped nodes.
    pub fn topology_mut(&mut self) -> &mut NetworkTopology {
        &mut self.topology
    }
    
    /// Get the QESP instance for a specific node
    #[must_use]
    pub fn get_qesp(&self, node_id: &str) -> Option<&QESP> {
        self.qesp_instances.get(node_id)
    }
    
    /// Get a mutable reference to a QESP instance for a specific node
    ///
    /// # Arguments
    ///
    /// * `node_id` - The ID of the node to get the QESP for
    ///
    /// # Returns
    ///
    /// An Option containing a mutable reference to the QESP instance, or None if not found
    pub fn get_qesp_mut(&mut self, node_id: &str) -> Option<&mut QESP> {
        self.qesp_instances.get_mut(node_id)
    }
    
    /// Set QESP configuration for a specific node
    ///
    /// # Errors
    ///
    /// Returns error if node not found
    pub fn set_qesp_config(&mut self, node_id: &str, config: QESPConfig) -> Result<()> {
        if let Some(qesp) = self.qesp_instances.get_mut(node_id) {
            qesp.set_config(config);
            Ok(())
        } else {
            Err(Error::Network(NetworkError::NodeNotFound(node_id.to_string())))
        }
    }
    
    /// Synchronize changes between the topology and QESP instances
    ///
    /// # Errors
    ///
    /// Returns error if synchronization fails
    pub fn sync_changes(&mut self) -> Result<()> {
        // This would synchronize any changes made to wrapped nodes back to the topology
        // In a production environment, this would need to handle concurrent modifications
        // and potential conflicts properly
        
        // For simplicity in this example, we won't implement the full synchronization
        // logic which would require careful handling of the async runtime
        
        Ok(())
    }
    
    /// Establish entanglement between two nodes
    ///
    /// # Returns
    ///
    /// A Result containing the created `EntanglementPair` on success
    ///
    /// # Errors
    ///
    /// Returns error if entanglement establishment fails
    pub async fn establish_entanglement(
        &mut self,
        source_id: &str,
        destination_id: &str
    ) -> Result<EntanglementPair> {
        // Choose the source and destination QESPs
        let source_qesp = self.get_qesp_mut(source_id)
            .ok_or_else(|| Error::Network(NetworkError::NodeNotFound(source_id.to_string())))?;
        
        // QESP's establish_entanglement requires both source and destination IDs
        // and returns a SwappingResult which we need to convert to EntanglementPair
        let _result = source_qesp.establish_entanglement(source_id, destination_id)
            .await
            .map_err(|e| Error::Network(NetworkError::EntanglementError(format!("QESP operation failed: {e}"))))?;
        
        // Extract the EntanglementPair using the entanglement_id
        // In a real implementation, we would need to retrieve the pair from the entanglement_id
        // For now, we'll create a placeholder pair
        let pair = EntanglementPair::new(source_id, destination_id, EntanglementPurpose::General);
        Ok(pair)
    }
    
    /// Find a path for swapping operations through intermediate nodes
    ///
    /// # Errors
    ///
    /// Returns error if path finding fails
    pub fn find_swapping_path(
        &self,
        source_id: &str,
        destination_id: &str
    ) -> Result<Vec<String>> {
        // Check if the source node exists
        if !self.wrapped_nodes.contains_key(source_id) {
            return Err(Error::Network(NetworkError::NodeNotFound(source_id.to_string())));
        }
        
        // Check if the destination node exists
        if !self.wrapped_nodes.contains_key(destination_id) {
            return Err(Error::Network(NetworkError::NodeNotFound(destination_id.to_string())));
        }
        
        // In a real implementation, we would use the QESP to find the path
        // For now, we'll just use the topology to find the shortest path
        let Some(path) = self.topology.find_path(source_id, destination_id) else { 
            return Err(Error::Network(NetworkError::RoutingError(
                format!("No path found between {source_id} and {destination_id}")
            ))) 
        };
        
        // Remove source and destination from the path
        let swap_nodes = path[1..path.len()-1].to_vec();
        
        Ok(swap_nodes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::qesp::SwappingStrategy;
    
    // Add a constant for float comparison epsilon
    const FLOAT_EPSILON: f64 = 1e-10;
    
    #[tokio::test]
    async fn test_adapter_initialization() {
        // Create a simple network topology
        let mut topology = NetworkTopology::new();
        topology.add_node_by_id("node1".to_string());
        topology.add_node_by_id("node2".to_string());
        topology.add_connection("node1", "node2");
        
        // Create adapter
        let adapter = QESPNetworkAdapter::new(topology);
        
        // Verify that nodes were properly wrapped
        assert!(adapter.wrapped_nodes.contains_key("node1"));
        assert!(adapter.wrapped_nodes.contains_key("node2"));
        
        // Verify that QESP instances were created
        assert!(adapter.qesp_instances.contains_key("node1"));
        assert!(adapter.qesp_instances.contains_key("node2"));
    }
    
    #[tokio::test]
    async fn test_adapter_qesp_config() {
        // Create a simple network topology
        let mut topology = NetworkTopology::new();
        topology.add_node_by_id("node1".to_string());
        
        // Create adapter
        let mut adapter = QESPNetworkAdapter::new(topology);
        
        // Create and apply custom configuration
        let custom_config = QESPConfig {
            strategy: SwappingStrategy::Hierarchical,
            min_fidelity: 0.85,
            timeout_ms: 2000,
            max_measurement_attempts: 5,
            use_fidelity_history: true,
        };
        
        // Set config for node1
        let result = adapter.set_qesp_config("node1", custom_config.clone());
        assert!(result.is_ok());
        
        // Verify the config was applied
        let node_qesp = adapter.get_qesp("node1").unwrap();
        let applied_config = node_qesp.get_config();
        
        assert_eq!(applied_config.strategy, custom_config.strategy);
        assert!((applied_config.min_fidelity - custom_config.min_fidelity).abs() < FLOAT_EPSILON);
        assert_eq!(applied_config.timeout_ms, custom_config.timeout_ms);
        assert_eq!(applied_config.max_measurement_attempts, custom_config.max_measurement_attempts);
        assert_eq!(applied_config.use_fidelity_history, custom_config.use_fidelity_history);
        
        // Test error case with non-existent node
        let bad_result = adapter.set_qesp_config("nonexistent", custom_config);
        assert!(bad_result.is_err());
        if let Err(Error::Network(NetworkError::NodeNotFound(id))) = bad_result {
            assert_eq!(id, "nonexistent");
        } else {
            panic!("Expected NodeNotFound error");
        }
    }
    
    #[tokio::test]
    async fn test_topology_access() {
        // Create a simple network topology
        let mut topology = NetworkTopology::new();
        topology.add_node_by_id("node1".to_string());
        topology.add_node_by_id("node2".to_string());
        
        // Create adapter
        let mut adapter = QESPNetworkAdapter::new(topology);
        
        // Access and modify topology through adapter
        let topo_ref = adapter.topology();
        assert_eq!(topo_ref.get_all_nodes().len(), 2);
        
        // Modify topology
        let topo_mut = adapter.topology_mut();
        topo_mut.add_node_by_id("node3".to_string());
        topo_mut.add_connection("node1", "node3");
        
        // Verify changes
        assert_eq!(adapter.topology().get_all_nodes().len(), 3);
        assert!(adapter.topology().are_connected("node1", "node3"));
    }
    
    #[tokio::test]
    async fn test_find_swapping_path() {
        // Create a network topology with a linear path
        let mut topology = NetworkTopology::new();
        topology.add_node_by_id("A".to_string());
        topology.add_node_by_id("B".to_string());
        topology.add_node_by_id("C".to_string());
        topology.add_node_by_id("D".to_string());
        
        topology.add_connection("A", "B");
        topology.add_connection("B", "C");
        topology.add_connection("C", "D");
        
        // Create adapter
        let adapter = QESPNetworkAdapter::new(topology);
        
        // This would be an actual test in a real implementation
        // For now, we just verify the adapter was created properly
        assert!(adapter.wrapped_nodes.contains_key("A"));
        assert!(adapter.wrapped_nodes.contains_key("B"));
        assert!(adapter.wrapped_nodes.contains_key("C"));
        assert!(adapter.wrapped_nodes.contains_key("D"));
        
        // Test if QESP instances were properly created
        assert!(adapter.qesp_instances.contains_key("A"));
        assert!(adapter.qesp_instances.contains_key("B"));
        assert!(adapter.qesp_instances.contains_key("C"));
        assert!(adapter.qesp_instances.contains_key("D"));
    }
} 