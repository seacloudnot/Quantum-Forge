// QESP with Adapter Example
//
// This example demonstrates how to use the QESPNetworkAdapter to bridge
// between NetworkTopology and QESP components, enabling proper entanglement
// swapping in a quantum network.

use quantum_protocols::network::NetworkTopology;
use quantum_protocols::network::adapters::QESPNetworkAdapter;
use quantum_protocols::network::qesp::{QESPConfig, SwappingStrategy};
use quantum_protocols::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=================================================");
    println!("  QESP with Network Adapter Example");
    println!("=================================================");
    
    // 1. Create a network topology with nodes arranged in a line:
    // node_a --- node_b --- node_c --- node_d
    println!("\nCreating network topology with 4 nodes in a line...");
    let topology = create_network_topology();
    
    println!("Network configuration:");
    println!("  node_a <--> node_b <--> node_c <--> node_d");
    println!("  (only adjacent nodes have direct connections)");
    
    // 2. Create the adapter that bridges NetworkTopology and QESP
    println!("\nCreating QESPNetworkAdapter to bridge NetworkTopology and QESP...");
    let mut adapter = QESPNetworkAdapter::new(topology);
    
    // 3. Configure the QESP for each node
    println!("\nConfiguring QESP for each node...");
    
    // Use different strategies for demonstration
    let node_a_config = QESPConfig {
        strategy: SwappingStrategy::Sequential,
        min_fidelity: 0.7,
        timeout_ms: 5000,
        max_measurement_attempts: 3,
        use_fidelity_history: true,
    };
    
    let node_b_config = QESPConfig {
        strategy: SwappingStrategy::Hierarchical,
        ..node_a_config.clone()
    };
    
    let node_c_config = QESPConfig {
        strategy: SwappingStrategy::FidelityAdaptive,
        ..node_a_config.clone()
    };
    
    // Apply configurations
    adapter.set_qesp_config("node_a", node_a_config)?;
    adapter.set_qesp_config("node_b", node_b_config)?;
    adapter.set_qesp_config("node_c", node_c_config)?;
    
    // 4. Find a swapping path between non-adjacent nodes
    println!("\nFinding swapping path from node_a to node_d...");
    
    // In a real application, we would execute this, but for demonstration:
    println!("Would execute: adapter.find_swapping_path(\"node_a\", \"node_d\").await");
    println!("Expected path: node_a -> node_b -> node_c -> node_d");
    
    // 5. Establish entanglement between non-adjacent nodes
    println!("\nEstablishing entanglement between node_a and node_d...");
    
    // In a real application, we would execute this, but for demonstration:
    println!("Would execute: adapter.establish_entanglement(\"node_a\", \"node_d\").await");
    println!("This would:");
    println!("1. Use the QESP for node_a");
    println!("2. Find a path through intermediate nodes (node_b, node_c)");
    println!("3. Create entanglement pairs between adjacent nodes");
    println!("4. Perform Bell measurements at intermediate nodes");
    println!("5. Create end-to-end entanglement between node_a and node_d");
    
    // 6. Demonstrate how the adapter synchronizes changes
    println!("\nSynchronizing changes between wrapped nodes and network topology...");
    
    // In a real application, we would execute this, but for demonstration:
    println!("Would execute: adapter.sync_changes().await");
    println!("This would update the NetworkTopology with any changes made through the QESP");
    
    // 7. Explain the advantages of using the adapter
    println!("\n=================================================");
    println!("  Benefits of the Adapter Pattern");
    println!("=================================================");
    println!("1. Separation of concerns: NetworkTopology manages the network structure");
    println!("   while QESP handles entanglement swapping operations");
    println!("2. Compatibility: Bridges between different node representations");
    println!("   (Node vs Arc<RwLock<Node>>)");
    println!("3. Unified API: Provides a consistent interface for entanglement");
    println!("   operations across the network");
    println!("4. Maintainability: Changes to either component can be made");
    println!("   without affecting the other");
    
    println!("\nExample completed successfully");
    Ok(())
}

/// Create a sample network topology
fn create_network_topology() -> NetworkTopology {
    let mut topology = NetworkTopology::new();
    
    // Add nodes
    topology.add_node_by_id("node_a".to_string());
    topology.add_node_by_id("node_b".to_string());
    topology.add_node_by_id("node_c".to_string());
    topology.add_node_by_id("node_d".to_string());
    
    // Add connections between adjacent nodes only
    topology.add_connection("node_a", "node_b");
    topology.add_connection("node_b", "node_c");
    topology.add_connection("node_c", "node_d");
    
    // Set link qualities
    topology.set_link_quality("node_a", "node_b", 0.95);
    topology.set_link_quality("node_b", "node_c", 0.90);
    topology.set_link_quality("node_c", "node_d", 0.92);
    
    topology
} 