#![allow(clippy::too_many_lines)]

// Basic QCAP Example
//
// This example demonstrates the Quantum Capability Announcement Protocol (QCAP)
// which allows nodes to advertise their quantum capabilities to the network.

use std::sync::Arc;
use tokio::sync::RwLock;

use quantum_protocols::prelude::*;
use quantum_protocols::network::qcap::{QCAP, QuantumCapability, CapabilityLevel};
use quantum_protocols::network::Node;

// Create nodes with varying capabilities
async fn create_network_with_capabilities() -> Vec<Arc<RwLock<Node>>> {
    // Create nodes with different capability levels
    let mut nodes = Vec::new();
    
    println!("Creating nodes with different quantum capabilities...");
    
    // Node 1: Advanced node with all capabilities
    let node1 = Arc::new(RwLock::new(Node::new("Node1")));
    {
        let node = node1.read().await;
        let node_id = node.id().to_string();
        
        let mut qcap = QCAP::new(node_id);
        
        // Add advanced capabilities
        qcap.register_capability(QuantumCapability::new(
            "state_preparation".to_string(),
            "Advanced state preparation capability".to_string(),
            CapabilityLevel::Advanced,
            "3.0".to_string()
        ));
        
        qcap.register_capability(QuantumCapability::new(
            "state_transfer".to_string(),
            "Expert state transfer capability".to_string(),
            CapabilityLevel::Experimental,
            "4.0".to_string()
        ));
        
        qcap.register_capability(QuantumCapability::new(
        "entanglement".to_string(),
            "Advanced entanglement capability".to_string(),
        CapabilityLevel::Advanced,
            "3.0".to_string()
        ));
    
        qcap.register_capability(QuantumCapability::new(
        "teleportation".to_string(), 
            "Advanced teleportation capability".to_string(),
            CapabilityLevel::Advanced,
            "3.0".to_string()
        ));
    
        qcap.register_capability(QuantumCapability::new(
            "error_correction".to_string(),
            "Basic error correction capability".to_string(),
            CapabilityLevel::Basic,
            "1.0".to_string()
        ));
        
        qcap.register_capability(QuantumCapability::new(
            "quantum_memory".to_string(),
            "Advanced quantum memory capability".to_string(),
            CapabilityLevel::Advanced,
            "3.0".to_string()
        ));
        
        // Add extended properties for entanglement
        let mut entanglement_cap = qcap.get_capability("entanglement").unwrap().clone();
        entanglement_cap.set_parameter("fidelity".to_string(), "0.98".to_string());
        entanglement_cap.set_parameter("max_pairs".to_string(), "64".to_string());
        entanglement_cap.set_parameter("coherence_time_ms".to_string(), "5000".to_string());
        
        qcap.register_capability(entanglement_cap);
    }
    nodes.push(node1);
    
    // Node 2: Intermediate node with basic capabilities
    let intermediate_node = Arc::new(RwLock::new(Node::new("Node2")));
    {
        let node = intermediate_node.read().await;
        let node_id = node.id().to_string();
        
        let mut qcap = QCAP::new(node_id);
        
        qcap.register_capability(QuantumCapability::new(
            "state_preparation".to_string(),
            "Basic state preparation capability".to_string(),
            CapabilityLevel::Basic,
            "1.0".to_string()
        ));
        
        qcap.register_capability(QuantumCapability::new(
            "state_transfer".to_string(),
            "Basic state transfer capability".to_string(),
            CapabilityLevel::Basic,
            "1.0".to_string()
        ));
        
        qcap.register_capability(QuantumCapability::new(
        "entanglement".to_string(),
            "Basic entanglement capability".to_string(),
        CapabilityLevel::Basic,
        "1.0".to_string()
        ));
    }
    nodes.push(intermediate_node);
    
    // Node 3: Basic node with limited capabilities
    let basic_node = Arc::new(RwLock::new(Node::new("Node3")));
    {
        let node = basic_node.read().await;
        let node_id = node.id().to_string();
        
        let mut qcap = QCAP::new(node_id);
        
        qcap.register_capability(QuantumCapability::new(
            "state_preparation".to_string(),
            "Experimental state preparation capability".to_string(),
            CapabilityLevel::Experimental,
            "4.0".to_string()
        ));
        
        qcap.register_capability(QuantumCapability::new(
            "state_transfer".to_string(),
            "Experimental state transfer capability".to_string(),
            CapabilityLevel::Experimental,
            "4.0".to_string()
        ));
        
        qcap.register_capability(QuantumCapability::new(
            "entanglement".to_string(),
            "Experimental entanglement capability".to_string(),
            CapabilityLevel::Experimental,
            "4.0".to_string()
        ));
        
        qcap.register_capability(QuantumCapability::new(
            "teleportation".to_string(),
            "Experimental teleportation capability".to_string(),
            CapabilityLevel::Experimental,
            "4.0".to_string()
        ));
        
        qcap.register_capability(QuantumCapability::new(
        "error_correction".to_string(),
            "Advanced error correction capability".to_string(),
            CapabilityLevel::Advanced,
            "3.0".to_string()
        ));
        
        qcap.register_capability(QuantumCapability::new(
            "quantum_memory".to_string(),
            "Experimental quantum memory capability".to_string(),
            CapabilityLevel::Experimental,
            "4.0".to_string()
        ));
        
        qcap.register_capability(QuantumCapability::new(
            "quantum_processing".to_string(),
            "Experimental quantum processing capability".to_string(),
        CapabilityLevel::Experimental,
            "4.0".to_string()
        ));
    
        // Add extended properties
        let mut entanglement_cap = qcap.get_capability("entanglement").unwrap().clone();
        entanglement_cap.set_parameter("fidelity".to_string(), "0.995".to_string());
        qcap.register_capability(entanglement_cap);
        
        let mut memory_cap = qcap.get_capability("quantum_memory").unwrap().clone();
        memory_cap.set_parameter("coherence_time_ms".to_string(), "10000".to_string());
        qcap.register_capability(memory_cap);
    }
    nodes.push(basic_node);
    
    // Node 4: Specialized node with specific capabilities
    let specialized_node = Arc::new(RwLock::new(Node::new("Node4")));
    {
        let node = specialized_node.read().await;
        let node_id = node.id().to_string();
        
        let mut qcap = QCAP::new(node_id);
        
        qcap.register_capability(QuantumCapability::new(
            "state_preparation".to_string(),
            "Basic state preparation capability".to_string(),
            CapabilityLevel::Basic,
            "1.0".to_string()
        ));
    }
    nodes.push(specialized_node);
    
    // Node 5: Minimal node with very limited capabilities
    let minimal_node = Arc::new(RwLock::new(Node::new("Node5")));
    {
        let node = minimal_node.read().await;
        let node_id = node.id().to_string();
        
        let mut qcap = QCAP::new(node_id);
        
        qcap.register_capability(QuantumCapability::new(
            "state_preparation".to_string(),
            "Basic state preparation capability".to_string(),
            CapabilityLevel::Basic,
        "1.0".to_string()
        ));
    }
    nodes.push(minimal_node);
    
    nodes
}

// Helper function to find nodes with a specific capability level
fn find_nodes_with_capability(
    nodes: &[Arc<RwLock<Node>>], 
    capability: &str, 
    min_level: CapabilityLevel
) -> Vec<String> {
    let mut nodes_with_capability = Vec::new();
    
    // Use asynchronous code inside a blocking context
    let runtime = tokio::runtime::Runtime::new().unwrap();
    
    // For each node, check if it has the required capability at the specified level
    for node_arc in nodes {
        runtime.block_on(async {
            let node = node_arc.read().await;
            let node_id = node.id().to_string();
    
            // Create a QCAP instance to use for capability checking
            let qcap = QCAP::new(node_id.clone());
            
            // For this example, create a dummy capability to check
            if let Some(cap) = qcap.get_capability(capability) {
                if cap.level >= min_level {
                    nodes_with_capability.push(node_id);
                }
            }
        });
    }
    
    nodes_with_capability
}

// Helper function to find nodes compatible with a protocol
fn find_compatible_nodes_for(
    nodes: &[Arc<RwLock<Node>>],
    protocol: &str,
    requirements: &[(&str, CapabilityLevel)]
) -> Vec<String> {
    let mut compatible_nodes = Vec::new();
    
    // Use asynchronous code inside a blocking context
    let runtime = tokio::runtime::Runtime::new().unwrap();
    
    // For each node, check if it meets all requirements
    for node_arc in nodes {
        runtime.block_on(async {
            let node = node_arc.read().await;
            let node_id = node.id().to_string();
            
            // Create a QCAP instance to use for capability checking
            let qcap = QCAP::new(node_id.clone());
    
            // Check if the node meets all requirements
            let meets_requirements = requirements
                .iter()
                .all(|(cap_name, min_level)| {
                    if let Some(cap) = qcap.get_capability(cap_name) {
                        cap.level >= *min_level
                    } else {
                        false
                    }
                });
            
            if meets_requirements {
                compatible_nodes.push(node_id);
            }
        });
    }
    
    // Print results
    if compatible_nodes.is_empty() {
        println!("No nodes compatible with {protocol} found");
    } else {
        println!("Nodes compatible with {protocol}:");
        for node in &compatible_nodes {
            println!("    * {node}");
        }
    }
    
    compatible_nodes
    }
    
// Use tokio to run the async main function
#[tokio::main]
async fn main() -> Result<()> {
    println!("Quantum Capability Announcement Protocol (QCAP) Example");
    println!("-----------------------------------------------------");
    
    let nodes = create_network_with_capabilities().await;

    println!("\nPrinting detailed node capabilities:");
    
    // Print detailed capabilities for each node
    for node_arc in &nodes {
        let node = node_arc.read().await;
        let node_id = node.id().to_string();
        let qcap = QCAP::new(node_id);
        
        println!("\n{}:", node.id());
        // Since we can't iterate over capabilities directly in this example,
        // we'll check for known capability types
        for cap_name in &["state_preparation", "state_transfer", "entanglement", 
                         "teleportation", "error_correction", "quantum_memory", 
                         "quantum_processing"] {
            if let Some(cap) = qcap.get_capability(cap_name) {
                println!("  * {}: {:?}", cap_name, cap.level);
                
                // Print extended properties if any
                if !cap.parameters.is_empty() {
                    println!("    Parameters:");
                    for (key, value) in &cap.parameters {
                        println!("    - {key}: {value}");
                    }
                }
            }
        }
    }
    
    // Find nodes with specific capabilities
    println!("\nSearching for nodes with specific capabilities:");
    
    let teleportation_nodes = find_nodes_with_capability(
        &nodes, 
        "teleportation", 
        CapabilityLevel::Basic
    );
    println!("Nodes with teleportation (Basic or better): {teleportation_nodes:?}");
    
    let entanglement_nodes = find_nodes_with_capability(
        &nodes, 
        "entanglement", 
        CapabilityLevel::Basic
    );
    println!("Nodes with entanglement (Basic or better): {entanglement_nodes:?}");
    
    let advanced_entanglement_nodes = find_nodes_with_capability(
        &nodes, 
        "entanglement", 
        CapabilityLevel::Advanced
    );
    println!("Nodes with entanglement (Advanced or better): {advanced_entanglement_nodes:?}");
    
    // Find nodes compatible with specific protocols
    println!("\nFinding nodes compatible with specific protocols:");
    
    // Find nodes compatible with QBFT
    let qbft_requirements = vec![
        ("entanglement", CapabilityLevel::Advanced),
        ("state_transfer", CapabilityLevel::Advanced),
        ("error_correction", CapabilityLevel::Basic),
    ];
    find_compatible_nodes_for(&nodes, "QBFT", &qbft_requirements);
    
    // Find nodes compatible with advanced QKD
    let qkd_requirements = vec![
        ("state_preparation", CapabilityLevel::Advanced),
        ("state_transfer", CapabilityLevel::Basic),
    ];
    find_compatible_nodes_for(&nodes, "Advanced QKD", &qkd_requirements);
    
    // Find nodes compatible with quantum computing
    let qc_requirements = vec![
        ("quantum_processing", CapabilityLevel::Basic),
        ("quantum_memory", CapabilityLevel::Basic),
    ];
    find_compatible_nodes_for(&nodes, "Quantum Computing", &qc_requirements);
    
    println!("\nExample completed successfully");
    
    Ok(())
} 