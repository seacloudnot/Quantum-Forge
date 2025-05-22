// QCAP Capability Levels Example
//
// This example demonstrates how to use the CapabilityLevel feature of QCAP
// and the find_capabilities_by_level method to discover capabilities
// at different levels across the network.

use std::collections::HashMap;
use quantum_protocols::network::qcap::{QCAP, QuantumCapability, CapabilityLevel};
use quantum_protocols::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("QCAP Capability Levels Example");
    println!("------------------------------");
    
    // Create nodes
    let mut node_a = QCAP::new("node-alpha".to_string());
    let mut node_b = QCAP::new("node-beta".to_string());
    let mut node_c = QCAP::new("node-gamma".to_string());
    
    // Register capabilities for node A
    node_a.register_capability(QuantumCapability::new(
        "entanglement".to_string(),
        "Advanced entanglement".to_string(),
        CapabilityLevel::Advanced,
        "3.0".to_string()
    ));
    node_a.register_capability(QuantumCapability::new(
        "qubits".to_string(),
        "Standard qubits".to_string(),
        CapabilityLevel::Standard,
        "2.0".to_string()
    ));
    
    // Register capabilities for node B
    node_b.register_capability(QuantumCapability::new(
        "entanglement".to_string(),
        "Standard entanglement".to_string(),
        CapabilityLevel::Standard,
        "2.0".to_string()
    ));
    node_b.register_capability(QuantumCapability::new(
        "error_correction".to_string(),
        "Experimental error correction".to_string(),
        CapabilityLevel::Experimental,
        "0.9".to_string()
    ));
    
    // Register capabilities for node C
    node_c.register_capability(QuantumCapability::new(
        "entanglement".to_string(),
        "Basic entanglement".to_string(),
        CapabilityLevel::Basic,
        "1.0".to_string()
    ));
    node_c.register_capability(QuantumCapability::new(
        "teleportation".to_string(),
        "Standard teleportation".to_string(),
        CapabilityLevel::Standard,
        "1.5".to_string()
    ));
    
    println!("\nCapabilities registered for each node:");
    println!("  node-alpha: Advanced entanglement, Standard qubits");
    println!("  node-beta: Standard entanglement, Experimental error correction");
    println!("  node-gamma: Basic entanglement, Standard teleportation");
    
    // Exchange capabilities
    println!("\nExchanging capabilities between nodes...");
    node_a.receive_announcement(&node_b.create_announcement());
    node_a.receive_announcement(&node_c.create_announcement());
    
    // Query capabilities by level
    println!("\nExample of searching capabilities by level (from node-alpha's perspective):");
    
    println!("\n1. Capabilities at Basic level or higher:");
    display_capabilities(node_a.find_capabilities_by_level(CapabilityLevel::Basic));
    
    println!("\n2. Capabilities at Standard level or higher:");
    display_capabilities(node_a.find_capabilities_by_level(CapabilityLevel::Standard));
    
    println!("\n3. Capabilities at Advanced level or higher:");
    display_capabilities(node_a.find_capabilities_by_level(CapabilityLevel::Advanced));
    
    println!("\n4. Capabilities at Experimental level:");
    display_capabilities(node_a.find_capabilities_by_level(CapabilityLevel::Experimental));
    
    println!("\nExample completed - CapabilityLevel ordering is:");
    println!("  None < Basic < Standard < Advanced < Experimental");
    
    Ok(())
}

// Display capabilities with the nodes that provide them
fn display_capabilities(capabilities: HashMap<String, Vec<String>>) {
    if capabilities.is_empty() {
        println!("  No capabilities found at this level");
        return;
    }
    
    // Get the capability names and sort them for consistent output
    let mut cap_names: Vec<_> = capabilities.keys().collect();
    cap_names.sort();
    
    for name in cap_names {
        let nodes = &capabilities[name];
        println!("  - {}: provided by {} node(s)", name, nodes.len());
        for node in nodes {
            println!("    * {node}");
        }
    }
} 