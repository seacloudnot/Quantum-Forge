// Basic QEP Example
//
// This example demonstrates how to use the Quantum Entanglement Protocol
// to establish and manage entanglement between nodes in a quantum blockchain network.

use quantum_protocols::prelude::*;
use quantum_protocols::EntanglementProtocol;
use quantum_protocols::network::entanglement::EntanglementPurpose;
use tokio::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Quantum Entanglement Protocol (QEP) Example");
    println!("-------------------------------------------");
    
    // Create a QEP instance to manage entanglement
    let mut qep = QEP::new("entanglement-manager".to_string());
    
    println!("Created entanglement manager: {}", qep.node_id());
    
    // Define our network nodes
    let nodes = vec![
        "node-1".to_string(),
        "node-2".to_string(),
        "node-3".to_string(),
        "node-4".to_string(),
    ];
    
    println!("Network nodes: {nodes:?}");
    
    // Step 1: Create direct entanglement between adjacent nodes
    println!("\nStep 1: Creating direct entanglement between adjacent nodes...");
    
    // Create entanglement between node-1 and node-2
    let pair_1_2 = qep.create_entanglement(
        &nodes[0], &nodes[1], EntanglementPurpose::General
    ).await?;
    
    println!("Created entanglement: {pair_1_2}");
    
    // Create entanglement between node-2 and node-3
    let pair_2_3 = qep.create_entanglement(
        &nodes[1], &nodes[2], EntanglementPurpose::General
    ).await?;
    
    println!("Created entanglement: {pair_2_3}");
    
    // Create entanglement between node-3 and node-4
    let pair_3_4 = qep.create_entanglement(
        &nodes[2], &nodes[3], EntanglementPurpose::General
    ).await?;
    
    println!("Created entanglement: {pair_3_4}");
    
    // Step 2: Check existing entanglement
    println!("\nStep 2: Checking existing entanglement...");
    
    // Check entanglement between node-1 and node-2
    let check_result = qep.check_entanglement(&nodes[0], &nodes[1]).await?;
    println!("Entanglement between {} and {}: {}", 
        nodes[0], nodes[1], 
        check_result.map_or("None".to_string(), |p| format!("{p}"))
    );
    
    // Check non-existent direct entanglement
    let check_result = qep.check_entanglement(&nodes[0], &nodes[3]).await?;
    println!("Entanglement between {} and {}: {}", 
        nodes[0], nodes[3], 
        check_result.map_or("None".to_string(), |p| format!("{p}"))
    );
    
    // Step 3: Perform entanglement swapping to connect non-adjacent nodes
    println!("\nStep 3: Performing entanglement swapping...");
    
    // Swap entanglement to connect node-1 and node-3
    let swap_1_3 = qep.swap_entanglement(&pair_1_2.id, &pair_2_3.id).await?;
    println!("Created swapped entanglement: {swap_1_3}");
    
    // Swap again to connect node-1 and node-4
    let swap_1_4 = qep.swap_entanglement(&swap_1_3.id, &pair_3_4.id).await?;
    println!("Created swapped entanglement: {swap_1_4}");
    
    // Check the new entanglement
    let check_result = qep.check_entanglement(&nodes[0], &nodes[3]).await?;
    println!("Entanglement between {} and {}: {}", 
        nodes[0], nodes[3], 
        check_result.map_or("None".to_string(), |p| format!("{p}"))
    );
    
    // Step 4: Create multiple entanglement pairs and purify them
    println!("\nStep 4: Creating multiple entanglement pairs for purification...");
    
    // Create multiple low-fidelity pairs
    let mut pairs = Vec::new();
    for i in 0..3 {
        let pair = qep.create_entanglement(
            &nodes[0], &nodes[1], EntanglementPurpose::General
        ).await?;
        
        // Artificially reduce fidelity
        if let Some(stored_pair) = qep.entanglement_pairs().get(&pair.id) {
            println!("Created pair {}: {}", i+1, stored_pair);
            pairs.push(pair.id.clone());
        }
    }
    
    // Purify the entanglement
    println!("\nPurifying entanglement...");
    let purified = qep.purify_entanglement(&pairs).await?;
    println!("Purified entanglement: {purified}");
    
    // Step 5: Demonstrate entanglement decoherence and refreshing
    println!("\nStep 5: Demonstrating entanglement decoherence and refreshing...");
    
    // Create a pair with artificially short lifetime
    let mut config = qep.config().clone();
    config.default_lifetime_ms = 500; // 500 ms lifetime
    qep.config_mut().default_lifetime_ms = 500;
    
    let short_pair = qep.create_entanglement(
        &nodes[0], &nodes[1], EntanglementPurpose::Teleportation
    ).await?;
    
    println!("Created short-lived pair: {short_pair}");
    
    // Wait for partial decoherence
    tokio::time::sleep(Duration::from_millis(300)).await;
    
    // Check its state
    if let Some(pair) = qep.get_pair(&short_pair.id) {
        println!("After 300ms: {} (decohered: {})", pair, pair.is_decohered());
        println!("Coherence remaining: {:.1}%", pair.coherence_remaining());
    }
    
    // Refresh the entanglement
    println!("\nRefreshing the entanglement...");
    let refreshed = qep.refresh_entanglement(&short_pair.id).await?;
    println!("Refreshed entanglement: {refreshed}");
    println!("Coherence remaining: {:.1}%", refreshed.coherence_remaining());
    
    // Wait until it fully decoheres
    tokio::time::sleep(Duration::from_millis(600)).await;
    
    // Try to refresh again (should create new pair)
    println!("\nAfter full decoherence, refreshing again...");
    let new_pair = qep.refresh_entanglement(&short_pair.id).await?;
    println!("New pair created: {new_pair}");
    
    // Step 6: Get all entanglement for a node
    println!("\nStep 6: Getting all entanglement for a node...");
    let node_entanglements = qep.get_node_entanglements(&nodes[0]).await?;
    
    println!("Node {} has {} entanglement pairs:", nodes[0], node_entanglements.len());
    for (i, pair) in node_entanglements.iter().enumerate() {
        println!("  {}. {}", i+1, pair);
    }
    
    Ok(())
} 