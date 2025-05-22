// Basic QSTP Example
//
// This example demonstrates how to use the Quantum State Transfer Protocol
// to send quantum states between nodes in a quantum blockchain network.

use quantum_protocols::prelude::*;
use quantum_protocols::core::qstp::QSTPTransport;
use tokio::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Quantum State Transfer Protocol (QSTP) Example");
    println!("----------------------------------------------");
    
    // Create two QSTP instances representing different nodes
    let mut node_a = QSTP::new("node-alpha".to_string());
    let node_b = QSTP::new("node-beta".to_string());
    
    println!("Created nodes: {} and {}", node_a.node_id(), node_b.node_id());
    
    // Create a quantum state to transfer
    let mut state = QuantumState::new(3);
    state.set_metadata("Transaction data for block #1234".to_string());
    
    println!("Created quantum state: {state}");
    println!("With {} qubits", state.num_qubits());
    
    // Check if the destination node is available
    println!("\nChecking destination node availability...");
    let is_available = node_a.check_node_availability(node_b.node_id()).await?;
    
    if is_available {
        println!("Destination node is available!");
        
        // Send the state using direct transfer
        println!("\nSending quantum state via direct transfer...");
        let transfer_result = node_a.send_state(&state, node_b.node_id()).await?;
        
        println!("Transfer result: {transfer_result}");
        
        if transfer_result.success {
            println!("Fidelity: {:.2}", transfer_result.fidelity);
            println!("Duration: {} ms", transfer_result.duration_ms);
        }
        
        // Now try teleportation
        println!("\nSending quantum state via teleportation...");
        let teleport_result = node_a.teleport_state(&state, node_b.node_id()).await?;
        
        println!("Teleportation result: {teleport_result}");
        
        if teleport_result.success {
            println!("Fidelity: {:.2}", teleport_result.fidelity);
            println!("Duration: {} ms", teleport_result.duration_ms);
            
            // Compare with direct transfer
            println!("\nComparison:");
            println!("Direct transfer fidelity: {:.2}", transfer_result.fidelity);
            println!("Teleportation fidelity: {:.2}", teleport_result.fidelity);
            println!("Direct transfer time: {} ms", transfer_result.duration_ms);
            println!("Teleportation time: {} ms", teleport_result.duration_ms);
            
            // Determine which is better for this case
            if teleport_result.fidelity > transfer_result.fidelity {
                println!("Teleportation provided better fidelity!");
            } else {
                println!("Direct transfer provided better fidelity!");
            }
            
            if teleport_result.duration_ms < transfer_result.duration_ms {
                println!("Teleportation was faster!");
            } else {
                println!("Direct transfer was faster!");
            }
        }
    } else {
        println!("Destination node is not available!");
    }
    
    // Demonstrate decoherence over time
    println!("\nDemonstrating quantum decoherence:");
    let mut decoherence_state = QuantumState::new(2);
    println!("Initial state: {decoherence_state}");
    println!("Initial fidelity: {:.4}", decoherence_state.fidelity());
    
    // Apply noise to simulate passing time
    for i in 1..=5 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        decoherence_state.apply_decoherence(0.05 * f64::from(i));
        println!(
            "After {} iterations: fidelity = {:.4}", 
            i, 
            decoherence_state.fidelity()
        );
    }
    
    println!("\nIn a real quantum blockchain system:");
    println!("1. States with low fidelity would be rejected by consensus");
    println!("2. Entanglement would be constantly refreshed");
    println!("3. Error correction would be applied to maintain state quality");
    println!("4. Multiple paths would be used for redundancy");
    
    // Apply simulated decoherence to demonstrate fidelity loss
    println!("\nApplying incremental decoherence to demonstrate fidelity loss:");
    let mut decoherence_state = state.clone();
    
    // Get state vectors for fidelity calculation (assuming the function is in scope)
    let original_vec = state.state_vector();
    
    for i in 1..=10 {
        decoherence_state.apply_decoherence(0.05 * f64::from(i));
        let current_vec = decoherence_state.state_vector();
        println!("  After decoherence step {}: Fidelity = {:.4}", 
            i, calculate_fidelity(&original_vec, &current_vec));
    }
    
    Ok(())
} 