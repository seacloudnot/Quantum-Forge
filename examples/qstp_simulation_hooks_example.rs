// QSTP Simulation Hooks Example
//
// This example demonstrates how to use the QSTP simulation hooks to test
// quantum state transfers under controlled conditions.

use quantum_protocols::core::qstp::{QSTP, QSTPTransport};
use quantum_protocols::simulation::qstp_hooks::{
    QSTPSimulationParams,
    activate_qstp_simulation,
    deactivate_qstp_simulation,
    get_qstp_registry,
    simulate_qstp_transfer
};
use quantum_protocols::test::create_test_quantum_state;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("QSTP Simulation Hooks Example");
    println!("============================");
    
    // Create a test QSTP instance
    let mut qstp = QSTP::new("source_node".to_string());
    
    // Create a test quantum state
    let state = create_test_quantum_state(2, true);
    println!("Created test quantum state with {} qubits", state.num_qubits());
    println!("Initial fidelity: {:.4}", state.fidelity());
    
    let dest_node = "destination_node";
    
    // First, try a normal transfer without simulation
    println!("\nAttempting normal transfer without simulation:");
    match qstp.send_state(&state, dest_node).await {
        Ok(result) => {
            println!("  Transfer successful!");
            println!("  Result fidelity: {:.4}", result.fidelity);
            println!("  Duration: {}ms", result.duration_ms);
        },
        Err(e) => {
            println!("  Transfer failed: {}", e);
        }
    }
    
    // Now activate simulation with custom parameters
    println!("\nActivating QSTP simulation with custom parameters:");
    let params = QSTPSimulationParams {
        success_rate: 0.8,
        fidelity_loss: 0.1,
        latency_ms: 200,
        decoherence_probability: 0.05,
        ..Default::default()
    };
    
    activate_qstp_simulation(Some(params));
    println!("  Simulation activated with:");
    println!("  - Success rate: 80%");
    println!("  - Fidelity loss: 10%");
    println!("  - Latency: 200ms");
    println!("  - Decoherence probability: 5%");
    
    // Try a simulated transfer
    println!("\nAttempting simulated transfer:");
    match simulate_qstp_transfer(&mut qstp, &state, dest_node).await {
        Ok(result) => {
            println!("  Transfer successful!");
            println!("  Result fidelity: {:.4}", result.fidelity);
            println!("  Duration: {}ms", result.duration_ms);
            println!("  Transaction ID: {}", result.transaction_id);
        },
        Err(e) => {
            println!("  Transfer failed: {}", e);
        }
    }
    
    // Retrieve and display events
    let registry = get_qstp_registry();
    let events = {
        let registry = registry.lock().unwrap();
        registry.events().to_vec()
    };
    
    println!("\nSimulation Events:");
    for (i, event) in events.iter().enumerate() {
        println!("  Event {}: {:?} from {} to {}", 
                 i + 1, 
                 event.event_type, 
                 event.source, 
                 event.destination);
        println!("    Original fidelity: {:.4}", event.original_fidelity);
        if let Some(result_fidelity) = event.result_fidelity {
            println!("    Result fidelity: {:.4}", result_fidelity);
        }
        if let Some(error) = &event.error {
            println!("    Error: {}", error);
        }
        println!("    Duration: {}ms", event.duration_ms);
    }
    
    // Deactivate simulation
    deactivate_qstp_simulation();
    println!("\nSimulation deactivated");
    
    Ok(())
} 