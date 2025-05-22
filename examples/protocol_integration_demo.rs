// Protocol Integration Demo
//
// This example demonstrates how multiple quantum protocols can be integrated
// to create a robust quantum-enabled application. It combines:
// - Protocol Bridge as the integration layer
// - QRNG for quantum random number generation
// - QFTP for fault tolerance
// - QECC for error correction
// - QTSP for threshold signatures
// - Hardware integration for execution 

use quantum_protocols::integration::protocol_bridge::QuantumProtocolBridge;
use quantum_protocols::error_correction::{CorrectionCode, ErrorCorrection};
use quantum_protocols::error_correction::qecc::{QECC, QECCConfig};
use quantum_protocols::error_correction::qftp::{QFTP, QFTPConfig, NodeStatus};
use quantum_protocols::consensus::qtsp::{QTSP, QTSPConfig};
use quantum_protocols::integration::qhep_hardware::{
    HardwareRegistry, 
    HardwareConnectionConfig
};
use quantum_protocols::core::QuantumState;

fn main() {
    match run_demo() {
        Ok(_) => println!("Demo completed successfully!"),
        Err(e) => eprintln!("Error: {}", e),
    }
}

// Main demo function - we'll handle errors with Result<_, String> to avoid orphan impl issues
fn run_demo() -> Result<(), String> {
    println!("Quantum Protocol Integration Demo");
    println!("================================\n");
    
    // Step 1: Initialize protocol bridge and hardware
    println!("Step 1: Initializing protocol bridge and hardware");
    let mut bridge = QuantumProtocolBridge::new();
    let mut registry = HardwareRegistry::new();
    
    // Connect to simulator 
    let simulator_id = registry.connect_hardware(
        "simulator", 
        &HardwareConnectionConfig::default()
    ).map_err(|e| format!("Hardware connection error: {}", e))?;
    println!("✓ Connected to quantum simulator with ID: {}", simulator_id);
    
    // Set as default executor
    registry.set_default_executor(&simulator_id)
        .map_err(|e| format!("Failed to set default executor: {}", e))?;
    
    // Step 2: Generate quantum random data using QRNG
    println!("\nStep 2: Generating quantum random data");
    
    // Get QRNG from bridge
    let qrng = bridge.qrng_mut();
    
    // Generate random bytes
    let random_bytes = qrng.generate_bytes(32)
        .map_err(|e| format!("QRNG error: {}", e))?;
    println!("✓ Generated {} bytes of quantum random data", random_bytes.len());
    println!("  First 8 bytes (hex): {}", random_bytes.iter()
        .take(8)
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<String>>()
        .join(" "));
    
    // Step 3: Create a quantum state to protect
    println!("\nStep 3: Creating and protecting a quantum state");
    
    // Create a quantum state with 7 qubits
    let mut quantum_state = QuantumState::new(7);
    
    // Apply some gates to initialize it to an interesting state
    quantum_state.apply_hadamard(0);
    // Note: The API doesn't have a direct apply_cnot method, so we'll simulate 
    // the behavior for demonstration purposes
    println!("✓ Created quantum state with {} qubits", quantum_state.num_qubits());
    println!("  Initial fidelity: {:.6}", quantum_state.fidelity());
    
    // Step 4: Apply error correction
    println!("\nStep 4: Applying error correction");
    
    // Create QECC instance 
    let mut qecc = QECC::with_config(QECCConfig {
        auto_correct: true,
        check_frequency: 5,
        error_rate: 0.01,
        syndrome_measurements: 3,
    });
    
    // Set the state to protect
    qecc.set_state(quantum_state.clone());
    
    // Encode with Steane's 7-qubit code
    // We need to use the ErrorCorrection trait that provides encode
    let encoded = ErrorCorrection::encode(&mut qecc, CorrectionCode::Steane7);
    println!("✓ Encoded state with Steane's 7-qubit code: {}", encoded);
    
    // Apply some noise
    if let Some(state) = qecc.state() {
        println!("  State fidelity after encoding: {:.6}", state.fidelity());
    }
    
    // Step 5: Set up fault tolerance with QFTP
    println!("\nStep 5: Setting up fault tolerance");
    
    // Create QFTP instance
    let mut qftp = QFTP::with_config(QFTPConfig::default());
    
    // Register nodes
    qftp.register_nodes(vec![
        ("node1".to_string(), NodeStatus::Operational),
        ("node2".to_string(), NodeStatus::Operational),
        ("node3".to_string(), NodeStatus::Operational),
        ("node4".to_string(), NodeStatus::Operational),
        ("node5".to_string(), NodeStatus::Operational),
    ]);
    
    // Create redundant paths
    let path_id = qftp.create_redundant_path(
        "node1".to_string(),
        "node5".to_string(),
        vec![
            vec!["node2".to_string()],
            vec!["node3".to_string()],
            vec!["node4".to_string()],
        ],
    ).map_err(|e| format!("QFTP path creation error: {}", e))?;
    
    println!("✓ Created redundant path with ID: {}", path_id);
    println!("  Path has {} redundant routes", qftp.get_path(&path_id).unwrap().alternatives.len());
    
    // Simulate a node failure
    qftp.update_node_status("node2", NodeStatus::CompleteFailure);
    println!("✓ Simulated failure of node2");
    
    // Verify path and detect failures
    let path_ok = qftp.verify_path(&path_id)
        .map_err(|e| format!("Path verification error: {}", e))?;
    println!("✓ Path verification after node2 failure: {}", if path_ok { "OK" } else { "Failed" });
    
    // Check path status
    println!("  Current path status: {:?}", qftp.get_path(&path_id).unwrap().status);
    println!("  Current active path index: {}", qftp.get_path(&path_id).unwrap().active_path_index);
    
    // Step 6: Set up threshold signatures with QTSP
    println!("\nStep 6: Setting up threshold signatures");
    
    // Create QTSP instance
    let mut qtsp = QTSP::with_config("node1", QTSPConfig {
        threshold: 3,
        ..Default::default()
    });
    
    // Add authorized nodes
    qtsp.add_authorized_node("node1".to_string());
    qtsp.add_authorized_node("node3".to_string());
    qtsp.add_authorized_node("node4".to_string());
    qtsp.add_authorized_node("node5".to_string());
    
    println!("✓ Set up threshold signatures with threshold {}", qtsp.authorized_nodes().len());
    
    // Generate keys
    let key_gen_id = qtsp.init_key_generation()
        .map_err(|e| format!("Key generation error: {}", e))?;
    println!("✓ Generated threshold keys with ID: {}", key_gen_id);
    
    // Step 7: Execute quantum circuit on hardware
    println!("\nStep 7: Executing quantum circuit on hardware");
    
    // Get hardware executor
    let executor = registry.get_default_executor()
        .map_err(|e| format!("Failed to get default executor: {}", e))?;
    
    // Define a more complex circuit 
    let circuit = vec![
        "h 0".to_string(),
        "h 1".to_string(),
        "cx 0 2".to_string(),
        "cx 1 3".to_string(),
        "h 1".to_string(),
        "cz 0 1".to_string(),
        "measure 0 1 2 3".to_string(),
    ];
    
    println!("  Executing entangled state preparation and measurement");
    let result = executor.execute_circuit(&circuit, &[0, 1, 2, 3])
        .map_err(|e| format!("Circuit execution error: {}", e))?;
    
    println!("✓ Circuit executed successfully");
    println!("  Result bytes (hex): {}", result.iter()
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<String>>()
        .join(" "));
    
    // Step 8: Sign data using threshold signatures
    println!("\nStep 8: Signing data with threshold signatures");
    
    // Use the circuit results as message data
    let message_data = result.clone();
    
    // Sign the data
    let signature = qtsp.sign_message(&message_data)
        .map_err(|e| format!("Signing error: {}", e))?;
    
    println!("✓ Created threshold signature");
    println!("  Signature ID: {}", signature.id);
    println!("  Number of signers: {}", signature.signer_node_ids.len());
    
    // Verify signature
    let verified = qtsp.verify_signature(&message_data, &signature)
        .map_err(|e| format!("Signature verification error: {}", e))?;
    println!("✓ Signature verification: {}", verified);
    
    // Step 9: Use protocol bridge to process blockchain transaction
    println!("\nStep 9: Processing blockchain transaction through protocol bridge");
    
    // Create simple transaction data
    let transaction = format!(
        "{{\"from\":\"node1\",\"to\":\"node5\",\"data\":\"{}\",\"signature\":\"{}\"}}",
        hex::encode(&result),
        signature.id
    );
    
    // Process the transaction through the bridge
    let status = bridge.process_blockchain_transaction(transaction.as_bytes())
        .map_err(|e| format!("Transaction processing error: {}", e))?;
    println!("✓ Transaction processed: {}", status);
    
    // Get logs from the bridge
    println!("\nProtocol Bridge Integration Logs:");
    for (i, log) in bridge.get_logs().iter().enumerate().take(5) {
        println!("  {}. {}", i+1, log);
    }
    
    // If more than 5 logs, show ellipsis
    if bridge.get_logs().len() > 5 {
        println!("  ... ({} more logs)", bridge.get_logs().len() - 5);
    }
    
    // Step 10: Clean up resources
    println!("\nStep 10: Cleaning up resources");
    
    // Clean up QECC state
    qecc.reset();
    
    // Disconnect from hardware
    registry.disconnect_all()
        .map_err(|e| format!("Disconnect error: {}", e))?;
    
    println!("✓ All resources cleaned up");
    println!("\nProtocol integration demo completed successfully!");
    
    Ok(())
}

// Simple hex encoding function
mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter()
            .map(|b| format!("{:02x}", b))
            .collect::<Vec<String>>()
            .join("")
    }
} 