# Quantum Protocols Ecosystem: Usage Guide

This guide provides practical examples of how to use the Quantum Protocols Ecosystem for various applications. Each example includes code snippets that demonstrate key features.

## Table of Contents

1. [Basic Setup](#basic-setup)
2. [Quantum Random Number Generation](#quantum-random-number-generation)
3. [Error Correction](#error-correction)
4. [Fault Tolerance](#fault-tolerance)
5. [Threshold Signatures](#threshold-signatures)
6. [Hardware Integration](#hardware-integration)
7. [Blockchain Integration](#blockchain-integration)
8. [Complete Application Example](#complete-application-example)

## Basic Setup

To get started with the Quantum Protocols Ecosystem, you'll first need to create a protocol bridge that will coordinate all the quantum protocols:

```rust
use quantum_protocols::integration::protocol_bridge::QuantumProtocolBridge;

fn main() -> Result<(), String> {
    // Create the protocol bridge
    let mut bridge = QuantumProtocolBridge::new();
    
    // Protocol bridge is now ready to use
    println!("Protocol bridge initialized");
    
    Ok(())
}
```

## Quantum Random Number Generation

Generating cryptographically secure random numbers using quantum processes:

```rust
use quantum_protocols::integration::protocol_bridge::QuantumProtocolBridge;
use quantum_protocols::security::qrng::QRNGConfig;

fn generate_random_data() -> Result<(), String> {
    // Create protocol bridge
    let mut bridge = QuantumProtocolBridge::new();
    
    // Access the QRNG component
    let mut qrng = bridge.qrng_mut();
    
    // Configure QRNG (optional)
    qrng.set_config(QRNGConfig {
        entropy_sources: 3,
        mixing_rounds: 2,
        ..Default::default()
    });
    
    // Generate 32 bytes of random data
    let random_bytes = qrng.generate_bytes(32)
        .map_err(|e| format!("QRNG error: {}", e))?;
    
    println!("Generated {} random bytes", random_bytes.len());
    
    // Generate a secure key with specific length
    let key = qrng.generate_secure_key(256) // 256-bit key
        .map_err(|e| format!("Key generation error: {}", e))?;
    
    println!("Generated secure key with length: {} bits", key.len() * 8);
    
    Ok(())
}
```

## Error Correction

Protecting quantum states from noise and decoherence:

```rust
use quantum_protocols::core::QuantumState;
use quantum_protocols::error_correction::{CorrectionCode, ErrorCorrection};
use quantum_protocols::error_correction::qecc::{QECC, QECCConfig};

fn protect_quantum_state() -> Result<(), String> {
    // Create a quantum state with 7 qubits
    let mut quantum_state = QuantumState::new(7);
    
    // Apply some operations to the state
    quantum_state.apply_hadamard(0);
    // Add more operations here...
    
    // Create error correction with custom configuration
    let mut qecc = QECC::with_config(QECCConfig {
        auto_correct: true,
        check_frequency: 5,
        error_rate: 0.01,
        syndrome_measurements: 3,
    });
    
    // Set the state to protect
    qecc.set_state(quantum_state.clone());
    
    // Encode with Steane's 7-qubit code
    let encoded = ErrorCorrection::encode(&mut qecc, CorrectionCode::Steane7);
    println!("State encoded: {}", encoded);
    
    // Check if errors detected
    let errors_detected = qecc.check_for_errors()
        .map_err(|e| format!("Error checking failed: {}", e))?;
    
    if errors_detected {
        // Correct errors
        let corrected = qecc.correct_errors()
            .map_err(|e| format!("Error correction failed: {}", e))?;
        
        println!("Errors corrected: {}", corrected);
    }
    
    // Get the protected state
    if let Some(protected_state) = qecc.state() {
        println!("Protected state fidelity: {:.6}", protected_state.fidelity());
    }
    
    Ok(())
}
```

## Fault Tolerance

Setting up fault-tolerant quantum communication channels:

```rust
use quantum_protocols::error_correction::qftp::{QFTP, QFTPConfig, NodeStatus};

fn setup_fault_tolerance() -> Result<(), String> {
    // Create QFTP instance
    let mut qftp = QFTP::with_config(QFTPConfig {
        detection_interval_ms: 1000,
        recovery_threshold: 3,
        ..Default::default()
    });
    
    // Register network nodes
    qftp.register_nodes(vec![
        ("node1".to_string(), NodeStatus::Operational),
        ("node2".to_string(), NodeStatus::Operational),
        ("node3".to_string(), NodeStatus::Operational),
        ("node4".to_string(), NodeStatus::Operational),
        ("node5".to_string(), NodeStatus::Operational),
    ]);
    
    // Create redundant communication paths
    let path_id = qftp.create_redundant_path(
        "node1".to_string(),
        "node5".to_string(),
        vec![
            vec!["node2".to_string()],
            vec!["node3".to_string()],
            vec!["node4".to_string()],
        ],
    ).map_err(|e| format!("Path creation failed: {}", e))?;
    
    println!("Created redundant path with ID: {}", path_id);
    
    // Simulate a node failure
    qftp.update_node_status("node2", NodeStatus::CompleteFailure);
    println!("Node2 failure simulated");
    
    // Verify path and check if it's still operational
    let path_ok = qftp.verify_path(&path_id)
        .map_err(|e| format!("Path verification failed: {}", e))?;
    
    println!("Path operational after node failure: {}", path_ok);
    
    // Get current active path
    let path = qftp.get_path(&path_id).unwrap();
    println!("Current active path index: {}", path.active_path_index);
    
    Ok(())
}
```

## Threshold Signatures

Creating and verifying quantum-resistant threshold signatures:

```rust
use quantum_protocols::consensus::qtsp::{QTSP, QTSPConfig};

fn use_threshold_signatures() -> Result<(), String> {
    // Create QTSP instance for node1
    let mut qtsp = QTSP::with_config("node1", QTSPConfig {
        threshold: 3,
        signature_scheme: "dilithium".to_string(),
        ..Default::default()
    });
    
    // Add authorized nodes to the threshold group
    qtsp.add_authorized_node("node1".to_string());
    qtsp.add_authorized_node("node3".to_string());
    qtsp.add_authorized_node("node4".to_string());
    qtsp.add_authorized_node("node5".to_string());
    
    println!("Set up threshold signature with {} nodes", qtsp.authorized_nodes().len());
    
    // Generate keys
    let key_gen_id = qtsp.init_key_generation()
        .map_err(|e| format!("Key generation failed: {}", e))?;
    
    println!("Generated keys with ID: {}", key_gen_id);
    
    // Sign a message (transaction data)
    let message = b"Transfer 10 QCoins from Alice to Bob";
    
    let signature = qtsp.sign_message(message)
        .map_err(|e| format!("Signing failed: {}", e))?;
    
    println!("Created threshold signature with ID: {}", signature.id);
    println!("Number of signers: {}", signature.signer_node_ids.len());
    
    // Verify the signature
    let verified = qtsp.verify_signature(message, &signature)
        .map_err(|e| format!("Verification failed: {}", e))?;
    
    println!("Signature verification result: {}", verified);
    
    Ok(())
}
```

## Hardware Integration

Connecting to and using quantum hardware (or simulators):

```rust
use quantum_protocols::integration::qhep_hardware::{
    HardwareRegistry,
    HardwareConnectionConfig,
    AuthMethod
};
use std::collections::HashMap;

fn use_quantum_hardware() -> Result<(), String> {
    // Create hardware registry
    let mut registry = HardwareRegistry::new();
    
    // Connect to simulator (always available)
    let simulator_id = registry.connect_hardware(
        "simulator", 
        &HardwareConnectionConfig::default()
    ).map_err(|e| format!("Hardware connection error: {}", e))?;
    
    println!("Connected to quantum simulator: {}", simulator_id);
    
    // Get executor for running circuits
    let executor = registry.get_executor(&simulator_id)
        .map_err(|e| format!("Failed to get executor: {}", e))?;
    
    // Define a quantum circuit (Bell state preparation and measurement)
    let circuit = vec![
        "h 0".to_string(),            // Hadamard on qubit 0
        "cnot 0 1".to_string(),       // CNOT with control 0, target 1
        "measure 0 1".to_string(),    // Measure both qubits
    ];
    
    // Execute the circuit
    let result = executor.execute_circuit(&circuit, &[0, 1])
        .map_err(|e| format!("Circuit execution failed: {}", e))?;
    
    println!("Circuit executed successfully");
    println!("Measurement results: {:?}", result);
    
    // Try to connect to real hardware (if available)
    let ibm_config = HardwareConnectionConfig {
        provider: "ibmq".to_string(),
        endpoint: "https://api.quantum-computing.ibm.com/api".to_string(),
        auth: AuthMethod::ApiKey("YOUR_API_KEY".to_string()),
        timeout_ms: 10000,
        secure: true,
        options: {
            let mut options = HashMap::new();
            options.insert("backend".to_string(), "ibmq_manila".to_string());
            options
        },
    };
    
    match registry.connect_hardware("ibmq", &ibm_config) {
        Ok(id) => {
            println!("Connected to IBM Quantum hardware: {}", id);
            
            // Run the same circuit on real hardware
            let real_executor = registry.get_executor(&id)
                .map_err(|e| format!("Failed to get executor: {}", e))?;
            
            let real_result = real_executor.execute_circuit(&circuit, &[0, 1])
                .map_err(|e| format!("Circuit execution failed: {}", e))?;
            
            println!("Circuit executed on real hardware");
            println!("Measurement results: {:?}", real_result);
        },
        Err(e) => {
            println!("Could not connect to real hardware (expected): {}", e);
            println!("Falling back to simulator");
        }
    }
    
    // Disconnect from all hardware
    registry.disconnect_all()
        .map_err(|e| format!("Disconnect failed: {}", e))?;
    
    Ok(())
}
```

## Blockchain Integration

Using quantum protocols to enhance blockchain security:

```rust
use quantum_protocols::integration::protocol_bridge::QuantumProtocolBridge;
use quantum_protocols::consensus::qtsp::{QTSP, QTSPConfig};
use quantum_protocols::core::QuantumState;
use std::collections::HashMap;

fn quantum_blockchain_integration() -> Result<(), String> {
    // Create protocol bridge
    let mut bridge = QuantumProtocolBridge::new();
    
    // Setup QTSP for threshold signatures
    let mut qtsp = QTSP::with_config("validator1", QTSPConfig {
        threshold: 2,
        ..Default::default()
    });
    
    // Add validators
    qtsp.add_authorized_node("validator1".to_string());
    qtsp.add_authorized_node("validator2".to_string());
    qtsp.add_authorized_node("validator3".to_string());
    
    // Generate keys
    let key_gen_id = qtsp.init_key_generation()
        .map_err(|e| format!("Key generation failed: {}", e))?;
    
    // Create a simple blockchain transaction
    let transaction = format!(
        "{{\"from\":\"user1\",\"to\":\"user2\",\"amount\":10,\"timestamp\":{}}}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
    );
    
    // Sign the transaction with threshold signature
    let signature = qtsp.sign_message(transaction.as_bytes())
        .map_err(|e| format!("Signing failed: {}", e))?;
    
    // Create complete transaction with signature
    let signed_transaction = format!(
        "{{\"transaction\":{},\"signature\":\"{}\"}}",
        transaction,
        signature.id
    );
    
    // Process the transaction through the protocol bridge
    let result = bridge.process_blockchain_transaction(signed_transaction.as_bytes())
        .map_err(|e| format!("Transaction processing failed: {}", e))?;
    
    println!("Transaction processed: {}", result);
    
    // Get processing logs
    for (i, log) in bridge.get_logs().iter().enumerate() {
        println!("Log {}: {}", i+1, log);
    }
    
    Ok(())
}
```

## Complete Application Example

This example shows how to create a complete application that uses multiple quantum protocols together:

```rust
use quantum_protocols::integration::protocol_bridge::QuantumProtocolBridge;
use quantum_protocols::integration::qhep_hardware::{
    HardwareRegistry,
    HardwareConnectionConfig
};
use quantum_protocols::error_correction::{CorrectionCode, ErrorCorrection};
use quantum_protocols::error_correction::qecc::{QECC, QECCConfig};
use quantum_protocols::error_correction::qftp::{QFTP, QFTPConfig, NodeStatus};
use quantum_protocols::consensus::qtsp::{QTSP, QTSPConfig};
use quantum_protocols::core::QuantumState;

fn main() -> Result<(), String> {
    println!("Quantum-Enhanced Secure Transaction System");
    println!("=========================================\n");
    
    // Step 1: Initialize the system
    println!("Step 1: Initializing system components");
    let mut bridge = QuantumProtocolBridge::new();
    let mut registry = HardwareRegistry::new();
    
    // Connect to quantum hardware (simulator in this example)
    let hardware_id = registry.connect_hardware(
        "simulator", 
        &HardwareConnectionConfig::default()
    ).map_err(|e| format!("Hardware connection failed: {}", e))?;
    
    registry.set_default_executor(&hardware_id)
        .map_err(|e| format!("Failed to set default executor: {}", e))?;
    
    println!("✓ Connected to quantum hardware: {}", hardware_id);
    
    // Step 2: Set up the network topology
    println!("\nStep 2: Setting up fault-tolerant network");
    let mut qftp = QFTP::with_config(QFTPConfig::default());
    
    // Register validator nodes
    qftp.register_nodes(vec![
        ("validator1".to_string(), NodeStatus::Operational),
        ("validator2".to_string(), NodeStatus::Operational),
        ("validator3".to_string(), NodeStatus::Operational),
        ("validator4".to_string(), NodeStatus::Operational),
        ("validator5".to_string(), NodeStatus::Operational),
    ]);
    
    // Create redundant communication paths
    let path_id = qftp.create_redundant_path(
        "validator1".to_string(),
        "validator5".to_string(),
        vec![
            vec!["validator2".to_string()],
            vec!["validator3".to_string()],
            vec!["validator4".to_string()],
        ],
    ).map_err(|e| format!("Path creation failed: {}", e))?;
    
    println!("✓ Created fault-tolerant validator network");
    println!("  Network has {} nodes and {} redundant paths", 
             qftp.nodes().len(), 
             qftp.get_path(&path_id).unwrap().alternatives.len());
    
    // Step 3: Set up threshold signature scheme
    println!("\nStep 3: Setting up threshold signatures");
    let mut qtsp = QTSP::with_config("validator1", QTSPConfig {
        threshold: 3,
        ..Default::default()
    });
    
    // Add authorized validators
    qtsp.add_authorized_node("validator1".to_string());
    qtsp.add_authorized_node("validator2".to_string());
    qtsp.add_authorized_node("validator3".to_string());
    qtsp.add_authorized_node("validator4".to_string());
    qtsp.add_authorized_node("validator5".to_string());
    
    // Generate keys
    let key_id = qtsp.init_key_generation()
        .map_err(|e| format!("Key generation failed: {}", e))?;
    
    println!("✓ Created threshold signature scheme");
    println!("  Threshold: 3 out of {} validators", qtsp.authorized_nodes().len());
    println!("  Key ID: {}", key_id);
    
    // Step 4: Prepare quantum states for the transaction
    println!("\nStep 4: Preparing protected quantum states");
    
    // Create and protect a quantum state
    let mut quantum_state = QuantumState::new(7);
    quantum_state.apply_hadamard(0);
    
    // Apply error correction
    let mut qecc = QECC::with_config(QECCConfig {
        auto_correct: true,
        check_frequency: 3,
        error_rate: 0.01,
        syndrome_measurements: 3,
    });
    
    qecc.set_state(quantum_state);
    let encoded = ErrorCorrection::encode(&mut qecc, CorrectionCode::Steane7);
    
    println!("✓ Created protected quantum state");
    println!("  Encoded with Steane's 7-qubit code: {}", encoded);
    
    // Step 5: Generate random data for transaction
    println!("\nStep 5: Generating quantum random data");
    let mut qrng = bridge.qrng_mut();
    let nonce = qrng.generate_bytes(16)
        .map_err(|e| format!("QRNG error: {}", e))?;
    
    println!("✓ Generated random nonce for transaction");
    
    // Step 6: Create and sign a transaction
    println!("\nStep 6: Creating and signing transaction");
    
    // Create transaction data
    let transaction = format!(
        "{{\"from\":\"Alice\",\"to\":\"Bob\",\"amount\":5,\"nonce\":\"{}\"}}",
        hex_encode(&nonce)
    );
    
    // Sign the transaction
    let signature = qtsp.sign_message(transaction.as_bytes())
        .map_err(|e| format!("Signing failed: {}", e))?;
    
    println!("✓ Created threshold signature");
    println!("  Signature ID: {}", signature.id);
    println!("  Number of signers: {}", signature.signer_node_ids.len());
    
    // Step 7: Verify and process the transaction
    println!("\nStep 7: Verifying and processing transaction");
    
    // Verify signature
    let verified = qtsp.verify_signature(transaction.as_bytes(), &signature)
        .map_err(|e| format!("Verification failed: {}", e))?;
    
    println!("✓ Signature verification: {}", verified);
    
    // Process through protocol bridge
    let signed_transaction = format!(
        "{{\"transaction\":{},\"signature\":\"{}\"}}",
        transaction,
        signature.id
    );
    
    let result = bridge.process_blockchain_transaction(signed_transaction.as_bytes())
        .map_err(|e| format!("Transaction processing failed: {}", e))?;
    
    println!("✓ Transaction processed: {}", result);
    
    // Step 8: Clean up resources
    println!("\nStep 8: Cleaning up resources");
    
    registry.disconnect_all()
        .map_err(|e| format!("Disconnect failed: {}", e))?;
    
    println!("✓ System resources cleaned up");
    println!("\nSecure transaction completed successfully!");
    
    Ok(())
}

// Helper function to encode bytes as hex
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
}
```

This example demonstrates a complete workflow for a quantum-enhanced secure transaction system that combines multiple quantum protocols to achieve enhanced security, fault tolerance, and reliability.

## Next Steps

To learn more about specific protocols and advanced features, refer to:

- [ECOSYSTEM.md](ECOSYSTEM.md) - Complete overview of the ecosystem
- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed architecture and design
- No API system has been implemented yet
