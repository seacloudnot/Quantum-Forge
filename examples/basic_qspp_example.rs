// Basic QSPP Example
//
// This example demonstrates how to use Quantum Side-channel Protection Protocol (QSPP)
// to protect quantum operations from timing and other side-channel attacks.

use std::thread;
use std::time::Duration;
use quantum_protocols::security::qspp::{
    QSPP, QSPPConfig, ProtectionProfile,
    SideChannelAttackType, CountermeasureTechnique
};

fn main() {
    println!("QSPP (Quantum Side-channel Protection Protocol) Example");
    println!("-----------------------------------------------------");
    
    // Create a QSPP instance with default configuration
    let mut qspp = QSPP::new();
    
    println!("\nDefault protection configuration:");
    println!("  Protection level: {}", qspp.config().protection_level);
    println!("  Detection only mode: {}", qspp.config().detection_only);
    println!("  Detection sensitivity: {}", qspp.config().detection_sensitivity);
    println!("  Maximum timing variation: {}ns", qspp.config().max_timing_variation_ns);
    
    // Register components for protection
    let measurement_component = ProtectionProfile {
        component_id: "quantum_measurement".to_string(),
        component_type: "measurement".to_string(),
        attack_vectors: vec![
            SideChannelAttackType::Timing, 
            SideChannelAttackType::Power
        ],
        required_countermeasures: vec![
            CountermeasureTechnique::ConstantTime,
            CountermeasureTechnique::RandomDelays
        ],
        risk_level: 4,
    };
    
    qspp.register_component(measurement_component);
    println!("\nRegistered quantum measurement component for protection");
    
    // Start protection
    qspp.start_protection().expect("Failed to start protection");
    println!("Protection started");
    
    // Example: Protect a timing-sensitive quantum measurement operation
    println!("\nRunning a protected quantum measurement operation...");
    let result = qspp.protect_quantum_operation(
        "measure",
        "quantum_measurement",
        || {
            // Simulating a quantum measurement that could leak timing information
            println!("  Performing quantum measurement (simulated)");
            thread::sleep(Duration::from_millis(5));
            42 // Result of measurement
        }
    ).expect("Protected operation failed");
    
    println!("Measurement result: {result}");
    
    // Example: Timing attack simulation
    println!("\nSimulating timing analysis under protection...");
    
    // Create a custom configuration with high sensitivity
    let custom_config = QSPPConfig {
        detection_sensitivity: 95,
        max_timing_variation_ns: 100,
        ..Default::default()
    };
    
    // Create a new QSPP instance with the custom configuration
    let mut high_security_qspp = QSPP::with_config(custom_config);
    high_security_qspp.start_protection().expect("Failed to start protection");
    
    // Try to perform an operation with timing variations
    let slow_operation_result = high_security_qspp.protect_timing(|| {
        // This operation takes longer than allowed
        thread::sleep(Duration::from_millis(50));
        "sensitive data"
    });
    
    // The operation might be detected as a timing attack due to high sensitivity
    match slow_operation_result {
        Ok(data) => println!("  Operation completed successfully: {data}"),
        Err(err) => println!("  Attack detected: {err}"),
    }
    
    println!("\nExample completed");
} 