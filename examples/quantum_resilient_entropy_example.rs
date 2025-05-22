// Quantum Resilient Entropy Example
//
// This example demonstrates how to use the Quantum-Resilient Entropy service
// as part of the QSPP to protect against predictable entropy attacks.

use std::time::Duration;
use quantum_protocols::security::qspp::{
    QSPP, QSPPConfig, ProtectionProfile, ProtectionContext,
    SideChannelAttackType, CountermeasureTechnique
};

fn main() {
    println!("Quantum-Resilient Entropy Protection Example");
    println!("-------------------------------------------");
    
    // Create a custom QSPP instance with high protection
    let config = QSPPConfig {
        protection_level: 4, // Higher protection level
        entropy_mixing_rounds: 5, // More entropy mixing rounds
        min_entropy_sources: 2, // Require at least 2 entropy sources
        ..QSPPConfig::default()
    };
    
    let mut qspp = QSPP::with_config(config);
    
    println!("\nQSPP Configuration:");
    println!("  Protection level: {}", qspp.config().protection_level);
    println!("  Entropy mixing rounds: {}", qspp.config().entropy_mixing_rounds);
    println!("  Minimum entropy sources: {}", qspp.config().min_entropy_sources);
    
    // Register a component that requires entropy protection
    let key_generator = ProtectionProfile {
        component_id: "quantum_key_generator".to_string(),
        component_type: "key_generator".to_string(),
        attack_vectors: vec![
            SideChannelAttackType::PredictableEntropy,
            SideChannelAttackType::Timing
        ],
        required_countermeasures: vec![
            CountermeasureTechnique::QuantumResilientEntropy,
            CountermeasureTechnique::ConstantTime
        ],
        risk_level: 5, // High risk
    };
    
    qspp.register_component(key_generator);
    println!("\nRegistered quantum key generator component for protection");
    
    // Start protection
    qspp.start_protection().expect("Failed to start protection");
    println!("Protection started successfully");
    
    // Examine entropy sources
    if let Some(entropy_service) = qspp.entropy_service() {
        println!("\nActive entropy sources:");
        for source in entropy_service.sources() {
            println!("  - {source:?}");
        }
    }
    
    // Generate secure random bytes with quantum protection
    println!("\nGenerating quantum-resilient random bytes...");
    match qspp.generate_secure_random_bytes(16) {
        Ok(bytes) => {
            println!("Successfully generated {} secure random bytes:", bytes.len());
            print!("  ");
            for byte in &bytes {
                print!("{byte:02x} ");
            }
            println!();
        },
        Err(err) => {
            println!("Error generating secure random bytes: {err}");
        }
    }
    
    // Demonstrate protection against entropy prediction attacks
    println!("\nProtecting key generation against entropy prediction attacks...");
    let result = qspp.protect_against_entropy_prediction(|entropy| {
        // This operation would normally be vulnerable to entropy prediction attacks
        println!("  Performing quantum key generation with resilient entropy");
        
        if let Some(entropy_data) = entropy {
            println!("  Using {} bytes of quantum-resilient entropy", entropy_data.len());
            
            // Simulate key generation using the entropy
            std::thread::sleep(Duration::from_millis(100));
            
            // Return a simulated key
            let mut key = vec![0u8; 8];
            for i in 0..key.len() {
                // Use a safer calculation that won't overflow
                key[i] = entropy_data[i % entropy_data.len()] ^ ((i as u8).wrapping_mul(37));
            }
            key
        } else {
            println!("  Warning: No entropy provided, falling back to less secure method");
            vec![1, 2, 3, 4, 5, 6, 7, 8] // Predictable (insecure) key
        }
    }).expect("Protected operation failed");
    
    println!("\nGenerated key:");
    print!("  ");
    for byte in &result {
        print!("{byte:02x} ");
    }
    println!();
    
    // Create a protection context for context-specific protection
    let context = ProtectionContext {
        component_id: "quantum_key_generator".to_string(),
        operation_type: "key_generation".to_string(),
        attack_vectors: vec![SideChannelAttackType::PredictableEntropy],
        protection_level: 5,
    };
    
    // Use context-specific protection
    println!("\nUsing context-specific protection for quantum operation...");
    qspp.protect_with_context(&context, || {
        println!("  Performing protected quantum operation with full context");
        std::thread::sleep(Duration::from_millis(50));
        // Operation result
        42
    }).expect("Context-specific protection failed");
    
    println!("\nExample completed successfully");
} 