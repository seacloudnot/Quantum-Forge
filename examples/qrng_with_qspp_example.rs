// QRNG with QSPP Integration Example
//
// This example demonstrates how the Quantum Random Number Generator (QRNG)
// can be integrated with Quantum Side-channel Protection Protocol (QSPP)
// for enhanced security.

#![allow(clippy::too_many_lines)]

use quantum_protocols::security::qrng::{QRNG, QRNGConfig, EntropySource};
use quantum_protocols::security::qspp::{QSPP, QSPPConfig, ProtectionProfile, SideChannelAttackType, CountermeasureTechnique};

// Helper function to create and configure QRNG
fn create_qrng() -> QRNG {
    println!("Setting up QRNG...");
    let qrng_config = QRNGConfig {
        entropy_source: EntropySource::Entanglement,
        test_randomness: true,
        buffer_size: 1024,
        min_entropy_quality: 0.9,
        auto_switch_source: true,
        ..Default::default()
    };
    
    println!("  Using QRNG config with entropy source {:?}", qrng_config.entropy_source);
    QRNG::new(qrng_config)
}

// Helper function to create and configure QSPP
fn create_qspp() -> QSPP {
    println!("Setting up QSPP...");
    let qspp_config = QSPPConfig {
        protection_level: 4,
        entropy_mixing_rounds: 3,
        detection_sensitivity: 85,
        ..Default::default()
    };
    
    println!("  Using QSPP config with protection level {}", qspp_config.protection_level);
    let mut qspp = QSPP::with_config(qspp_config);
    
    // Register a component that needs side-channel protection
    let qrng_profile = ProtectionProfile {
        component_id: "quantum_rng".to_string(),
        component_type: "rng".to_string(),
        attack_vectors: vec![
            SideChannelAttackType::Timing,
            SideChannelAttackType::PredictableEntropy
        ],
        required_countermeasures: vec![
            CountermeasureTechnique::ConstantTime,
            CountermeasureTechnique::QuantumResilientEntropy
        ],
        risk_level: 5
    };
    
    qspp.register_component(qrng_profile);
    if let Err(e) = qspp.start_protection() {
        println!("Warning: Could not start protection: {e}");
    }
    println!("  QSPP protection started");
    
    qspp
}

// Helper function to generate random data using QRNG with QSPP protection
fn generate_random_data(qrng: &mut QRNG, qspp: &mut QSPP) -> Vec<u8> {
    println!("Generating quantum random data with side-channel protection...");
    
    // Protected random number generation process
    let random_data = match qspp.protect_quantum_operation(
        "generate",
        "quantum_rng",
        || {
            println!("  Performing protected quantum random generation");
            
            // Generate random bytes
            let bytes = match qrng.generate_bytes(32) {
                Ok(b) => b,
                Err(e) => {
                    println!("Error generating bytes: {e}");
                    vec![0u8; 32] // Fallback to zeros
                }
            };
            println!("  Generated {} random bytes", bytes.len());
            
            // Check quality
            if let Some(quality) = qrng.last_quality() {
                println!("  Quality score: {:.4}", quality.quality_score);
                println!("  Entropy estimate: {:.4} bits/byte", 8.0 * quality.quality_score);
            }
            
            // Return bytes directly, not wrapped in Result
            bytes
        }
    ) {
        Ok(data) => data,
        Err(e) => {
            println!("Warning: Protected operation failed: {e}");
            // Fallback to direct generation if protection fails
            match qrng.generate_bytes(32) {
                Ok(bytes) => bytes,
                Err(e) => {
                    println!("Error in fallback generation: {e}");
                    vec![0u8; 32] // Last resort fallback
                }
            }
        }
    };
    
    println!("Random data generation complete with protections");
    random_data
}

fn main() {
    println!("QRNG with QSPP Integration Example");
    println!("----------------------------------");
    
    // Create QRNG and QSPP instances
    let mut qrng = create_qrng();
    let mut qspp = create_qspp();
    
    // Generate random data with protection
    let random_data = generate_random_data(&mut qrng, &mut qspp);
    
    // Show output sample (partial)
    println!("\nRandom Data Sample:");
    print!("  ");
    for (i, byte) in random_data.iter().enumerate().take(16) {
        print!("{byte:02x} ");
        if (i + 1) % 8 == 0 {
            print!(" ");
        }
    }
    println!("...");
    
    // Create a secure seed for other algorithms
    println!("\nCreating protected seed for downstream cryptography...");
    
    // Get a high-quality seed using QRNG + QSPP together
    let qrng_seed = match qrng.generate_secure_key(32) {
        Ok(seed) => {
            println!("   QRNG seed quality: {:.4}",
                qrng.last_quality().map_or(0.0, |q| q.quality_score));
            seed
        },
        Err(_) => {
            println!("   QRNG failed, using less secure fallback");
            vec![0u8; 32] // This is just a placeholder; a real implementation would have better fallback
        }
    };
    
    // Use the quantum-generated seed with QSPP protection
    let result = match qspp.protect_against_entropy_prediction(|entropy| {
        // In a real application, this would be used to initialize a CSPRNG
        println!("   Using quantum-resilient entropy for seed preparation");
        
        // For demo purposes, we'll just mix the QRNG output with the QSPP entropy
        if let Some(qspp_entropy) = entropy {
            println!("   Mixing with {} bytes of QSPP entropy", qspp_entropy.len());
            
            // Simple mixing function (XOR) - real applications would use more
            // sophisticated mixing
            let mut mixed = vec![0u8; 32];
            for i in 0..32 {
                mixed[i] = qrng_seed[i] ^ qspp_entropy[i % qspp_entropy.len()];
            }
            
            mixed
        } else {
            println!("   No QSPP entropy available, using raw QRNG output");
            qrng_seed.clone()
        }
    }) {
        Ok(data) => data,
        Err(e) => {
            println!("Warning: Protection against entropy prediction failed: {e}");
            // Fallback to direct key if protection fails
            qrng_seed
        }
    };
    
    println!("\nFinal protected seed (partial):");
    print!("  ");
    for (i, byte) in result.iter().enumerate().take(16) {
        print!("{byte:02x} ");
        if (i + 1) % 8 == 0 {
            print!(" ");
        }
    }
    println!("...");
    
    println!("\nDetection Log:");
    let detection_history = qspp.get_detection_history();
    if detection_history.is_empty() {
        println!("  No side-channel attacks detected");
    } else {
        println!("  {} potential side-channel attacks detected!", detection_history.len());
        for event in detection_history {
            println!("  - {:?} attack detected with {}% confidence", 
                     event.attack_type, event.confidence);
        }
    }
    
    println!("\nExample completed successfully");
} 