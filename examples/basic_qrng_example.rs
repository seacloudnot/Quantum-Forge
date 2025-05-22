// Basic QRNG Example
//
// This example demonstrates the Quantum Random Number Generator (QRNG) for
// generating high-quality random numbers from quantum sources.

use quantum_protocols::security::qrng::{QRNG, QRNGConfig, EntropySource};
use quantum_protocols::prelude::*;

fn main() -> Result<()> {
    println!("Quantum Random Number Generator (QRNG) Example");
    println!("----------------------------------------------");
    
    // Create a QRNG with default configuration (uses QuantumMeasurement source)
    let mut qrng = QRNG::new_default();
    println!("\n1. Using default configuration");
    println!("   Entropy source: {:?}", qrng.config().entropy_source);
    
    // Generate a series of random bytes
    println!("\n2. Generating 16 random bytes:");
    let random_bytes = qrng.generate_bytes(16)?;
    print!("   ");
    for byte in &random_bytes {
        print!("{byte:02x} ");
    }
    println!();
    
    // Generate random numbers in a range
    println!("\n3. Generating 5 random numbers between 1 and 100:");
    print!("   ");
    for _ in 0..5 {
        let num = qrng.generate_range(100)? + 1; // +1 to get range 1-100
        print!("{num} ");
    }
    println!();
    
    // Assess quantum entropy quality
    println!("\n4. Assessing entropy quality with 1024 sample bytes:");
    let quality = qrng.assess_entropy_quality(1024)?;
    println!("   Overall quality score: {:.4}", quality.quality_score);
    println!("   Is quality acceptable: {}", quality.is_acceptable);
    
    // Print individual test results
    println!("\n5. Individual test results:");
    for result in &quality.test_results {
        println!("   {:?} test: passed={}, score={:.4}, details: {}",
            result.test, result.passed, result.score, result.details);
    }
    
    // Try different entropy sources
    println!("\n6. Testing different entropy sources:");
    
    let sources = [
        EntropySource::SimulatedQuantum,
        EntropySource::QuantumMeasurement,
        EntropySource::Superposition,
        EntropySource::Entanglement,
    ];
    
    for source in &sources {
        // Create a new config with this source
        let config = QRNGConfig {
            entropy_source: *source,
            ..QRNGConfig::default()
        };
        
        let mut source_qrng = QRNG::new(config);
        
        // Generate 8 bytes and measure quality
        let bytes = source_qrng.generate_bytes(8)?;
        let quality = source_qrng.assess_entropy_quality(256)?;
        
        println!("\n   {source:?} Source:");
        print!("   Sample: ");
        for byte in &bytes {
            print!("{byte:02x} ");
        }
        println!();
        println!("   Quality: {:.4}", quality.quality_score);
    }
    
    // Generate a cryptographically secure key with auto-switching
    println!("\n7. Generating a cryptographically secure key (32 bytes):");
    
    // Create a QRNG that can auto-switch to better sources
    let config = QRNGConfig {
        auto_switch_source: true,
        test_randomness: true,
        entropy_source: EntropySource::SimulatedQuantum, // Start with worst source
        min_entropy_quality: 0.8, // Set high quality requirement
        ..QRNGConfig::default()
    };
    
    let mut auto_qrng = QRNG::new(config);
    
    // Generate secure key - should auto-switch to a better source
    let secure_key = auto_qrng.generate_secure_key(32)?;
    
    print!("   Key: ");
    for byte in &secure_key {
        print!("{byte:02x} ");
    }
    println!();
    
    // Check if source was switched
    println!("   Started with: {:?}", EntropySource::SimulatedQuantum);
    println!("   Ended with:   {:?}", auto_qrng.config().entropy_source);
    
    // Get some usage statistics
    let (total_bytes, failures, avg_quality) = auto_qrng.get_stats();
    println!("\n8. QRNG Usage Statistics:");
    println!("   Total bytes generated: {total_bytes}");
    println!("   Test failures: {failures}");
    if let Some(quality) = avg_quality {
        println!("   Average quality: {quality:.4}");
    }
    
    println!("\nExample completed successfully");
    Ok(())
} 