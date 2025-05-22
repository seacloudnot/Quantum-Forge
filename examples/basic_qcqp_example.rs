// examples/basic_qcqp_example.rs
use quantum_protocols::error_correction::{QCQP, QCQPConfig, VerificationMethod};
use quantum_protocols::error_correction::qcqp::ProtectedData;
use quantum_protocols::core::QuantumRegister;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Classical-Quantum Protection Protocol (QCQP) Example");
    println!("===================================================\n");

    // Initialize QCQP with default configuration
    let mut qcqp = QCQP::new_default();
    
    println!("QCQP Configuration:");
    println!("  Verification Method: {:?}", qcqp.config().verification_method);
    println!("  Verification Threshold: {}", qcqp.config().verification_threshold);
    println!("  Verification Rounds: {}", qcqp.config().verification_rounds);
    println!("  Use Quantum Randomness: {}", qcqp.config().use_quantum_randomness);
    println!("  Apply Error Correction: {}", qcqp.config().apply_error_correction);
    println!();
    
    // Create classical data to protect
    println!("Protecting classical data for use in quantum domain...");
    let classical_data = b"Important classical data for quantum processing";
    let protected_classical = qcqp.protect_classical_data(classical_data)?;
    
    println!("  Protected data ID: {}", protected_classical.id);
    println!("  Source domain: {:?}", protected_classical.source_domain);
    println!("  Target domain: {:?}", protected_classical.target_domain);
    println!("  Protection method: {:?}", protected_classical.protection_method);
    println!("  Data size: {} bytes", protected_classical.data.len());
    println!("  Verification data size: {} bytes", protected_classical.verification_data.len());
    println!();
    
    // Verify the protected classical data
    println!("Verifying protected classical data...");
    let verification_result = match qcqp.verify_protected_data(&protected_classical) {
        Ok(result) => result,
        Err(e) => {
            println!("  Verification failed: {}", e);
            return Err(Box::new(e));
        }
    };
    
    println!("  Verification success: {}", verification_result.verified);
    println!("  Confidence score: {:.4}", verification_result.confidence_score);
    println!("  Rounds performed: {}", verification_result.record.rounds_performed);
    println!();
    
    // Create quantum data
    println!("Creating quantum register...");
    let mut register = QuantumRegister::new(3);
    
    // Initialize qubits in superposition
    register.hadamard_all();
    
    // Add entanglement 
    register.cnot(0, 1);
    register.cnot(1, 2);
    
    println!("  Register size: {} qubits", register.size());
    println!("  State: Entangled GHZ-like state");
    println!();
    
    // Protect quantum data for classical domain
    println!("Protecting quantum data for use in classical domain...");
    let metadata = b"Quantum register metadata";
    let protected_quantum = qcqp.protect_quantum_data(&register, metadata)?;
    
    println!("  Protected data ID: {}", protected_quantum.id);
    println!("  Source domain: {:?}", protected_quantum.source_domain);
    println!("  Target domain: {:?}", protected_quantum.target_domain);
    println!("  Protection method: {:?}", protected_quantum.protection_method);
    println!("  Data size: {} bytes", protected_quantum.data.len());
    println!("  Verification data size: {} bytes", protected_quantum.verification_data.len());
    println!();
    
    // Verify the protected quantum data
    println!("Verifying protected quantum data...");
    let quantum_verification = qcqp.verify_protected_data(&protected_quantum)?;
    
    println!("  Verification success: {}", quantum_verification.verified);
    println!("  Confidence score: {:.4}", quantum_verification.confidence_score);
    println!("  Rounds performed: {}", quantum_verification.record.rounds_performed);
    println!();
    
    // Test different verification methods
    println!("Testing different verification methods:");
    
    // Create a custom configuration with Hash verification
    let hash_config = QCQPConfig {
        verification_method: VerificationMethod::Hash,
        ..QCQPConfig::default()
    };
    let mut hash_qcqp = QCQP::new(hash_config);
    let hash_protected = hash_qcqp.protect_classical_data(b"Data for hash verification")?;
    let hash_result = hash_qcqp.verify_protected_data(&hash_protected)?;
    println!("  Hash verification: {}, confidence: {:.4}", 
             hash_result.verified, hash_result.confidence_score);
    
    // Create a custom configuration with Quantum Fingerprint verification
    let qf_config = QCQPConfig {
        verification_method: VerificationMethod::QuantumFingerprint,
        ..QCQPConfig::default()
    };
    let mut qf_qcqp = QCQP::new(qf_config);
    let qf_protected = qf_qcqp.protect_classical_data(b"Data for quantum fingerprint")?;
    let qf_result = qf_qcqp.verify_protected_data(&qf_protected)?;
    println!("  Quantum Fingerprint verification: {}, confidence: {:.4}", 
             qf_result.verified, qf_result.confidence_score);
    
    // Display verification history
    println!("\nVerification History:");
    for (i, record) in qcqp.verification_history().iter().enumerate() {
        println!("  {}. ID: {}, Source: {:?}, Target: {:?}, Verified: {}, Confidence: {:.4}", 
                 i + 1, record.id, record.source_domain, record.target_domain,
                 record.verified, record.confidence_score);
    }
    
    // Create a corrupted version to show error detection
    println!("\nTesting error detection with corrupted data:");
    let mut corrupted_data = protected_classical.data.clone();
    // Modify some bytes to simulate corruption
    if !corrupted_data.is_empty() {
        corrupted_data[0] = corrupted_data[0].wrapping_add(1);
    }
    
    let corrupted_protected = ProtectedData {
        id: protected_classical.id.clone(),
        source_domain: protected_classical.source_domain,
        target_domain: protected_classical.target_domain,
        data: corrupted_data,
        verification_data: protected_classical.verification_data.clone(),
        protection_method: protected_classical.protection_method,
        timestamp: protected_classical.timestamp,
    };
    
    println!("  Verifying corrupted data...");
    match qcqp.verify_protected_data(&corrupted_protected) {
        Ok(result) => {
            println!("  Corruption detected: {}", !result.verified);
            println!("  Confidence score: {:.4}", result.confidence_score);
        },
        Err(e) => {
            println!("  Verification error: {}", e);
        }
    }
    
    // Testing Entanglement Witness verification
    println!("\nTesting Entanglement Witness verification:");
    let ew_config = QCQPConfig {
        verification_method: VerificationMethod::EntanglementWitness,
        ..QCQPConfig::default()
    };
    let mut ew_qcqp = QCQP::new(ew_config);
    let ew_protected = ew_qcqp.protect_quantum_data(&register, b"Data for entanglement witness")?;
    let ew_result = ew_qcqp.verify_protected_data(&ew_protected)?;
    println!("  Entanglement Witness verification: {}, confidence: {:.4}", 
             ew_result.verified, ew_result.confidence_score);
    
    // Testing cross-domain verification
    println!("\nTesting cross-domain verification:");
    let classical_to_quantum = qcqp.protect_classical_data(b"Classical data crossing to quantum domain")?;
    let quantum_to_classical = qcqp.protect_quantum_data(&register, b"Quantum data crossing to classical domain")?;
    
    println!("  C→Q: {} (confidence: {:.4})", 
             qcqp.verify_protected_data(&classical_to_quantum)?.verified,
             qcqp.verify_protected_data(&classical_to_quantum)?.confidence_score);
    println!("  Q→C: {} (confidence: {:.4})", 
             qcqp.verify_protected_data(&quantum_to_classical)?.verified,
             qcqp.verify_protected_data(&quantum_to_classical)?.confidence_score);
    
    println!("\nQCQP Example completed successfully!");
    Ok(())
}