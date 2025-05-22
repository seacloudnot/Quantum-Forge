use quantum_protocols::{
    benchmark::{
        get_profiler, visualization::{create_performance_dashboard, export_to_csv},
        OperationType,
    },
    consensus::{qpcp::QPCP, qtsp::QTSP, qvcp::QVCP},
    core::{QuantumRegister, QuantumState, Qubit},
    error_correction::{qcqp::QCQP, qecc::QECC, CorrectionCode, qftp::QFTP},
    integration::qdap::{EncodingScheme, QDAP},
    security::qrng::QRNG,
    // Import needed traits
    error_correction::ErrorCorrection, consensus::ConsensusProtocol,
};

use std::fs::create_dir_all;
use std::path::Path;

fn main() {
    // Create benchmarks directory if it doesn't exist
    let benchmark_dir = Path::new("benchmark_results");
    if !benchmark_dir.exists() {
        create_dir_all(benchmark_dir).expect("Failed to create benchmark directory");
    }

    println!("Running Quantum Protocols Benchmark Suite");
    println!("=========================================");
    
    // Get global profiler instance
    let profiler = get_profiler();
    
    // Run all benchmarks
    println!("\nRunning core benchmarks...");
    run_core_benchmarks();
    
    println!("\nRunning error correction benchmarks...");
    run_error_correction_benchmarks();
    
    println!("\nRunning consensus benchmarks...");
    run_consensus_benchmarks();
    
    println!("\nRunning integration benchmarks...");
    run_integration_benchmarks();
    
    println!("\nRunning security benchmarks...");
    run_security_benchmarks();
    
    // Generate comprehensive report
    println!("\nGenerating performance reports...");
    
    // Create dashboard HTML
    let dashboard_path = benchmark_dir.join("performance_dashboard.html");
    create_performance_dashboard(&profiler, dashboard_path.to_str().unwrap())
        .expect("Failed to create performance dashboard");
    println!("Dashboard created at: {}", dashboard_path.display());
    
    // Export to CSV for further analysis
    let csv_path = benchmark_dir.join("performance_data.csv");
    export_to_csv(&profiler, csv_path.to_str().unwrap())
        .expect("Failed to export performance data to CSV");
    println!("CSV data exported to: {}", csv_path.display());
    
    // Print summary
    println!("\nBenchmark Summary");
    println!("----------------");
    println!("{}", profiler.generate_report());
}

/// Run benchmarks for core quantum operations
fn run_core_benchmarks() {
    let profiler = get_profiler();
    
    // Single qubit operations
    for _ in 0..10 {
        profiler.profile("qubit_x_gate", &OperationType::QubitOperation, || {
            let mut qubit = Qubit::new();
            qubit.x();
            qubit
        });
        
        profiler.profile("qubit_h_gate", &OperationType::QubitOperation, || {
            let mut qubit = Qubit::new();
            qubit.hadamard();
            qubit
        });
        
        profiler.profile("qubit_measure", &OperationType::QubitOperation, || {
            let mut qubit = Qubit::new();
            qubit.hadamard(); // Put in superposition first
            qubit.measure()
        });
    }
    
    // Register operations
    for &size in &[2, 4, 8] {
        profiler.profile(
            &format!("register_create_{size}"),
            &OperationType::RegisterOperation,
            || {
                QuantumRegister::new(size)
            },
        );
        
        profiler.profile(
            &format!("register_hadamard_all_{size}"),
            &OperationType::RegisterOperation,
            || {
                let mut register = QuantumRegister::new(size);
                register.hadamard_all();
                register
            },
        );
        
        profiler.profile(
            &format!("register_measure_all_{size}"),
            &OperationType::RegisterOperation,
            || {
                let mut register = QuantumRegister::new(size);
                register.hadamard_all();
                register.measure_all()
            },
        );
    }
    
    // Quantum state operations
    for &size in &[2, 4, 8] {
        profiler.profile(
            &format!("state_create_{size}"),
            &OperationType::StateOperation,
            || {
                QuantumState::new(size)
            },
        );
    }
    
    profiler.profile("state_bell_pair", &OperationType::StateOperation, || {
        QuantumState::bell_pair()
    });
    
    profiler.profile("state_ghz_state", &OperationType::StateOperation, || {
        QuantumState::ghz(3)
    });
    
    // State serialization
    let state = QuantumState::new(4);
    
    profiler.profile("state_serialize", &OperationType::StateOperation, || {
        state.serialize()
    });
}

/// Run benchmarks for error correction
fn run_error_correction_benchmarks() {
    let profiler = get_profiler();
    
    // QECC benchmarks
    for &code in &[CorrectionCode::Repetition, CorrectionCode::Steane7, CorrectionCode::Shor9] {
        let code_name = match code {
            CorrectionCode::None => "none",
            CorrectionCode::Repetition => "repetition",
            CorrectionCode::Steane7 => "steane7",
            CorrectionCode::Shor9 => "shor9",
            CorrectionCode::Surface => "surface",
            CorrectionCode::BitFlip3 => "bit_flip_3",
            CorrectionCode::PhaseFlip3 => "phase_flip_3",
        };
        
        profiler.profile(
            &format!("qecc_encode_{code_name}"),
            &OperationType::ErrorCorrection,
            || {
                let state = QuantumState::new(9); // Big enough for all codes
                let mut qecc = QECC::new();
                qecc.set_state(state);
                // Call encode through the trait method
                <QECC as ErrorCorrection>::encode(&mut qecc, code)
            },
        );
    }
    
    // QCQP benchmarks (quantum classical quantum protocol)
    let mut qcqp = QCQP::new_default();
    let data = b"This is test data for benchmarking the QCQP protocol";
    
    let _ = profiler.profile("qcqp_protect_data", &OperationType::ErrorCorrection, || {
        qcqp.protect_classical_data(data)
    });
    
    if let Ok(protected) = qcqp.protect_classical_data(data) {
        let _ = profiler.profile("qcqp_verify_data", &OperationType::ErrorCorrection, || {
            qcqp.verify_protected_data(&protected)
        });
    }
    
    // QFTP (Quantum Fault Tolerance Protocol) - simple creation
    profiler.profile("qftp_create", &OperationType::ErrorCorrection, || {
        QFTP::new()
    });
}

/// Run benchmarks for consensus protocols
fn run_consensus_benchmarks() {
    let profiler = get_profiler();
    
    // QPCP (Quantum Proof of Consensus Protocol)
    let mut qpcp = QPCP::new("benchmark_node".to_string());
    let proposal_id = "test_proposal_id";
    
    profiler.profile("qpcp_vote", &OperationType::ConsensusProtocol, || {
        // Use the trait method without actually awaiting
        let _future = <QPCP as ConsensusProtocol>::vote(&mut qpcp, proposal_id, true);
        true // We're just measuring function call overhead
    });
    
    // Basic operations that don't require type-specific methods
    profiler.profile("qpcp_operations", &OperationType::ConsensusProtocol, || {
        // Just measure struct operations
        QPCP::new("benchmark_operation".to_string())
    });
    
    // QTSP (Quantum Threshold Signature Protocol)
    let mut qtsp = QTSP::new("benchmark_node");
    
    let _ = profiler.profile("qtsp_rotate_keys", &OperationType::ConsensusProtocol, || {
        qtsp.rotate_keys()
    });
    
    let _ = profiler.profile("qtsp_sign_message", &OperationType::ConsensusProtocol, || {
        qtsp.sign_message(b"test message")
    });
    
    // QVCP (Quantum Verifiable Consensus Protocol)
    profiler.profile("qvcp_create", &OperationType::ConsensusProtocol, || {
        QVCP::new("benchmark_node".to_string())
    });
}

/// Run benchmarks for integration protocols
fn run_integration_benchmarks() {
    let profiler = get_profiler();
    
    // QDAP (Quantum Data Automatic Packing)
    let mut qdap = QDAP::new_default();
    let data = b"This is test data for benchmarking the QDAP protocol with different encoding schemes";
    
    // Test different encoding schemes
    for &encoding in &[
        EncodingScheme::AmplitudeEncoding, 
        EncodingScheme::PhaseEncoding,
        EncodingScheme::DualEncoding,
        EncodingScheme::BinaryEncoding,
    ] {
        let encoding_name = match encoding {
            EncodingScheme::AmplitudeEncoding => "amplitude",
            EncodingScheme::PhaseEncoding => "phase",
            EncodingScheme::DualEncoding => "dual",
            EncodingScheme::BinaryEncoding => "binary",
        };
        
        profiler.profile(
            &format!("qdap_encode_{encoding_name}"),
            &OperationType::ClassicalQuantumTranslation,
            || {
                qdap.encode_to_quantum(data, encoding, None)
            },
        );
    }
    
    // Encode and then decode - use a specific result to avoid decode_from_quantum
    if let Ok((encoded, metadata)) = qdap.encode_to_quantum(data, EncodingScheme::AmplitudeEncoding, None) {
        profiler.profile(
            "qdap_decode_amplitude",
            &OperationType::ClassicalQuantumTranslation,
            || {
                qdap.decode_to_classical(&encoded, &metadata)
            },
        );
    }
}

/// Run benchmarks for security protocols
fn run_security_benchmarks() {
    let profiler = get_profiler();
    
    // QRNG (Quantum Random Number Generator)
    let mut qrng = QRNG::new_default();
    
    // Generate single bytes - use generate_bytes(1) instead of generate_byte
    for _ in 0..10 {
        profiler.profile("qrng_generate_single_byte", &OperationType::SecurityOperation, || {
            qrng.generate_bytes(1)
        });
    }
    
    // Generate byte arrays
    for &size in &[16, 32, 64, 128] {
        profiler.profile(
            &format!("qrng_generate_bytes_{size}"),
            &OperationType::SecurityOperation,
            || {
                qrng.generate_bytes(size)
            },
        );
    }
    
    // QSYP (Quantum Synchronization Protocol) - requires network simulation, skip for benchmarks
    
    // Combined operations - secure random value with entropy verification
    profiler.profile("qrng_secure_random", &OperationType::SecurityOperation, || {
        if let Ok(mut bytes) = qrng.generate_bytes(32) {
            // Add entropy verification to measure overhead
            if let Some(first_byte) = bytes.first_mut() {
                // Apply entropy verification (simplified)
                *first_byte ^= *first_byte >> 4;
            }
            
            bytes
        } else {
            Vec::new() // Fallback if generation fails
        }
    });
} 