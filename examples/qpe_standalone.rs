// Standalone Quantum Phase Estimation example
//
// This demonstrates quantum phase estimation with varying qubit counts

use std::time::Instant;
use quantum_protocols::core::{QuantumRegister, QuantumPhaseEstimation};
use std::f64::consts::PI;

// Phase rotation function with phase 0.25
fn phase_025(reg: &mut QuantumRegister, control: usize, target: usize, power: usize) {
    // Check if control qubit is in |1⟩ state
    let control_is_one = reg.qubit(control).map_or(false, |q| q.prob_1() > 0.5);
    
    if control_is_one {
        // Apply phase rotation U^(2^j)
        let angle = 0.25 * 2.0 * PI * (power as f64);
        reg.phase(target, angle);
    }
}

fn main() {
    println!("=== Quantum Phase Estimation Example ===");
    println!("Testing with phase = 0.25 (π/2)");
    println!();
    
    // Initialize target register preparation function
    let init_target = |reg: &mut QuantumRegister, start_idx: usize, _num_qubits: usize| {
        reg.x(start_idx); // Set target qubit to |1⟩
    };
    
    println!("Qubits\tMax Error\tEstimated Phase\tTime (ms)");
    
    // Try with increasing precision (more qubits)
    for precision_qubits in 3..12 {
        // Start timing
        let start = Instant::now();
        
        // Create QPE instance
        let mut qpe = QuantumPhaseEstimation::new(precision_qubits, 1, phase_025)
            .with_target_initialization(init_target);
        
        // Run the algorithm
        let _result = qpe.run();
        
        // Get timing
        let elapsed = start.elapsed();
        let time_ms = elapsed.as_secs_f64() * 1000.0;
        
        // Get the estimated phase
        let estimated_phase = qpe.estimated_phase();
        
        // Calculate maximum theoretical error
        let max_error = 1.0 / (2.0_f64.powi(precision_qubits as i32));
        
        println!("{}\t{:.8}\t{:.8}\t{:.2}", 
                precision_qubits, 
                max_error, 
                estimated_phase,
                time_ms);
    }
    
    // Print a simple QFT vs QPE comparison
    println!("\n=== QFT vs QPE Gate Count Comparison ===");
    println!("Qubits\tQFT Gates\tQPE Gates\tRatio");
    
    for qubits in 3..10 {
        let qft_gates = qubits * (qubits + 1) / 2; // Approximate QFT gate count
        let qpe_gates = qubits * (qubits + 1) / 2 + qubits; // QPE adds controlled operations
        let ratio = qpe_gates as f64 / qft_gates as f64;
        
        println!("{}\t{}\t\t{}\t\t{:.2}", qubits, qft_gates, qpe_gates, ratio);
    }
    
    println!("\nNotes:");
    println!("1. QPE builds on QFT but adds controlled unitary operations");
    println!("2. QPE requires n+m qubits where n is precision qubits and m is target register size");
    println!("3. Higher precision (more qubits) yields more accurate phase estimation");
} 