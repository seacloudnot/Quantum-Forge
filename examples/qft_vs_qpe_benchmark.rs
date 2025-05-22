// Benchmark comparing Quantum Fourier Transform vs Quantum Phase Estimation
//
// This example measures the performance of QFT and QPE algorithms
// with varying numbers of qubits.

use quantum_protocols::core::{QuantumFourierTransform, QuantumPhaseEstimation, QuantumRegister};
use std::f64::consts::PI;
use std::time::Instant;

// Simple phase rotation function for QPE
fn phase_025(reg: &mut QuantumRegister, control: usize, target: usize, power: usize) {
    // Check if control qubit is in |1⟩ state
    let control_is_one = reg.qubit(control).is_some_and(|q| q.prob_1() > 0.5);
    
    if control_is_one {
        // Apply phase rotation U^(2^j)
        let angle = 0.25 * 2.0 * PI * (power as f64);
        reg.phase(target, angle);
    }
}

fn main() {
    println!("=== QFT vs QPE Performance Benchmark ===");
    println!();
    
    // Run benchmarks with increasing qubit counts
    println!("Qubits\tQFT Time (ms)\tQPE Time (ms)\tQFT Gates\tQPE Gates\tRatio");
    
    for qubits in 3..12 {
        let (qft_time, qft_gates) = benchmark_qft(qubits);
        let (qpe_time, qpe_gates) = benchmark_qpe(qubits);
        
        println!("{}\t{:.2}\t\t{:.2}\t\t{}\t\t{}\t\t{:.2}", 
            qubits, 
            qft_time, 
            qpe_time, 
            qft_gates, 
            qpe_gates,
            qpe_time / qft_time
        );
    }
    
    // Provide analysis
    println!("\nAnalysis:");
    println!("1. QPE uses more gates than QFT because it requires additional controlled-U operations");
    println!("2. As qubit count increases, the performance difference grows due to exponential complexity");
    println!("3. QPE requires an eigenstate preparation step that QFT doesn't need");
    println!("4. The time ratio between QPE and QFT demonstrates the computational overhead of phase estimation");
}

fn benchmark_qft(num_qubits: usize) -> (f64, usize) {
    // Create a register for benchmarking
    let register = QuantumRegister::new(num_qubits);
    
    // Start timing
    let start = Instant::now();
    
    // Create and run QFT
    let mut qft = QuantumFourierTransform::new(num_qubits)
        .with_register(register);
    
    let result = qft.run();
    
    // End timing
    let elapsed = start.elapsed();
    let time_ms = elapsed.as_secs_f64() * 1000.0;
    
    // Get gate count from result
    let gate_count = result.statistics.gate_count;
    
    (time_ms, gate_count)
}

fn benchmark_qpe(num_qubits: usize) -> (f64, usize) {
    // Phase qubits = num_qubits - 1, target = 1
    let phase_qubits = num_qubits - 1;
    
    // Initialize target register to |1⟩
    let init_target = |reg: &mut QuantumRegister, start_idx: usize, _num_qubits: usize| {
        reg.x(start_idx);
    };
    
    // Start timing
    let start = Instant::now();
    
    // Create and run QPE
    let mut qpe = QuantumPhaseEstimation::new(phase_qubits, 1, phase_025)
        .with_target_initialization(init_target);
    
    let result = qpe.run();
    
    // End timing
    let elapsed = start.elapsed();
    let time_ms = elapsed.as_secs_f64() * 1000.0;
    
    // Get gate count from result
    let gate_count = result.statistics.gate_count;
    
    (time_ms, gate_count)
} 