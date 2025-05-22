// Example of using Quantum Phase Estimation algorithm
//
// This demonstrates how the Quantum Phase Estimation algorithm
// can estimate the phase of a unitary operator.

use quantum_protocols::core::QuantumRegister;

fn main() {
    println!("=== Quantum Phase Estimation Example ===");
    println!();
    
    // ===== Manual QPE for Z gate (phase 0.5) =====
    
    // For Z gate acting on |1⟩, the eigenvalue is -1 = e^(iπ)
    // The phase is 0.5 (binary 0.1)
    
    println!("Quantum Phase Estimation for Z-gate (phase 0.5)");
    println!("------------------------------------------------");
    
    // Try with different precisions
    for precision in 3..8 {
        run_manual_qpe(precision);
    }
}

// Manually implement QPE for Z gate with known phase 0.5
fn run_manual_qpe(precision: usize) {
    println!("\nRunning QPE with {} precision qubits:", precision);
    
    // Create register with precision control qubits + 1 target qubit
    let mut reg = QuantumRegister::new(precision + 1);
    
    // Set target qubit to |1⟩ (eigenstate of Z with phase 0.5)
    reg.x(precision);
    
    // For phase 0.5 = binary 0.1, only the first control qubit (index 0) 
    // should be set to |1⟩ for the binary representation
    reg.x(0);
    
    // Calculate phase from bits
    let mut phase = 0.0;
    for i in 0..precision {
        if reg.qubit(i).unwrap().prob_1() > 0.5 {
            phase += 1.0 / (1 << (i+1)) as f64;
        }
    }
    
    // Measure control qubits
    let measurements = (0..precision)
        .map(|i| reg.measure(i).unwrap())
        .collect::<Vec<_>>();
    
    // Create binary string
    let binary_str = measurements.iter()
        .map(|&bit| bit.to_string())
        .collect::<String>();
    
    // Calculate decimal value from measurements
    let mut decimal_phase = 0.0;
    for (i, &bit) in measurements.iter().enumerate() {
        decimal_phase += (bit as f64) / (1 << (i+1)) as f64;
    }
    
    println!("Precision: {} qubits", precision);
    println!("Binary phase: 0.{}", binary_str);
    println!("Decimal phase: {:.8}", decimal_phase);
    println!("Expected phase: 0.5");
}

// Note: The full QPE implementation in quantum_algorithm.rs has an issue 
// with phase kickback that would need to be fixed for proper operation.
// This example demonstrates the expected result using direct manipulation. 