// Test of phase kickback for Quantum Phase Estimation
//
// The phase kickback effect is essential for QPE to work correctly.

use quantum_protocols::core::{Qubit, QuantumRegister};

fn main() {
    println!("=== Phase Kickback Test ===");
    
    // Test 1: Direct Z gate application to |1⟩ state
    println!("\nTest 1: Z gate on |1⟩ state");
    let mut q = Qubit::one();
    println!("Initial state: |1⟩");
    
    // Apply Z gate
    q.z();
    let (alpha, beta) = q.get_coeffs();
    println!("After Z gate:");
    println!("Alpha: ({:.6}, {:.6})", alpha.0, alpha.1);
    println!("Beta: ({:.6}, {:.6})", beta.0, beta.1);
    println!("Phase of |1⟩ component is -1, which corresponds to phase 0.5");
    
    // Test 2: Phase kickback in a 2-qubit system
    println!("\nTest 2: Phase kickback in 2-qubit system");
    let mut reg = QuantumRegister::new(2);
    
    // Prepare control qubit in |+⟩ state
    reg.hadamard(0);
    
    // Prepare target qubit in |1⟩ state
    reg.x(1);
    
    println!("Initial state: |+⟩ ⊗ |1⟩");
    
    // Apply controlled-Z
    reg.cz(0, 1);
    println!("Applied controlled-Z from qubit 0 to qubit 1");
    
    // Display state after controlled-Z
    let control = reg.qubit(0).unwrap();
    let (c_alpha, c_beta) = control.get_coeffs();
    
    let target = reg.qubit(1).unwrap();
    let (t_alpha, t_beta) = target.get_coeffs();
    
    println!("State after controlled-Z:");
    println!("Control qubit: alpha=({:.6}, {:.6}), beta=({:.6}, {:.6})", 
             c_alpha.0, c_alpha.1, c_beta.0, c_beta.1);
    println!("Target qubit: alpha=({:.6}, {:.6}), beta=({:.6}, {:.6})", 
             t_alpha.0, t_alpha.1, t_beta.0, t_beta.1);
    
    // Expected state: (|0⟩|1⟩ + |1⟩(-|1⟩))/√2 = (|0⟩|1⟩ - |1⟩|1⟩)/√2
    
    // Apply Hadamard to control qubit for phase kickback
    reg.hadamard(0);
    println!("Applied Hadamard to control qubit");
    
    // Get control qubit state
    let control = reg.qubit(0).unwrap();
    let (c_alpha, c_beta) = control.get_coeffs();
    
    println!("Control qubit after Hadamard:");
    println!("Alpha: ({:.6}, {:.6})", c_alpha.0, c_alpha.1);
    println!("Beta: ({:.6}, {:.6})", c_beta.0, c_beta.1);
    println!("Prob(|0⟩) = {:.6}, Prob(|1⟩) = {:.6}", control.prob_0(), control.prob_1());
    
    // Expected state of control: |1⟩
    println!("Expected state of control: |1⟩");
    
    // Measure
    let result = reg.measure(0).unwrap();
    println!("Measured control qubit: |{}⟩", result);
    
    // If kickback works correctly, we should measure |1⟩ with high probability
    
    // Test 3: Full 3+1 qubit phase estimation for Z gate
    println!("\nTest 3: Full QPE for Z gate (phase 0.5)");
    manual_z_qpe();
}

fn manual_z_qpe() {
    // Create 4-qubit register (3 control, 1 target)
    let mut reg = QuantumRegister::new(4);
    
    // Prepare target in |1⟩ state (eigenstate of Z with phase 0.5)
    reg.x(3);
    
    // Prepare all control qubits in |+⟩ state
    for i in 0..3 {
        reg.hadamard(i);
    }
    
    println!("Register initialized with target |1⟩ and controls |+⟩");
    
    // To get phase 0.5 (binary 0.1) for Z gate on |1⟩:
    // After controlled-Z from qubit 0, qubit 0 should be |1⟩ after Hadamard
    // After controlled-Z from qubit 1, qubit 1 should still be |+⟩
    // After controlled-Z from qubit 2, qubit 2 should still be |+⟩
    
    // Apply controlled operations
    // First qubit controls Z^1 = Z
    reg.cz(0, 3);
    
    // Do nothing for second and third qubits since Z^2 = Z^4 = I
    
    // Apply Hadamard to all control qubits to measure phase
    for i in 0..3 {
        reg.hadamard(i);
    }
    
    // Print control qubit states
    println!("Control qubit states after Hadamard:");
    for i in 0..3 {
        let q = reg.qubit(i).unwrap();
        println!("Qubit {}: Prob(|0⟩) = {:.6}, Prob(|1⟩) = {:.6}", 
                 i, q.prob_0(), q.prob_1());
    }
    
    // Measure
    let results = vec![
        reg.measure(0).unwrap(),
        reg.measure(1).unwrap(),
        reg.measure(2).unwrap()
    ];
    
    // Calculate phase
    let mut phase = 0.0;
    let mut denominator = 2.0;
    
    for &bit in &results {
        phase += (bit as f64) / denominator;
        denominator *= 2.0;
    }
    
    println!("Measurement results: {:?}", results);
    println!("Binary phase: 0.{}", results.iter().map(|&b| b.to_string()).collect::<String>());
    println!("Decimal phase: {:.4}, Expected: 0.5", phase);
} 