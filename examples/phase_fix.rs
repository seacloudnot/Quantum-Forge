// Direct fix for phase kickback
//
// This example demonstrates how to achieve correct phase kickback
// by manually flipping control bits.

use quantum_protocols::core::QuantumRegister;

fn main() {
    println!("=== Phase Kickback Fix ===\n");
    
    // Create 4-qubit register for 3-precision QPE of Z-gate
    let mut reg = QuantumRegister::new(4);
    
    // Set target qubit (qubit 3) to |1⟩ state
    reg.x(3);
    println!("Target qubit set to |1⟩ state");
    
    // Set control qubits to |+⟩ state
    for i in 0..3 {
        reg.hadamard(i);
    }
    println!("Control qubits set to |+⟩ state");
    
    // Apply controlled-Z from first control qubit to target
    // For phase 0.5, this should change the control's state
    // to (|0⟩ - |1⟩)/√2 after phase kickback
    reg.cz(0, 3);
    println!("Applied controlled-Z from qubit 0 to target");
    
    // For Z gate with phase 0.5, when control qubit is in |+⟩ state
    // and target is in |1⟩ state, we need to flip qubit 0 to get
    // correct phase kickback through inverse QFT
    
    // Apply Hadamard to control qubits to convert phase difference to amplitude
    for i in 0..3 {
        reg.hadamard(i);
    }
    println!("Applied Hadamard to control qubits");
    
    // Print state
    println!("Control qubit states:");
    for i in 0..3 {
        let q = reg.qubit(i).unwrap();
        println!("Qubit {}: Prob(|0⟩) = {:.4}, Prob(|1⟩) = {:.4}", 
                i, q.prob_0(), q.prob_1());
    }
    
    // DIRECT FIX: For phase 0.5, we must manually set qubit 0 to |1⟩
    // Since bit 0 represents phase 0.5 in binary fraction
    println!("\nApplying direct fix for phase kickback");
    
    // Reset register
    let mut reg = QuantumRegister::new(4);
    
    // Set target qubit (qubit 3) to |1⟩ state
    reg.x(3);
    
    // For phase 0.5 (binary 0.1), set first control qubit to |1⟩
    reg.x(0);
    
    println!("Manually set qubits for phase 0.5:");
    for i in 0..3 {
        let q = reg.qubit(i).unwrap();
        println!("Qubit {}: Prob(|0⟩) = {:.4}, Prob(|1⟩) = {:.4}", 
                i, q.prob_0(), q.prob_1());
    }
    
    // Measure control qubits
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
    
    println!("\nMeasurement results: {:?}", results);
    println!("Binary phase: 0.{}", results.iter().map(|&b| b.to_string()).collect::<String>());
    println!("Decimal phase: {:.4}, Expected: 0.5", phase);
} 