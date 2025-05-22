// Detailed debugging of Quantum Phase Estimation for Z gate
//
// This example steps through each stage of the QPE algorithm
// to understand why phase kickback isn't working correctly.

use quantum_protocols::core::QuantumRegister;

fn main() {
    println!("=== Quantum Phase Estimation Detailed Debugging ===\n");
    
    // Use a minimal example: 2 precision qubits + 1 target qubit
    // For Z gate with phase 0.5 (binary 0.1)
    
    // Create register
    let mut reg = QuantumRegister::new(3);
    
    // STEP 1: Initialize target qubit (qubit 2) to |1⟩ eigenstate
    reg.x(2);
    println!("Step 1: Target qubit initialized to |1⟩ state");
    print_state(&reg, 3);
    
    // STEP 2: Initialize control qubits to |+⟩ state with Hadamard
    reg.hadamard(0);
    reg.hadamard(1);
    println!("\nStep 2: Control qubits initialized to |+⟩ state");
    print_state(&reg, 3);
    
    // STEP 3: Apply controlled operations
    // For Z gate with phase 0.5:
    // - control 0 applies Z^1 (phase π) to target
    // - control 1 applies Z^2 (phase 2π = identity) to target
    
    println!("\nStep 3: Applying controlled-Z operations");
    
    // Theoretical analysis:
    println!("  Theoretical state before controlled operations:");
    println!("  |Ψ⟩ = (|00⟩ + |01⟩ + |10⟩ + |11⟩)|1⟩/2");
    
    // Apply controlled-Z from qubit 0 to target (Z^1)
    reg.cz(0, 2);
    println!("\n  Applied controlled-Z from qubit 0 to target");
    print_state(&reg, 3);
    
    // Theoretical analysis:
    println!("\n  Theoretical state after controlled-Z from qubit 0:");
    println!("  |Ψ⟩ = (|00⟩ + |01⟩ + |10⟩(-1) + |11⟩(-1))|1⟩/2");
    println!("  |Ψ⟩ = (|00⟩ + |01⟩ - |10⟩ - |11⟩)|1⟩/2");
    
    // No operation for qubit 1 (Z^2 = I)
    println!("\n  No operation needed for qubit 1 (Z^2 = I)");
    
    // STEP 4: Apply inverse QFT to control qubits
    println!("\nStep 4: Applying inverse QFT to control qubits");
    
    // For 2 qubits, inverse QFT is:
    // 1. Apply controlled-S† from qubit 1 to qubit 0
    // 2. Apply H to qubit 1
    // 3. Apply H to qubit 0
    // 4. Swap qubits 0 and 1 (not needed for 2 qubits)
    
    // Apply controlled-S† from qubit 1 to qubit 0
    if reg.qubit(1).unwrap().prob_1() > 0.5 {
        reg.s_dagger(0);
    }
    println!("  Applied conditional S† gate");
    print_state(&reg, 3);
    
    // Apply H to qubit 1
    reg.hadamard(1);
    println!("\n  Applied H to qubit 1");
    print_state(&reg, 3);
    
    // Apply H to qubit 0
    reg.hadamard(0);
    println!("\n  Applied H to qubit 0");
    print_state(&reg, 3);
    
    // Theoretical analysis:
    println!("\n  Theoretical state after inverse QFT:");
    println!("  |Ψ⟩ = |01⟩|1⟩");
    println!("  First qubit is |0⟩, second qubit is |1⟩");
    println!("  This represents phase 0.5 (binary 0.01 read right-to-left)");
    
    // STEP 5: Measure control qubits
    let results = [reg.measure(0).unwrap(),
        reg.measure(1).unwrap()];
    
    // Calculate phase
    let mut phase = 0.0;
    for (i, &bit) in results.iter().enumerate() {
        phase += (bit as f64) / (1 << (i+1)) as f64;
    }
    
    println!("\nStep 5: Measurement results");
    println!("  Control qubits: [{}, {}]", results[0], results[1]);
    println!("  Binary phase: 0.{}{}", results[1], results[0]); // Read from right to left
    println!("  Decimal phase: {:.4}, Expected: 0.5", phase);
    
    // STEP 6: Compare with direct fix
    println!("\nStep 6: Compare with direct fix");
    direct_fix();
}

// Helper function to print register state
fn print_state(reg: &QuantumRegister, num_qubits: usize) {
    println!("  Current state:");
    for i in 0..num_qubits {
        if let Some(q) = reg.qubit(i) {
            let (alpha, beta) = q.get_coeffs();
            println!("    Qubit {}: α=({:.4}, {:.4}), β=({:.4}, {:.4})",
                     i, alpha.0, alpha.1, beta.0, beta.1);
            println!("             |0⟩: {:.4}, |1⟩: {:.4}",
                     q.prob_0(), q.prob_1());
        }
    }
}

// Direct fix comparison
fn direct_fix() {
    // Create a new register
    let mut reg = QuantumRegister::new(3);
    
    // Set target qubit to |1⟩ state
    reg.x(2);
    
    // Set control qubit 0 to |1⟩ state to represent phase 0.5
    reg.x(0);
    
    println!("  Manual fix - direct initialization:");
    print_state(&reg, 3);
    
    // Measure
    let results = [reg.measure(0).unwrap(),
        reg.measure(1).unwrap()];
    
    // Calculate phase
    let mut phase = 0.0;
    for (i, &bit) in results.iter().enumerate() {
        phase += (bit as f64) / (1 << (i+1)) as f64;
    }
    
    println!("\n  Measurement results:");
    println!("    Control qubits: [{}, {}]", results[0], results[1]);
    println!("    Binary phase: 0.{}{}", results[1], results[0]); // Read from right to left
    println!("    Decimal phase: {:.4}, Expected: 0.5", phase);
} 