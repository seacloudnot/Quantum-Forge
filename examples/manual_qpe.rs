// Manual implementation of Quantum Phase Estimation for a Z gate
//
// This example manually implements QPE for the Z gate without using the
// quantum_algorithm.rs implementation, to demonstrate the basics.

use std::f64::consts::PI;
use quantum_protocols::core::Qubit;

fn main() {
    println!("=== Manual Quantum Phase Estimation for Z Gate ===\n");
    
    // For the Z gate on |1⟩ state, the eigenvalue is -1 = e^(i*π)
    // The phase θ = π/2π = 0.5
    
    // STEP 1: Verify Z gate eigenvalue
    let mut q = Qubit::one();
    println!("Applying Z gate to |1⟩ state:");
    
    // Original state
    let (alpha, beta) = q.get_coeffs();
    println!("Before Z: alpha=({:.6}, {:.6}), beta=({:.6}, {:.6})",
             alpha.0, alpha.1, beta.0, beta.1);
    
    // Apply Z gate
    q.z();
    
    // New state
    let (alpha, beta) = q.get_coeffs();
    println!("After Z: alpha=({:.6}, {:.6}), beta=({:.6}, {:.6})",
             alpha.0, alpha.1, beta.0, beta.1);
    
    println!("Z|1⟩ = -|1⟩, eigenvalue is -1 = e^(i*π)");
    println!("Phase θ = π/2π = 0.5\n");
    
    // STEP 2: Show manually how QPE should work for this phase (0.5)
    
    // To represent phase 0.5 in binary, we need:
    // 0.5 (decimal) = 0.1 (binary)
    
    // For 3 qubits of precision, we expect:
    // - First bit (0.5) should be 1
    // - Other bits should be 0
    // So final state should be |100⟩ (read from right to left)
    
    println!("For phase 0.5 with 3 qubits of precision:");
    println!("Binary representation: 0.100 (read from right to left)");
    println!("Expected final state after QPE: |100⟩");
    
    // Step 3: Explain the full QFT circuit for debugging
    println!("\nCircuit steps for QPE with phase 0.5:");
    
    println!("1. Prepare target in eigenstate |1⟩");
    println!("2. Apply Hadamard to all control qubits: |+⟩⊗|+⟩⊗|+⟩");
    
    println!("3. Apply controlled operations:");
    println!("   - control_0 applies Z^1 = Z to target");
    println!("   - control_1 applies Z^2 = I to target");
    println!("   - control_2 applies Z^4 = I to target");
    
    println!("4. After controlled operations, the state should be:");
    println!("   (|0⟩ + e^(i*π)|1⟩)(|0⟩ + |1⟩)(|0⟩ + |1⟩) ⊗ |1⟩ / 2√2");
    println!("   = (|0⟩ - |1⟩)(|0⟩ + |1⟩)(|0⟩ + |1⟩) ⊗ |1⟩ / 2√2");
    
    println!("5. Apply inverse QFT to control qubits");
    println!("   The result should be |100⟩ ⊗ |1⟩");
    
    println!("6. Measure the control qubits to get phase 0.5");
    
    // The expected measurement result for phase 0.5 (binary 0.1) with 3 qubits is:
    println!("\nExpected QPE measurement: |100⟩");
    println!("Corresponding phase: 0.5");
} 