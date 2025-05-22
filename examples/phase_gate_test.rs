// Direct test of phase gate functionality
//
// This example tests the phase gate implementation by applying different phase
// rotations and observing the effect on a qubit.

use quantum_protocols::core::{QuantumRegister, Qubit};

fn main() {
    println!("=== Phase Gate Test ===");
    println!();
    
    // Test if Z gate has correct phase (0.5)
    test_z_eigenstate();
}

// Verify that Z gate acts correctly on |1⟩ state with phase 0.5
fn test_z_eigenstate() {
    println!("Testing Z gate as eigenvalue on |1⟩ state");
    
    // Create target in |1⟩ state
    let mut qubit = Qubit::one();
    
    println!("Initial state: |1⟩");
    println!("Applying Z gate...");
    
    // Apply Z gate (should add phase 0.5 = π rotation)
    qubit.z();
    
    // Get the coefficients after Z operation
    let (alpha, beta) = qubit.get_coeffs();
    
    println!("State after Z gate:");
    println!("Alpha = ({:.4}, {:.4})", alpha.0, alpha.1);
    println!("Beta = ({:.4}, {:.4})", beta.0, beta.1);
    
    // For Z gate on |1⟩, we expect Beta = (-1, 0)
    let expected_phase = -1.0;
    println!("Expected phase factor for |1⟩: {:.4}", expected_phase);
    println!("Actual phase factor for |1⟩: {:.4}", beta.0);
    
    // Test controlled-Z operation with a register
    test_controlled_z();
}

// Verify that controlled-Z works correctly on QPE-like setup
fn test_controlled_z() {
    println!("\nTesting controlled-Z gate");
    
    // Create a 2-qubit register
    let mut reg = QuantumRegister::new(2);
    
    // Set control (qubit 0) to |+⟩ state
    reg.hadamard(0);
    
    // Set target (qubit 1) to |1⟩ state
    reg.x(1);
    
    println!("Initial state: |+⟩ ⊗ |1⟩");
    println!("Applying controlled-Z from qubit 0 to qubit 1...");
    
    // Apply controlled-Z
    reg.cz(0, 1);
    
    // Get the control qubit state
    let control = reg.qubit(0).unwrap();
    let (control_alpha, control_beta) = control.get_coeffs();
    
    // Get the target qubit state
    let target = reg.qubit(1).unwrap();
    let (target_alpha, target_beta) = target.get_coeffs();
    
    println!("Control qubit state:");
    println!("Alpha = ({:.4}, {:.4})", control_alpha.0, control_alpha.1);
    println!("Beta = ({:.4}, {:.4})", control_beta.0, control_beta.1);
    
    println!("Target qubit state:");
    println!("Alpha = ({:.4}, {:.4})", target_alpha.0, target_alpha.1);
    println!("Beta = ({:.4}, {:.4})", target_beta.0, target_beta.1);
    
    // For controlled-Z on |+⟩ ⊗ |1⟩, target should get -1 phase only in the |1⟩ component
    // Creating the state (|0⟩|1⟩ - |1⟩|1⟩)/√2
    
    // Now apply Hadamard to control to disentangle and observe the result
    println!("\nApplying Hadamard to control qubit to observe phase effect");
    reg.hadamard(0);
    
    // Get the control qubit state after Hadamard
    let control = reg.qubit(0).unwrap();
    println!("Control qubit after Hadamard:");
    println!("Prob(|0⟩) = {:.4}", control.prob_0());
    println!("Prob(|1⟩) = {:.4}", control.prob_1());
    
    // For controlled-Z from |+⟩ to |1⟩, after applying H to the control,
    // control qubit should be in |1⟩ state due to phase kickback
    
    // Measure to verify
    let control_result = reg.measure(0).unwrap();
    println!("Measured control qubit: |{control_result}⟩");
} 