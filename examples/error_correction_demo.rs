// Quantum Error Correction Example
//
// This example demonstrates bit flip and phase flip error correction codes.

use quantum_protocols::core::QuantumRegister;
use quantum_protocols::error_correction::{BitFlipCode, PhaseFlipCode};
use std::f64::consts::PI;

fn main() {
    println!("=== Quantum Error Correction Demo ===\n");
    
    // Test bit flip code
    test_bit_flip_code();
    
    // Test phase flip code
    test_phase_flip_code();
}

fn test_bit_flip_code() {
    println!("\n=== Bit Flip Code Demo ===");
    println!("The 3-qubit bit flip code protects against X errors (bit flips)");
    
    // Create a register with 5 qubits
    // q0: logical qubit
    // q1, q2, q3: code qubits
    // q4: auxiliary/output qubit
    let mut register = QuantumRegister::new(5);
    
    // Put in a superposition state
    register.hadamard(0);
    
    // Look at initial state
    println!("\nInitial state of logical qubit:");
    print_qubit_state(&register, 0);
    
    // Create a bit flip code
    let mut code = BitFlipCode::new(register, 1);
    
    // Encode the state
    println!("\nEncoding logical qubit into three physical qubits...");
    let (q1, q2, q3) = code.encode(0);
    
    // Print encoded state
    println!("\nEncoded state:");
    print_encoded_state(&code.register(), q1, q2, q3);
    
    // Introduce a bit flip error on q2
    println!("\nIntroducing a bit flip error on qubit {}...", q2);
    code.register_mut().x(q2);
    
    // Print state with error
    println!("\nState after error:");
    print_encoded_state(&code.register(), q1, q2, q3);
    
    // Correct the error
    println!("\nApplying error correction...");
    code.correct(q1, q2, q3);
    
    // Print corrected state
    println!("\nState after correction:");
    print_encoded_state(&code.register(), q1, q2, q3);
    
    // Decode back to the output qubit
    println!("\nDecoding back to output qubit...");
    code.decode(q1, q2, q3, 4);
    
    // Print the final state of the output qubit
    println!("\nFinal state of output qubit:");
    print_qubit_state(code.register(), 4);
    
    // Verify it matches the original qubit
    let original = code.register().qubit(0).unwrap();
    let output = code.register().qubit(4).unwrap();
    
    println!("\nVerification: Original and output qubits match: {}", 
             (original.prob_1() - output.prob_1()).abs() < 0.01);
}

fn test_phase_flip_code() {
    println!("\n\n=== Phase Flip Code Demo ===");
    println!("The 3-qubit phase flip code protects against Z errors (phase flips)");
    
    // Create a register with 5 qubits
    // q0: logical qubit
    // q1, q2, q3: code qubits
    // q4: auxiliary/output qubit
    let mut register = QuantumRegister::new(5);
    
    // Put in a |+⟩ state to demonstrate phase
    register.hadamard(0);
    
    // Look at initial state
    println!("\nInitial state of logical qubit:");
    print_qubit_state(&register, 0);
    
    // Create a phase flip code
    let mut code = PhaseFlipCode::new(register, 1);
    
    // Encode the state
    println!("\nEncoding logical qubit into three physical qubits...");
    let (q1, q2, q3) = code.encode(0);
    
    // Print encoded state
    println!("\nEncoded state:");
    print_encoded_state(&code.register(), q1, q2, q3);
    
    // Introduce a phase flip error on q2
    println!("\nIntroducing a phase flip error on qubit {}...", q2);
    code.apply_phase_flip(q2);
    
    // Print state with error
    println!("\nState after error:");
    print_encoded_state(&code.register(), q1, q2, q3);
    
    // Correct the error
    println!("\nApplying error correction...");
    code.correct(q1, q2, q3);
    
    // Print corrected state
    println!("\nState after correction:");
    print_encoded_state(&code.register(), q1, q2, q3);
    
    // Decode back to the output qubit
    println!("\nDecoding back to output qubit...");
    code.decode(q1, q2, q3, 4);
    
    // Print the final state of the output qubit
    println!("\nFinal state of output qubit:");
    print_qubit_state(code.register(), 4);
    
    // Verify it matches the original qubit
    let original = code.register().qubit(0).unwrap();
    let output = code.register().qubit(4).unwrap();
    
    println!("\nVerification: Original and output qubits match: {}", 
             (original.prob_1() - output.prob_1()).abs() < 0.01);
}

// Helper function to print a qubit's state
fn print_qubit_state(register: &QuantumRegister, qubit_idx: usize) {
    if let Some(q) = register.qubit(qubit_idx) {
        let ((alpha_re, alpha_im), (beta_re, beta_im)) = q.get_coeffs();
        
        println!("  |0⟩ component: {:.4} + {:.4}i", alpha_re, alpha_im);
        println!("  |1⟩ component: {:.4} + {:.4}i", beta_re, beta_im);
        println!("  Probabilities: |0⟩={:.4}, |1⟩={:.4}", q.prob_0(), q.prob_1());
    }
}

// Helper function to print the encoded state
fn print_encoded_state(register: &QuantumRegister, q1: usize, q2: usize, q3: usize) {
    println!("  Qubit {}: |0⟩={:.4}, |1⟩={:.4}", 
             q1, register.qubit(q1).unwrap().prob_0(), register.qubit(q1).unwrap().prob_1());
    println!("  Qubit {}: |0⟩={:.4}, |1⟩={:.4}", 
             q2, register.qubit(q2).unwrap().prob_0(), register.qubit(q2).unwrap().prob_1());
    println!("  Qubit {}: |0⟩={:.4}, |1⟩={:.4}", 
             q3, register.qubit(q3).unwrap().prob_0(), register.qubit(q3).unwrap().prob_1());
} 