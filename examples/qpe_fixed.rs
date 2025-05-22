// Quantum Phase Estimation with fixed phase kickback implementation
//
// This example demonstrates the QPE algorithm using the Z gate 
// which has a known phase of 0.5.

use quantum_protocols::core::{QuantumPhaseEstimation, QuantumRegister};
use std::f64::consts::PI;

fn main() {
    println!("=== Quantum Phase Estimation (Fixed Implementation) ===\n");
    
    // Try with different precision levels
    println!("Z-Gate Phase Estimation (Expected: 0.5)");
    println!("-------------------------------------------");
    
    for precision in 3..8 {
        test_z_gate_phase(precision);
    }
    
    // Try with other gates for comparison
    println!("\nT-Gate Phase Estimation (Expected: 0.125)");
    println!("-------------------------------------------");
    
    for precision in 4..9 {
        test_t_gate_phase(precision);
    }
}

// Function to estimate phase of Z-gate (phase 0.5)
fn test_z_gate_phase(precision: usize) {
    println!("\nRunning QPE with {} precision qubits:", precision);
    
    // Initialize QPE algorithm for Z gate with desired precision
    // and 1 target qubit
    let mut qpe = QuantumPhaseEstimation::new(
        precision,     // number of precision qubits
        1,             // 1 target qubit
        z_gate_phase   // controlled-Z gate function
    );
    
    // Initialize target qubit to |1⟩ state (eigenstate of Z)
    qpe = qpe.with_target_initialization(|reg, phase_qubits, _| {
        // Set the target qubit to |1⟩
        reg.x(phase_qubits);
    });
    
    // Run the algorithm
    let results = qpe.run();
    
    // Display results
    let binary_result = results.results.iter()
        .map(|&b| b.to_string())
        .collect::<String>();
    
    // Calculate the phase from binary result
    let mut phase = 0.0;
    for (i, &bit) in results.results.iter().enumerate() {
        phase += (bit as f64) / (1 << (i+1)) as f64;
    }
    
    println!("Precision: {} qubits", precision);
    println!("Binary phase: 0.{}", binary_result);
    println!("Decimal phase: {:.8}", phase);
    println!("Error: {:.8}", (phase - 0.5).abs());
    
    // Show binary representation visually
    print!("Visual: |");
    for &bit in &results.results {
        print!("{}", bit);
    }
    println!("|");
    
    // Show phase as fraction
    if near(phase, 0.5, 0.01) {
        println!("Fraction: 1/2");
    } else if near(phase, 0.25, 0.01) {
        println!("Fraction: 1/4");
    } else if near(phase, 0.75, 0.01) {
        println!("Fraction: 3/4");
    } else {
        println!("Fraction: {}/{}", 
            (phase * (1 << precision) as f64).round() as usize, 
            1 << precision);
    }
}

// Function to estimate phase of T-gate (phase 0.125 = 1/8)
fn test_t_gate_phase(precision: usize) {
    println!("\nRunning QPE with {} precision qubits:", precision);
    
    // Initialize QPE algorithm for T gate with desired precision
    // and 1 target qubit
    let mut qpe = QuantumPhaseEstimation::new(
        precision,     // number of precision qubits
        1,             // 1 target qubit
        t_gate_phase   // controlled-T gate function
    );
    
    // Initialize target qubit to |1⟩ state (eigenstate of T)
    qpe = qpe.with_target_initialization(|reg, phase_qubits, _| {
        // Set the target qubit to |1⟩
        reg.x(phase_qubits);
    });
    
    // Run the algorithm
    let results = qpe.run();
    
    // Display results
    let binary_result = results.results.iter()
        .map(|&b| b.to_string())
        .collect::<String>();
    
    // Calculate the phase from binary result
    let mut phase = 0.0;
    for (i, &bit) in results.results.iter().enumerate() {
        phase += (bit as f64) / (1 << (i+1)) as f64;
    }
    
    println!("Precision: {} qubits", precision);
    println!("Binary phase: 0.{}", binary_result);
    println!("Decimal phase: {:.8}", phase);
    println!("Error: {:.8}", (phase - 0.125).abs());
    
    // Show binary representation visually
    print!("Visual: |");
    for &bit in &results.results {
        print!("{}", bit);
    }
    println!("|");
    
    // Show phase as fraction
    if near(phase, 0.125, 0.01) {
        println!("Fraction: 1/8");
    } else {
        println!("Fraction: {}/{}", 
            (phase * (1 << precision) as f64).round() as usize, 
            1 << precision);
    }
}

// Controlled-Z gate implementation for phase estimation
fn z_gate_phase(reg: &mut QuantumRegister, control: usize, target: usize, power: usize) {
    // Apply controlled-Z gate if power is odd
    // Z^(even power) is identity, so we don't need to apply anything
    if power % 2 == 1 {
        reg.cz(control, target);
        
        // Print debug info
        println!("  Applied Z^{} controlled by qubit {}", power, control);
    } else {
        // For even powers, Z^power = I (identity)
        println!("  Z^{} = I, no operation needed", power);
    }
}

// Controlled-T gate implementation for phase estimation
fn t_gate_phase(reg: &mut QuantumRegister, control: usize, target: usize, power: usize) {
    // T gate has eigenvalue e^(i*pi/4) on |1⟩
    // For power p, phase is (p*pi/4) mod 2pi
    
    // Apply appropriate phase based on power
    let angle = PI * (power % 8) as f64 / 4.0;
    
    // Apply controlled phase
    if reg.qubit(control).unwrap().prob_1() > 0.01 {
        if reg.qubit(target).unwrap().prob_1() > 0.99 {
            // Target in |1⟩ - apply phase
            reg.phase(target, angle);
            
            // Also apply phase kickback to control
            let control_qubit = reg.qubit(control).unwrap();
            let (alpha_c, beta_c) = control_qubit.get_coeffs();
            
            // Calculate new phase for |1⟩ component based on angle
            let phase_factor = (angle.cos(), angle.sin());
            reg.qubit_mut(control).unwrap().set_coeffs(
                alpha_c,
                (beta_c.0 * phase_factor.0 - beta_c.1 * phase_factor.1,
                 beta_c.0 * phase_factor.1 + beta_c.1 * phase_factor.0)
            );
        }
    }
}

// Helper function to check if two values are approximately equal
fn near(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
} 