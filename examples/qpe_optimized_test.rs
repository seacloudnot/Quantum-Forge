// Optimized Quantum Phase Estimation Test
//
// This example tests the optimized QPE implementation with different unitary gates
// and compares the different phase interpretations.

use quantum_protocols::core::{QuantumPhaseEstimation, QuantumRegister};
use std::f64::consts::PI;

fn main() {
    println!("=== Optimized Quantum Phase Estimation Test ===\n");
    
    // Test with different gates
    println!("Testing different gates with optimized QPE:");
    println!("-------------------------------------------");
    
    // Z gate phase (0.5)
    test_gate_phase("Z gate", 0.5, 5, z_gate_phase);
    
    // S gate phase (0.25)
    test_gate_phase("S gate", 0.25, 5, s_gate_phase);
    
    // T gate phase (0.125)
    test_gate_phase("T gate", 0.125, 6, t_gate_phase);
    
    // Custom phase (0.3) for comparison
    test_gate_phase("Custom φ=0.3", 0.3, 6, |reg, control, target, power| {
        custom_phase_gate(reg, control, target, power, 0.3)
    });
    
    // Compare performance with different precision levels
    println!("\n\nTesting precision scaling:");
    println!("-------------------------------------------");
    
    // Find the minimum precision needed to resolve phase 0.3 correctly
    for precision in 4..10 {
        test_precision_for_phase(0.3, precision);
    }
}

// Global phase value for testing
static mut CUSTOM_PHASE: f64 = 0.0;

// Test phase estimation for a given gate
fn test_gate_phase(gate_name: &str, expected_phase: f64, precision: usize, 
                   gate_fn: fn(&mut QuantumRegister, usize, usize, usize)) {
    println!("\n--- {} (phase = {:.6}) ---", gate_name, expected_phase);
    
    // Initialize QPE with the specified gate and precision
    let mut qpe = QuantumPhaseEstimation::new(
        precision,   // precision qubits
        1,           // 1 target qubit
        gate_fn      // controlled-gate function
    );
    
    // Initialize target qubit to |1⟩ state (eigenstate)
    qpe = qpe.with_target_initialization(|reg, phase_qubits, _| {
        reg.x(phase_qubits);
    });
    
    // Run QPE
    let results = qpe.run();
    
    // Get measurement results as binary string
    let binary = results.results.iter()
        .map(|&b| b.to_string())
        .collect::<String>();
    
    // Get both phase interpretations
    let (lsb_phase, msb_phase) = qpe.phase_interpretations();
    
    // Determine which interpretation is more accurate
    let lsb_error = (lsb_phase - expected_phase).abs();
    let msb_error = (msb_phase - expected_phase).abs();
    let complement_error = (1.0 - lsb_phase - expected_phase).abs();
    
    // Format binary fraction for expected phase
    let expected_binary = format_binary_fraction(expected_phase, precision);
    
    // Display results
    println!("Precision: {} qubits", precision);
    println!("Measured bits: {}", binary);
    println!("LSB-first phase: {:.8} (error: {:.8})", lsb_phase, lsb_error);
    println!("MSB-first phase: {:.8} (error: {:.8})", msb_phase, msb_error);
    println!("Complement phase: {:.8} (error: {:.8})", 1.0 - lsb_phase, complement_error);
    println!("Expected binary: {}", expected_binary);
    
    // Report best interpretation
    let (best_phase, best_error, interpretation) = if lsb_error <= msb_error && lsb_error <= complement_error {
        (lsb_phase, lsb_error, "LSB-first")
    } else if msb_error <= lsb_error && msb_error <= complement_error {
        (msb_phase, msb_error, "MSB-first")
    } else {
        (1.0 - lsb_phase, complement_error, "Complement")
    };
    
    println!("Best interpretation: {} = {:.8} (error: {:.8})", 
             interpretation, best_phase, best_error);
}

// Test how precision affects phase estimation accuracy
fn test_precision_for_phase(phase: f64, precision: usize) {
    println!("\nTesting phase {:.6} with {} qubits:", phase, precision);
    
    // Set the global phase value
    unsafe { CUSTOM_PHASE = phase; }
    
    let mut qpe = QuantumPhaseEstimation::new(
        precision,
        1,
        global_custom_phase_gate // Use global function instead of closure
    );
    
    // Initialize target to |1⟩
    qpe = qpe.with_target_initialization(|reg, phase_qubits, _| {
        reg.x(phase_qubits);
    });
    
    // Run QPE
    let results = qpe.run();
    
    // Get both interpretations
    let (lsb_phase, msb_phase) = qpe.phase_interpretations();
    
    // Calculate errors
    let lsb_error = (lsb_phase - phase).abs();
    let msb_error = (msb_phase - phase).abs();
    
    // Determine best interpretation
    let (best_phase, best_error, interpretation) = if lsb_error <= msb_error {
        (lsb_phase, lsb_error, "LSB-first")
    } else {
        (msb_phase, msb_error, "MSB-first")
    };
    
    // Calculate theoretical error bound (1/2^precision)
    let theoretical_error = 1.0 / (1 << precision) as f64;
    
    // Check if we're within the theoretical bound
    let within_bound = best_error <= theoretical_error;
    
    println!("Best estimate: {:.8} ({} interpretation)", best_phase, interpretation);
    println!("Error: {:.8}", best_error);
    println!("Theoretical error bound: {:.8}", theoretical_error);
    println!("Within theoretical bound: {}", if within_bound { "YES" } else { "NO" });
}

// Global function for custom phase gate
fn global_custom_phase_gate(reg: &mut QuantumRegister, control: usize, target: usize, power: usize) {
    // Get the phase from global variable
    let phase = unsafe { CUSTOM_PHASE };
    
    // Apply controlled phase rotation
    let angle = (power as f64) * 2.0 * PI * phase;
    apply_controlled_phase(reg, control, target, angle);
}

// Z gate phase (0.5)
fn z_gate_phase(reg: &mut QuantumRegister, control: usize, target: usize, power: usize) {
    // Z^odd = Z, Z^even = I
    if power % 2 == 1 {
        reg.cz(control, target);
    }
}

// S gate phase (0.25)
fn s_gate_phase(reg: &mut QuantumRegister, control: usize, target: usize, power: usize) {
    // S gate has phase π/2 (0.25)
    // Apply controlled phase rotation based on power
    let phase = (power as f64) * PI / 2.0;
    
    // Only apply if control is |1⟩ and target is |1⟩
    apply_controlled_phase(reg, control, target, phase);
}

// T gate phase (0.125)
fn t_gate_phase(reg: &mut QuantumRegister, control: usize, target: usize, power: usize) {
    // T gate has phase π/4 (0.125)
    // Apply controlled phase rotation based on power
    let phase = (power as f64) * PI / 4.0;
    
    // Only apply if control is |1⟩ and target is |1⟩
    apply_controlled_phase(reg, control, target, phase);
}

// Custom phase gate
fn custom_phase_gate(reg: &mut QuantumRegister, control: usize, target: usize, 
                     power: usize, phase: f64) {
    // Apply controlled phase rotation based on power and custom phase
    let angle = (power as f64) * 2.0 * PI * phase;
    
    // Only apply if control is |1⟩ and target is |1⟩
    apply_controlled_phase(reg, control, target, angle);
}

// Helper to apply controlled phase rotation
fn apply_controlled_phase(reg: &mut QuantumRegister, control: usize, target: usize, angle: f64) {
    // Check if control and target are in bounds
    if control >= reg.size() || target >= reg.size() || control == target {
        return;
    }
    
    // Check if target is in |1⟩ state (needs phase rotation)
    let target_in_one = reg.qubit(target).map_or(false, |q| q.prob_1() > 0.9);
    if !target_in_one {
        return;
    }
    
    // Apply controlled phase
    // This has two effects:
    // 1. Apply phase to target's |1⟩ component when control is |1⟩
    // 2. Apply phase kickback to control's |1⟩ component
    
    // Get control qubit coefficients
    if let Some(q) = reg.qubit(control) {
        let (alpha, beta) = q.get_coeffs();
        
        // Apply phase to control's |1⟩ component (phase kickback)
        // e^(i*angle) = cos(angle) + i*sin(angle)
        let phase_factor = (angle.cos(), angle.sin());
        
        // New beta = beta * e^(i*angle)
        let new_beta = (
            beta.0 * phase_factor.0 - beta.1 * phase_factor.1,
            beta.0 * phase_factor.1 + beta.1 * phase_factor.0
        );
        
        // Update control qubit
        reg.qubit_mut(control).unwrap().set_coeffs(alpha, new_beta);
        
        // Apply phase to target qubit if control has |1⟩ component
        if beta.0.abs() > 0.01 || beta.1.abs() > 0.01 {
            if let Some(target_q) = reg.qubit(target) {
                let (alpha_t, beta_t) = target_q.get_coeffs();
                let new_beta_t = (
                    beta_t.0 * phase_factor.0 - beta_t.1 * phase_factor.1,
                    beta_t.0 * phase_factor.1 + beta_t.1 * phase_factor.0
                );
                reg.qubit_mut(target).unwrap().set_coeffs(alpha_t, new_beta_t);
            }
        }
    }
}

// Format a decimal as a binary fraction
fn format_binary_fraction(value: f64, precision: usize) -> String {
    let mut result = String::from("0.");
    let mut val = value;
    
    for _ in 0..precision {
        val *= 2.0;
        if val >= 1.0 {
            result.push('1');
            val -= 1.0;
        } else {
            result.push('0');
        }
    }
    
    result
} 