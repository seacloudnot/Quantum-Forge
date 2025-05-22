// Direct manual implementation of Quantum Phase Estimation
//
// This example directly implements the QPE circuit for a Z gate
// without using the library's QPE implementation.

use quantum_protocols::core::QuantumRegister;
use std::f64::consts::PI;

fn main() {
    println!("=== Manual Phase Estimation for Z Gate ===\n");
    
    // The phase of Z gate should be 0.5
    println!("Testing Z gate with phase 0.5\n");
    
    // Try with different precision levels
    for precision in 3..8 {
        test_phase_direct(precision);
    }
}

fn test_phase_direct(precision: usize) {
    println!("\n--- QPE with {} precision qubits ---", precision);
    
    // Create register with precision qubits + 1 target qubit
    let mut reg = QuantumRegister::new(precision + 1);
    
    // Initialize target qubit (last qubit) to |1⟩ state (eigenstate of Z)
    reg.x(precision);
    println!("Target qubit initialized to |1⟩ state");
    
    // Step 1: Apply Hadamard to all control qubits
    for i in 0..precision {
        reg.hadamard(i);
    }
    println!("Applied Hadamard to all control qubits");
    
    // Step 2: Apply controlled-Z operations
    // For qubit j, we apply Z^(2^j)
    for j in 0..precision {
        // Z^(2^j) means Z if j=0, Z^2 if j=1, Z^4 if j=2, etc.
        // Since Z^2 = I (identity), we only need to apply Z when 2^j is odd
        let power = 1 << j; // 2^j
        if power % 2 == 1 {
            // Only when power is odd, apply Z (controlled by qubit j)
            reg.cz(j, precision);
            println!("Applied controlled-Z^{} from qubit {} to target", power, j);
        } else {
            println!("Skipped Z^{} since it equals identity", power);
        }
    }
    
    // After controlled-Z operations, only qubit 0 should have a phase change
    // The register should be in the state:
    // (|0⟩ + e^(iπ)|1⟩)|0...0⟩ ⊗ |1⟩ / sqrt(2^precision)
    // = (|0⟩ - |1⟩)|0...0⟩ ⊗ |1⟩ / sqrt(2^precision)
    println!("After controlled operations, first qubit should have phase π");
    
    // Step 3: Apply inverse QFT to control qubits
    println!("Applying inverse QFT to control qubits");
    
    // Inverse QFT implementation
    for i in 0..precision {
        // Apply controlled phase rotations before Hadamard
        for j in 0..i {
            let angle = -PI / (1 << (i - j)) as f64;
            if reg.qubit(j).unwrap().prob_1() > 0.01 {
                reg.phase(i, angle);
                println!("  Applied phase rotation {:.4} to qubit {}", angle, i);
            }
        }
        
        // Apply Hadamard
        reg.hadamard(i);
        println!("  Applied Hadamard to qubit {}", i);
    }
    
    // Swap qubits for proper ordering
    for i in 0..(precision / 2) {
        reg.swap(i, precision - i - 1);
    }
    
    // Measure results
    let measurements = (0..precision)
        .map(|i| reg.measure(i).unwrap())
        .collect::<Vec<_>>();
    
    // Calculate phase from measurements
    // For phase 0.5, qubit 0 should be 1 (MSB) and all others 0
    // This corresponds to binary 0.1 (base 2) = 0.5 (base 10)
    let mut phase = 0.0;
    for (i, &bit) in measurements.iter().enumerate() {
        phase += (bit as f64) / (1 << (i+1)) as f64;
    }
    
    // Check each qubit's state before measurement
    println!("\nQubit probabilities before measurement:");
    for i in 0..precision {
        if let Some(q) = reg.qubit(i) {
            println!("  Qubit {}: |0⟩={:.4}, |1⟩={:.4}", 
                    i, q.prob_0(), q.prob_1());
        }
    }
    
    // Display results
    println!("\nMeasurement results:");
    print!("  Binary: |");
    for bit in &measurements {
        print!("{}", bit);
    }
    println!("|");
    println!("  Phase (LSB first): {:.8} (expected: 0.5)", phase);
    
    // Calculate phase with reversed bit order (MSB first)
    let mut msb_phase = 0.0;
    for (i, &bit) in measurements.iter().rev().enumerate() {
        msb_phase += (bit as f64) / (1 << (i+1)) as f64;
    }
    println!("  Phase (MSB first): {:.8} (expected: 0.5)", msb_phase);
    
    // Show expected values for this precision
    println!("\nExpected binary for phase 0.5 with {} qubits:", precision);
    print!("  |");
    let mut expected = Vec::new();
    for i in 0..precision {
        if i == 0 {
            print!("1");
            expected.push(1);
        } else {
            print!("0");
            expected.push(0);
        }
    }
    println!("|");
    
    // Calculate expected phase
    let mut expected_phase = 0.0;
    for (i, &bit) in expected.iter().enumerate() {
        expected_phase += (bit as f64) / (1 << (i+1)) as f64;
    }
    println!("  Expected phase: {:.8}", expected_phase);
}

// Note: For phase 0.5, the binary representation is 0.1 (binary)
// So with 3 qubits of precision, we expect to measure |100| (MSB first)
// With 4 qubits, we expect |1000|, etc. 