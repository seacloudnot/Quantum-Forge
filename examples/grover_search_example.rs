// Example of using Grover's search algorithm
//
// This demonstrates how to use the GroverSearch quantum algorithm
// to find a marked item in an unsorted database.

use quantum_protocols::core::GroverSearch;

fn main() {
    println!("=== Grover's Search Algorithm Example ===");
    println!();
    
    // Define an oracle function that marks a specific state |110⟩ (binary 6)
    // The oracle returns true for the state we want to find
    let oracle = |bits: &[u8]| {
        if bits.len() < 3 {
            return false;
        }
        
        bits[0] == 0 && bits[1] == 1 && bits[2] == 1
    };
    
    // Create a new Grover search instance with 3 qubits
    let mut grover = GroverSearch::new(3, oracle);
    
    // Display information about the search
    println!("Searching for state |110⟩ (binary representation, with LSB first)");
    println!("Search space size: 2^3 = 8 states");
    println!();
    
    // Run the algorithm
    let result = grover.run();
    
    // Display results
    println!("Algorithm: {:?}", result.algorithm);
    println!("Success: {}", result.success);
    println!("Raw measurement results: {:?}", result.results);
    
    // Convert binary result to decimal (bits are LSB first)
    let decimal_result = result.results.iter()
        .enumerate()
        .fold(0, |acc, (i, &bit)| acc + (bit as usize) * (1 << i));
    
    println!("Measured result (decimal): {}", decimal_result);
    println!("Measured result (binary): {:03b}", decimal_result);
    
    // Display statistics
    println!();
    println!("Statistics:");
    println!("  Qubits used: {}", result.statistics.qubits_used);
    println!("  Gate count: {}", result.statistics.gate_count);
    println!("  Oracle calls: {}", result.statistics.oracle_calls);
    println!("  Theoretical success probability: {:.4}", 
             result.statistics.theoretical_success_prob);
    
    // Run again with more qubits for comparison
    println!();
    println!("=== Repeating search with 6 qubits ===");
    
    // Define an oracle function that marks a specific state |110000⟩ (binary 6)
    let oracle_6q = |bits: &[u8]| {
        if bits.len() < 6 {
            return false;
        }
        
        bits[0] == 0 && bits[1] == 1 && bits[2] == 1 &&
        bits[3] == 0 && bits[4] == 0 && bits[5] == 0
    };
    
    // Create a new Grover search instance with 6 qubits
    let mut grover_6q = GroverSearch::new(6, oracle_6q);
    
    // Display information about the search
    println!("Searching for state |110000⟩");
    println!("Search space size: 2^6 = 64 states");
    println!();
    
    // Run the algorithm
    let result_6q = grover_6q.run();
    
    // Convert binary result to decimal (bits are LSB first)
    let decimal_result_6q = result_6q.results.iter()
        .enumerate()
        .fold(0, |acc, (i, &bit)| acc + (bit as usize) * (1 << i));
    
    println!("Measured result (decimal): {}", decimal_result_6q);
    println!("Measured result (binary): {:06b}", decimal_result_6q);
    
    println!();
    println!("Statistics:");
    println!("  Qubits used: {}", result_6q.statistics.qubits_used);
    println!("  Gate count: {}", result_6q.statistics.gate_count);
    println!("  Oracle calls: {}", result_6q.statistics.oracle_calls);
    println!("  Theoretical success probability: {:.4}", 
             result_6q.statistics.theoretical_success_prob);
    
    // Note: As the number of qubits increases, Grover's algorithm provides
    // a quadratic speedup over classical search algorithms
    println!();
    println!("Notice how the number of iterations (oracle calls) scales with √N");
    println!("rather than N, demonstrating quantum speedup.");
} 