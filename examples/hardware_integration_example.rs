// Hardware Integration Example
//
// This example demonstrates how to use the hardware integration features
// of the quantum protocols library, including fallback mechanisms when
// real hardware is unavailable.

use quantum_protocols::integration::qhep_hardware::{
    HardwareRegistry,
    HardwareConnectionConfig,
    AuthMethod
};
use std::collections::HashMap;

fn main() {
    match run_example() {
        Ok(_) => println!("Example completed successfully!"),
        Err(e) => eprintln!("Error: {}", e),
    }
}

fn run_example() -> Result<(), String> {
    println!("Quantum Hardware Integration Example");
    println!("===================================");
    
    // Create a hardware registry with fallback enabled
    let mut registry = HardwareRegistry::new();
    println!("Created hardware registry with automatic fallback enabled");

    // Step 1: Attempt to connect to real hardware (which will likely fail)
    println!("\nStep 1: Attempting to connect to IBM Quantum hardware...");
    let ibm_config = create_ibm_config();
    
    match registry.connect_hardware("ibmq", &ibm_config) {
        Ok(id) => {
            println!("✓ Successfully connected to IBM Quantum hardware with ID: {}", id);
            
            // Set as default executor
            registry.set_default_executor(&id)
                .map_err(|e| format!("Failed to set IBM Quantum as default: {}", e))?;
            println!("✓ Set IBM Quantum as default executor");
        },
        Err(e) => {
            println!("✗ Failed to connect to IBM Quantum hardware: {}", e);
            println!("  (This is expected without valid credentials)");
        }
    }
    
    // Step 2: Try another provider (IonQ) which will also likely fail
    println!("\nStep 2: Attempting to connect to IonQ hardware...");
    let ionq_config = create_ionq_config();
    
    match registry.connect_hardware("ionq", &ionq_config) {
        Ok(id) => {
            println!("✓ Successfully connected to IonQ hardware with ID: {}", id);
            // Set as default if IBM failed
            if registry.get_default_executor().is_err() {
                registry.set_default_executor(&id)
                    .map_err(|e| format!("Failed to set IonQ as default: {}", e))?;
                println!("✓ Set IonQ as default executor");
            }
        },
        Err(e) => {
            println!("✗ Failed to connect to IonQ hardware: {}", e);
            println!("  (This is expected without valid credentials)");
        }
    }
    
    // Step 3: Manually connect to simulator
    println!("\nStep 3: Connecting to quantum simulator...");
    let sim_config = HardwareConnectionConfig::default();
    let sim_id = registry.connect_hardware("simulator", &sim_config)
        .map_err(|e| format!("Failed to connect to simulator: {}", e))?;
    println!("✓ Successfully connected to quantum simulator with ID: {}", sim_id);
    
    // Step 4: Check if we have a default executor
    // If not, set simulator as default
    println!("\nStep 4: Checking default hardware executor...");
    match registry.get_default_executor() {
        Ok(_) => {
            println!("✓ Default executor already set");
        },
        Err(_) => {
            registry.set_default_executor(&sim_id)
                .map_err(|e| format!("Failed to set simulator as default: {}", e))?;
            println!("✓ Set simulator as default executor");
        }
    }
    
    // Step 5: Execute a circuit on the default executor
    println!("\nStep 5: Executing quantum circuit on default executor...");
    let executor = registry.get_default_executor()
        .map_err(|e| format!("Failed to get default executor: {}", e))?;
    
    // Define a simple bell-pair circuit
    let circuit = vec![
        "H 0".to_string(),            // Hadamard on qubit 0
        "CNOT 0 1".to_string(),       // CNOT with control qubit 0, target qubit 1
        "MEASURE 0 1".to_string(),    // Measure qubits 0 and 1
    ];
    
    // Execute the circuit
    println!("Executing Bell pair circuit: H 0; CNOT 0 1; MEASURE 0 1");
    let qubit_mapping = vec![0, 1];
    let result = executor.execute_circuit(&circuit, &qubit_mapping)
        .map_err(|e| format!("Circuit execution failed: {}", e))?;
    
    // Display results
    println!("Circuit executed successfully!");
    println!("Result bytes (hex): {}", result.iter()
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<String>>()
        .join(" "));
    
    // Step 6: Get hardware status
    println!("\nStep 6: Getting hardware status information...");
    let status = executor.get_status()
        .map_err(|e| format!("Failed to get hardware status: {}", e))?;
    
    println!("Hardware Status:");
    println!("  - Online: {}", status.online);
    println!("  - Utilization: {:.1}%", status.utilization * 100.0);
    println!("  - Queue length: {}", status.queue_length);
    println!("  - Estimated wait: {} seconds", status.estimated_wait_sec);
    println!("  - Available qubits: {}", status.available_qubits);
    
    // Print some error rates
    println!("Error Rates:");
    for (gate, error) in status.error_rates.iter().take(3) {
        println!("  - {} gate: {:.4}", gate, error);
    }
    
    // Step 7: Disconnect all hardware
    println!("\nStep 7: Disconnecting from all hardware...");
    registry.disconnect_all()
        .map_err(|e| format!("Failed to disconnect: {}", e))?;
    println!("✓ Successfully disconnected from all hardware");
    
    // Step 8: Demonstrate fallback mechanism with forced error
    println!("\nStep 8: Demonstrating automatic fallback mechanism...");
    println!("Attempting to connect to invalid hardware with fallback enabled...");
    
    // Try to connect to a non-existent provider, which will trigger fallback
    let result = registry.connect_hardware("invalid_provider", &HardwareConnectionConfig::default());
    
    match result {
        Ok(id) => {
            println!("✓ Fallback successful! Connected to fallback simulator with ID: {}", id);
            
            // Check if this is really a simulator
            let executor = registry.get_executor(&id)
                .map_err(|e| format!("Failed to get executor: {}", e))?;
            println!("Provider of fallback hardware: {}", executor.provider_name());
            
            // Execute a simple circuit on the fallback
            let simple_circuit = vec!["H 0".to_string(), "MEASURE 0".to_string()];
            let result = executor.execute_circuit(&simple_circuit, &[0])
                .map_err(|e| format!("Circuit execution failed: {}", e))?;
            println!("Successfully executed circuit on fallback hardware");
            println!("Result bytes (hex): {}", result.iter()
                .map(|b| format!("{:02x}", b))
                .collect::<Vec<String>>()
                .join(" "));
        },
        Err(e) => {
            println!("✗ Fallback failed: {}", e);
        }
    }
    
    // Step 9: Create a custom simulator with specific noise model
    println!("\nStep 9: Creating custom simulator with specific noise model...");
    
    // Create a custom noise model with high error rates
    let mut gate_errors = HashMap::new();
    gate_errors.insert("x".to_string(), 0.05);  // 5% error on X gates
    gate_errors.insert("cx".to_string(), 0.10); // 10% error on CNOT gates
    
    // Build noise parameters directly into the options
    let qubit_count = 3;  // 3-qubit simulator
    
    // Create config for custom noisy simulator
    println!("Creating custom noisy simulator");
    let mut custom_options = HashMap::new();
    custom_options.insert("noise_model".to_string(), "custom".to_string());
    custom_options.insert("qubit_count".to_string(), qubit_count.to_string());
    custom_options.insert("error_rate_x".to_string(), "0.05".to_string());
    custom_options.insert("error_rate_cx".to_string(), "0.10".to_string());
    custom_options.insert("measurement_error".to_string(), "0.08".to_string());
    custom_options.insert("t1_us".to_string(), "20.0".to_string());
    custom_options.insert("t2_us".to_string(), "15.0".to_string());
    custom_options.insert("crosstalk".to_string(), "0.01".to_string());
    
    // Create custom simulator config
    let custom_config = HardwareConnectionConfig {
        provider: "simulator".to_string(),
        endpoint: "local".to_string(),
        auth: AuthMethod::ApiKey("simulator_token".to_string()),
        timeout_ms: 1000,
        secure: false,
        options: custom_options,
    };
    
    // Connect to custom simulator through regular hardware connection
    let custom_id = registry.connect_hardware("custom_noisy_sim", &custom_config)
        .map_err(|e| format!("Failed to connect to custom simulator: {}", e))?;
    
    // Get the executor
    let custom_executor = registry.get_executor(&custom_id)
        .map_err(|e| format!("Failed to get custom executor: {}", e))?;
    
    // Execute a simple circuit on the custom simulator
    let circuit = vec![
        "H 0".to_string(),
        "CNOT 0 1".to_string(),
        "CNOT 1 2".to_string(),
        "MEASURE 0 1 2".to_string(),
    ];
    
    println!("Executing GHZ state circuit on noisy simulator");
    println!("(H 0; CNOT 0 1; CNOT 1 2; MEASURE 0 1 2)");
    
    let results = custom_executor.execute_circuit(&circuit, &[0, 1, 2])
        .map_err(|e| format!("Circuit execution failed: {}", e))?;
    println!("Circuit executed on custom noisy simulator");
    println!("Result bytes (hex): {}", results.iter()
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<String>>()
        .join(" "));
    
    // Display the status of the custom simulator
    let status = custom_executor.get_status()
        .map_err(|e| format!("Failed to get simulator status: {}", e))?;
    println!("Custom simulator error rates:");
    for (gate, error) in status.error_rates.iter().take(3) {
        println!("  - {} gate: {:.4}", gate, error);
    }
    
    println!("\n✓ Example completed successfully!");
    
    Ok(())
}

// Create configuration for IBM Quantum hardware
fn create_ibm_config() -> HardwareConnectionConfig {
    // In a real application, you would provide actual credentials
    HardwareConnectionConfig {
        provider: "ibmq".to_string(),
        endpoint: "https://api.quantum-computing.ibm.com/api".to_string(),
        auth: AuthMethod::ApiKey("YOUR_IBM_API_KEY_HERE".to_string()),
        timeout_ms: 10000,
        secure: true,
        options: {
            let mut options = HashMap::new();
            options.insert("backend".to_string(), "ibmq_manila".to_string());
            options.insert("hub".to_string(), "ibm-q".to_string());
            options
        },
    }
}

// Create configuration for IonQ hardware
fn create_ionq_config() -> HardwareConnectionConfig {
    // In a real application, you would provide actual credentials
    HardwareConnectionConfig {
        provider: "ionq".to_string(),
        endpoint: "https://api.ionq.co/v0.2".to_string(),
        auth: AuthMethod::ApiKey("YOUR_IONQ_API_KEY_HERE".to_string()),
        timeout_ms: 8000,
        secure: true,
        options: HashMap::new(),
    }
} 