// Enhanced Fallback Example
//
// This example demonstrates the enhanced fallback mechanism with retry capabilities
// for quantum hardware connections, ensuring robustness in real-world scenarios.

use quantum_protocols::integration::qhep_hardware::{
    HardwareRegistry,
    HardwareConnectionConfig,
    AuthMethod,
    FallbackConfig
};
use std::collections::HashMap;

fn main() {
    match run_example() {
        Ok(_) => println!("Example completed successfully!"),
        Err(e) => eprintln!("Error: {}", e),
    }
}

fn run_example() -> Result<(), String> {
    println!("Enhanced Fallback Mechanism Example");
    println!("==================================\n");
    
    // Create a custom fallback configuration
    let fallback_config = FallbackConfig {
        enable_fallback: true,
        max_failures: 2,
        initial_retry_delay_ms: 500,
        max_retry_delay_ms: 5000,
        backoff_multiplier: 2.0,
        log_fallback_events: true,
    };
    
    // Create a hardware registry with custom fallback config
    let mut registry = HardwareRegistry::new_with_fallback_config(fallback_config);
    println!("Created hardware registry with enhanced fallback configuration");
    println!("Initial backoff: {initial_ms}ms, Max backoff: {max_ms}ms, Multiplier: {multiplier}x", 
             initial_ms = registry.fallback_config().initial_retry_delay_ms,
             max_ms = registry.fallback_config().max_retry_delay_ms,
             multiplier = registry.fallback_config().backoff_multiplier);
    
    // Step 1: Demonstrate automatic fallback
    println!("\nStep 1: Demonstrating automatic fallback mechanism");
    println!("Attempting to connect to IBM Quantum hardware (will fail and trigger fallback)...");
    
    // Create a config with invalid credentials to trigger a failure
    let invalid_config = HardwareConnectionConfig {
        provider: "ibmq".to_string(),
        endpoint: "https://invalid-endpoint.example.com".to_string(),
        auth: AuthMethod::ApiKey("INVALID_KEY".to_string()),
        timeout_ms: 2000,
        secure: true,
        options: HashMap::new(),
    };
    
    // Try to connect, which should fail and trigger fallback
    match registry.connect_hardware("ibmq", &invalid_config) {
        Ok(id) => {
            println!("✓ Connected to hardware with ID: {id}");
            
            // Check if this is a fallback simulator
            let executor = registry.get_executor(&id)
                .map_err(|e| format!("Failed to get executor: {}", e))?;
            
            if id.contains("fallback") {
                println!("✓ Automatic fallback to simulator successful!");
                println!("  Provider of fallback hardware: {}", executor.provider_name());
            } else {
                println!("Connected to real hardware: {}", executor.provider_name());
            }
        },
        Err(e) => {
            println!("✗ Connection failed and fallback didn't trigger: {e}");
        }
    }
    
    // Step 2: Demonstrate retry mechanism
    println!("\nStep 2: Demonstrating connection with retry mechanism");
    println!("Attempting to connect to IonQ hardware with retries (will eventually fall back)...");
    
    // Create another invalid config
    let another_invalid_config = HardwareConnectionConfig {
        provider: "ionq".to_string(),
        endpoint: "https://invalid-endpoint.example.com".to_string(),
        auth: AuthMethod::ApiKey("INVALID_KEY".to_string()),
        timeout_ms: 2000,
        secure: true,
        options: HashMap::new(),
    };
    
    // Try connecting with retries
    match registry.connect_hardware_with_retry("ionq", &another_invalid_config, 3) {
        Ok(id) => {
            println!("✓ Connected to hardware with ID: {id}");
            
            // Check if this is a fallback simulator
            let executor = registry.get_executor(&id)
                .map_err(|e| format!("Failed to get executor: {}", e))?;
            
            if id.contains("fallback") {
                println!("✓ Fallback to simulator after retries successful!");
                println!("  Provider of fallback hardware: {}", executor.provider_name());
            } else {
                println!("Connected to real hardware: {}", executor.provider_name());
            }
            
            // Get information about the last fallback
            if let Some((provider, fallback_id, _timestamp)) = registry.last_fallback() {
                println!("  Last fallback: {provider} -> {fallback_id}");
            }
        },
        Err(e) => {
            println!("✗ Connection failed after retries: {e}");
        }
    }
    
    // Step 3: Demonstrate connections to multiple providers with fallbacks
    println!("\nStep 3: Connecting to multiple providers with automatic fallbacks");
    
    let providers = vec!["ibmq", "ionq", "rigetti"];
    let mut connected_ids = Vec::new();
    
    for provider in providers {
        println!("Attempting to connect to {provider}...");
        
        // Create invalid config to trigger fallback
        let config = HardwareConnectionConfig {
            provider: provider.to_string(),
            endpoint: format!("https://{}-api.example.com", provider),
            auth: AuthMethod::ApiKey("INVALID_KEY".to_string()),
            timeout_ms: 1000,
            secure: true,
            options: HashMap::new(),
        };
        
        match registry.connect_hardware(provider, &config) {
            Ok(id) => {
                println!("✓ Connected to {provider} (with ID: {id})");
                connected_ids.push(id);
            },
            Err(e) => {
                println!("✗ Failed to connect to {provider}: {e}");
            }
        }
    }
    
    // Step 4: Connect to a simulator directly (should always succeed)
    println!("\nStep 4: Connecting to simulator directly");
    
    let sim_config = HardwareConnectionConfig::default();
    let sim_id = registry.connect_hardware("simulator", &sim_config)
        .map_err(|e| format!("Failed to connect to simulator: {}", e))?;
    
    println!("✓ Connected to simulator with ID: {}", sim_id);
    
    // Step 5: Execute a simple circuit on the simulator
    println!("\nStep 5: Executing quantum circuit on simulator");
    
    let executor = registry.get_executor(&sim_id)
        .map_err(|e| format!("Failed to get executor: {}", e))?;
    
    // Define a simple bell-pair circuit
    let circuit = vec![
        "H 0".to_string(),
        "CNOT 0 1".to_string(),
        "MEASURE 0 1".to_string(),
    ];
    
    // Execute the circuit
    println!("Executing Bell pair circuit");
    let result = executor.execute_circuit(&circuit, &[0, 1])
        .map_err(|e| format!("Circuit execution failed: {}", e))?;
    
    println!("✓ Circuit executed successfully");
    println!("  Result bytes (hex): {}", result.iter()
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<String>>()
        .join(" "));
    
    // Step 6: Disconnect all hardware
    println!("\nStep 6: Disconnecting from all hardware");
    registry.disconnect_all()
        .map_err(|e| format!("Failed to disconnect: {}", e))?;
    println!("✓ Successfully disconnected from all hardware");
    
    println!("\nEnhanced fallback example completed successfully!");
    
    Ok(())
} 