// QHEP Hardware Integration Example
//
// This example demonstrates how to use the QHEP hardware integration module
// to connect to various quantum hardware providers.

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
    println!("QHEP Hardware Integration Example");
    println!("================================\n");
    
    // Create a hardware registry
    let mut registry = HardwareRegistry::new();
    println!("Created hardware registry");
    
    // Connect to a simulator first (always available)
    println!("\nConnecting to local quantum simulator...");
    let simulator_config = HardwareConnectionConfig::default();
    let simulator_id = registry.connect_hardware("simulator", &simulator_config)
        .map_err(|e| format!("Failed to connect to simulator: {}", e))?;
    println!("  Connected to simulator with ID: {}", simulator_id);
    
    // Test simulator capabilities
    println!("Discovering simulator capabilities...");
    let executor = registry.get_executor(&simulator_id)
        .map_err(|e| format!("Failed to get executor: {}", e))?;
    let capabilities = executor.discover_capabilities()
        .map_err(|e| format!("Failed to discover capabilities: {}", e))?;
    
    println!("  Architecture: {:?}", capabilities.architecture);
    println!("  Qubit count: {}", capabilities.qubit_count);
    println!("  Supported gates: {:?}", capabilities.supported_gates.len());
    println!("  Instruction sets: {:?}", capabilities.instruction_sets);
    
    // Get simulator status
    println!("\nChecking simulator status...");
    let status = executor.get_status()
        .map_err(|e| format!("Failed to get status: {}", e))?;
    println!("  Online: {}", status.online);
    println!("  Available qubits: {}", status.available_qubits);
    println!("  Utilization: {:.1}%", status.utilization * 100.0);
    println!("  Queue length: {}", status.queue_length);
    
    // Execute a simple circuit on the simulator
    println!("\nExecuting a simple circuit on the simulator...");
    let circuit = vec![
        "h 0".to_string(),
        "cx 0 1".to_string(),
        "measure 0".to_string(),
        "measure 1".to_string(),
    ];
    let qubits = vec![0, 1];
    
    let result = executor.execute_circuit(&circuit, &qubits)
        .map_err(|e| format!("Circuit execution failed: {}", e))?;
    println!("  Result bytes: {:?}", result);
    
    // Connect to simulated IBM Quantum provider
    println!("\nConnecting to simulated IBM Quantum provider...");
    
    // Create a config for IBM Quantum
    let ibmq_config = HardwareConnectionConfig {
        provider: "ibmq".to_string(),
        endpoint: "https://api.quantum-computing.ibm.com/api".to_string(),
        auth: AuthMethod::ApiKey("SIMULATED_IBM_API_KEY".to_string()),
        timeout_ms: 10000,
        secure: true,
        options: {
            let mut opts = HashMap::new();
            opts.insert("backend".to_string(), "ibmq_qasm_simulator".to_string());
            opts
        },
    };
    
    let ibmq_id = registry.connect_hardware("ibmq", &ibmq_config)
        .map_err(|e| format!("Failed to connect to IBM: {}", e))?;
    println!("  Connected to IBM Quantum with ID: {}", ibmq_id);
    
    // Discover IBM Q capabilities
    println!("Discovering IBM Quantum capabilities...");
    let executor = registry.get_executor(&ibmq_id)
        .map_err(|e| format!("Failed to get IBM executor: {}", e))?;
    let capabilities = executor.discover_capabilities()
        .map_err(|e| format!("Failed to discover IBM capabilities: {}", e))?;
    
    println!("  Architecture: {:?}", capabilities.architecture);
    println!("  Qubit count: {}", capabilities.qubit_count);
    println!("  Supported gates: {:?}", capabilities.supported_gates);
    println!("  Error rates:");
    for (gate_type, rate) in &capabilities.error_rates {
        println!("    {}: {:.6}", gate_type, rate);
    }
    
    // Connect to simulated IonQ provider
    println!("\nConnecting to simulated IonQ provider...");
    
    // Create a config for IonQ
    let ionq_config = HardwareConnectionConfig {
        provider: "ionq".to_string(),
        endpoint: "https://api.ionq.co/v0.2".to_string(),
        auth: AuthMethod::ApiKey("SIMULATED_IONQ_API_KEY".to_string()),
        timeout_ms: 10000,
        secure: true,
        options: HashMap::new(),
    };
    
    let ionq_id = registry.connect_hardware("ionq", &ionq_config)
        .map_err(|e| format!("Failed to connect to IonQ: {}", e))?;
    println!("  Connected to IonQ with ID: {}", ionq_id);
    
    // Discover IonQ capabilities
    println!("Discovering IonQ capabilities...");
    let executor = registry.get_executor(&ionq_id)
        .map_err(|e| format!("Failed to get IonQ executor: {}", e))?;
    let capabilities = executor.discover_capabilities()
        .map_err(|e| format!("Failed to discover IonQ capabilities: {}", e))?;
    
    println!("  Architecture: {:?}", capabilities.architecture);
    println!("  Qubit count: {}", capabilities.qubit_count);
    println!("  Coherence times:");
    for (time_type, value) in &capabilities.coherence_times {
        println!("    {}: {:.1} μs", time_type, value);
    }
    
    // Try calibration (which is supported on IonQ but not on IBM)
    println!("\nAttempting calibration on different providers...");
    
    // Try on IBM (should fail)
    println!("Calibrating IBM Quantum (expected to fail)...");
    let executor = registry.get_executor(&ibmq_id)
        .map_err(|e| format!("Failed to get IBM executor: {}", e))?;
    match executor.calibrate() {
        Ok(result) => {
            println!("  Calibration succeeded (unexpected): {:?}", result.success);
        },
        Err(e) => {
            println!("  Calibration failed as expected: {}", e);
        }
    }
    
    // Try on IonQ (should succeed)
    println!("Calibrating IonQ...");
    let executor = registry.get_executor(&ionq_id)
        .map_err(|e| format!("Failed to get IonQ executor: {}", e))?;
    match executor.calibrate() {
        Ok(result) => {
            println!("  Calibration succeeded: {}", result.success);
            println!("  Updated error rates:");
            for (gate_type, rate) in &result.updated_error_rates {
                println!("    {}: {:.6}", gate_type, rate);
            }
        },
        Err(e) => {
            println!("  Calibration failed: {}", e);
        }
    }
    
    // List all connected hardware
    println!("\nCurrently connected hardware:");
    for (i, id) in registry.get_connected_hardware().iter().enumerate() {
        let executor = registry.get_executor(id)
            .map_err(|e| format!("Failed to get executor: {}", e))?;
        println!("  {}. {} ({})", i + 1, id, executor.provider_name());
    }
    
    // Use the default executor
    println!("\nUsing default executor...");
    let default = registry.get_default_executor()
        .map_err(|e| format!("Failed to get default executor: {}", e))?;
    println!("  Default provider: {}", default.provider_name());
    
    // Change default executor
    println!("Changing default executor to IonQ...");
    registry.set_default_executor(&ionq_id)
        .map_err(|e| format!("Failed to set default executor: {}", e))?;
    
    let default = registry.get_default_executor()
        .map_err(|e| format!("Failed to get new default executor: {}", e))?;
    println!("  New default provider: {}", default.provider_name());
    
    // Disconnect from all hardware
    println!("\nDisconnecting from all hardware...");
    registry.disconnect_all()
        .map_err(|e| format!("Failed to disconnect hardware: {}", e))?;
    println!("  All connections closed");
    
    println!("\nQHEP hardware integration example completed.");
    Ok(())
} 