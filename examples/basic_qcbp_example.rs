// Basic QCBP (Quantum-Classical Bridge Protocol) Example
//
// This example demonstrates the core functionality of the Quantum-Classical Bridge Protocol,
// which handles conversions between classical and quantum data representations.

use rand::Rng;
use std::time::Duration;

use quantum_protocols::integration::qcbp::{QCBP, QCBPConfig, DataFormat};
use quantum_protocols::util;
use quantum_protocols::error::Error;
use quantum_protocols::core::quantum_state::QuantumState;
use serde::{Serialize, Deserialize};

// Define a sample data structure to convert
#[derive(Debug, Serialize, Deserialize)]
struct BlockchainTransaction {
    sender: String,
    receiver: String,
    amount: f64,
    timestamp: u64,
    signature: String,
}

// Define our own transaction type for the example
#[derive(Debug, Serialize, Deserialize, Clone)]
struct Transaction {
    id: String,
    transaction_type: TransactionType,
    amount: f64,
    timestamp: u64,
    sender: String,
    receiver: String,
    data: Option<Vec<u8>>,
    signature: Option<Vec<u8>>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
enum TransactionType {
    Payment,
    DataStorage,
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Quantum Classical Blockchain Protocol Example");
    println!("--------------------------------------------");
    
    // Create an instance of QCBP with default config
    let config = QCBPConfig::default();
    let mut qcbp = QCBP::new(config);
    
    // Generate a sample transaction
    let transaction = generate_test_transaction();
    println!("\nGenerated test transaction:");
    println!("  ID: {}", transaction.id);
    println!("  Type: {:?}", transaction.transaction_type);
    println!("  Amount: {}", transaction.amount);
    
    // Convert the transaction using different data formats
    process_with_data_format(&transaction, DataFormat::Binary)?;
    process_with_data_format(&transaction, DataFormat::JSON)?;
    process_with_data_format(&transaction, DataFormat::Custom)?;
    
    // Generate a quantum state for the transaction
    println!("\nGenerating quantum state for transaction...");
    let serialized = bincode::serialize(&transaction).map_err(|e| Box::new(Error::from(e.to_string())))?;
    let (state, metadata) = qcbp.classical_to_quantum(&serialized, Some(DataFormat::Binary))?;
    
    println!("Created quantum state with {} qubits", state.size());
    println!("Quantum state metadata: {metadata:?}");
    
    // Convert back to classical form
    println!("\nConverting back to classical data...");
    let recovered_bytes = qcbp.quantum_to_classical(&state, &metadata, Some(DataFormat::Binary))?;
    
    // Deserialize back to transaction
    let recovered: Transaction = bincode::deserialize(&recovered_bytes)
        .map_err(|e| Box::new(Error::from(e.to_string())))?;
    
    // Verify the transaction
    println!("Recovered transaction ID: {}", recovered.id);
    println!("Original transaction ID: {}", transaction.id);
    println!("Transaction integrity: {}", if recovered.id == transaction.id { "OK" } else { "FAILED" });
    
    Ok(())
}

// Generate a test transaction
fn generate_test_transaction() -> Transaction {
    let mut rng = rand::thread_rng();
    
    Transaction {
        id: format!("TX-{:016x}", rng.gen::<u64>()),
        transaction_type: if rng.gen_bool(0.5) { TransactionType::Payment } else { TransactionType::DataStorage },
        amount: rng.gen_range(1.0..1000.0),
        timestamp: util::timestamp_now(),
        sender: format!("sender-{:04x}", rng.gen::<u16>()),
        receiver: format!("receiver-{:04x}", rng.gen::<u16>()),
        data: if rng.gen_bool(0.3) {
            Some(vec![rng.gen::<u8>(); rng.gen_range(10..100)])
        } else {
            None
        },
        signature: Some(vec![rng.gen::<u8>(); 32]),
    }
}

// Process a transaction with a specific data format
fn process_with_data_format(
    transaction: &Transaction,
    format: DataFormat
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("\nProcessing with {format:?} format:");
    
    // Perform the serialization
    let bytes = match format {
        DataFormat::Binary => bincode::serialize(transaction).map_err(|e| Box::new(Error::from(e.to_string())))?,
        DataFormat::JSON => {
            println!("JSON representation: {}", serde_json::to_string_pretty(transaction)?);
            serde_json::to_vec(transaction)?
        },
        DataFormat::Custom => {
            let mut simulated_custom = bincode::serialize(transaction).map_err(|e| Box::new(Error::from(e.to_string())))?;
            // In a real implementation, this would use a proper custom format
            // For this example, we'll just add a header to simulate different format
            let mut result = vec![0x43, 0x55, 0x53, 0x54]; // "CUST" header
            result.append(&mut simulated_custom);
            result
        },
        DataFormat::ProtoBuf => {
            println!("Format not implemented in this example");
            Vec::new()
        }
    };
    
    println!("Serialized size: {} bytes", bytes.len());
    
    // Create a simulated QuantumState from the bytes
    let quantum_data = QuantumState::new(bytes.len() * 8);
    
    // Wait a moment to simulate quantum operations
    std::thread::sleep(Duration::from_millis(50));
    
    println!("Simulated quantum encoding complete");
    println!("Simulated quantum state fidelity: {:.4}", quantum_data.fidelity());
    
    Ok(())
} 