// Integrated QCBP (Quantum-Classical Bridge Protocol) Example
//
// This example demonstrates QCBP integration with other quantum protocols
// using the QuantumProtocolBridge to create a complete quantum networking stack.

use quantum_protocols::integration::qcbp::DataFormat;
use quantum_protocols::security::qkd::QKD;
use quantum_protocols::network::qcap::QCAP;
use quantum_protocols::core::qstp::QSTP;
use quantum_protocols::util;
use serde::{Serialize, Deserialize};
use quantum_protocols::integration::protocol_bridge::QuantumProtocolBridge;

// Define a sample data structure for blockchain transactions
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BlockchainTransaction {
    sender: String,
    receiver: String,
    amount: f64,
    timestamp: u64,
    memo: String,
    nonce: u64,
    signature: String,
}

// Define a sample quantum network node
struct QuantumNetworkNode {
    id: String,
    bridge: QuantumProtocolBridge,
}

impl QuantumNetworkNode {
    fn new(id: &str) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        // Initialize basic protocols
        let qkd = QKD::new(id.to_string());
        let qcap = QCAP::new(id.to_string());
        let qstp = QSTP::new(id.to_string());
        
        // Create protocol bridge
        let bridge = QuantumProtocolBridge::new()
            .with_qkd(qkd)
            .with_qcap(qcap)
            .with_qstp(qstp);
        
        Ok(Self {
            id: id.to_string(),
            bridge,
        })
    }
    
    fn process_transaction(&mut self, transaction: &BlockchainTransaction) -> std::result::Result<(), Box<dyn std::error::Error>> {
        println!("[Node {}] Processing transaction", self.id);
        println!("  From: {}", transaction.sender);
        println!("  To: {}", transaction.receiver);
        println!("  Amount: {}", transaction.amount);
        
        // Serialize transaction
        let transaction_json = serde_json::to_vec(transaction)?;
        
        // Process through the bridge
        let result = self.bridge.process_blockchain_transaction(&transaction_json)?;
        println!("  Result: {}", result);
        
        Ok(())
    }
    
    fn announce_capabilities(&mut self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        println!("[Node {}] Announcing quantum capabilities", self.id);
        
        let capabilities = self.bridge.announce_capabilities()?;
        
        for (i, capability) in capabilities.iter().enumerate() {
            println!("  {}. {}: {:?} (version {})", 
                     i + 1, 
                     capability.name, 
                     capability.level, 
                     capability.version);
            
            for (key, value) in &capability.parameters {
                println!("     - {}: {}", key, value);
            }
        }
        
        Ok(())
    }
    
    fn generate_secure_data(&mut self, size: usize) -> std::result::Result<Vec<u8>, Box<dyn std::error::Error>> {
        println!("[Node {}] Generating quantum-secure data", self.id);
        
        let data = self.bridge.generate_quantum_secure_data(size)?;
        println!("  Generated {} bytes of secure data", data.len());
        
        Ok(data)
    }
    
    fn print_logs(&self) {
        println!("\n[Node {}] Activity Log:", self.id);
        for (i, log) in self.bridge.get_logs().iter().enumerate() {
            println!("  {}. {}", i + 1, log);
        }
    }
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Integrated Quantum-Classical Bridge Protocol (QCBP) Example");
    println!("==========================================================\n");

    // Create a quantum network node
    let mut node_a = QuantumNetworkNode::new("A")?;
    
    // Announce capabilities
    node_a.announce_capabilities()?;
    
    // Create a sample blockchain transaction
    let transaction = BlockchainTransaction {
        sender: "0xAlice_quantum_wallet".to_string(),
        receiver: "0xBob_quantum_wallet".to_string(),
        amount: 42.5,
        timestamp: util::timestamp_now(),
        memo: "Payment for quantum computing services".to_string(),
        nonce: 12345,
        signature: "SIG_quantum_resistant_VALID".to_string(),
    };
    
    println!("\nCreated blockchain transaction:");
    println!("  Sender: {}", transaction.sender);
    println!("  Receiver: {}", transaction.receiver);
    println!("  Amount: {}", transaction.amount);
    println!("  Memo: {}", transaction.memo);
    println!();
    
    // Process the transaction through the quantum network
    node_a.process_transaction(&transaction)?;
    
    // Generate quantum-secure random data
    let secure_data = node_a.generate_secure_data(64)?;
    println!("  First 16 bytes: {:02X?}", &secure_data[0..16.min(secure_data.len())]);
    
    // Demonstrate direct QCBP usage through the bridge
    println!("\nDirect QCBP usage through the protocol bridge:");
    
    // Get QCBP reference from the bridge
    let qcbp = node_a.bridge.qcbp_mut();
    
    // Create test data
    let test_data = r#"{"type":"quantum_event","data":"superposition_measured","probability":0.75}"#.as_bytes();
    println!("  Original data: {}", String::from_utf8_lossy(test_data));
    
    // Convert to quantum format
    let (quantum_register, metadata) = qcbp.classical_to_quantum(test_data, Some(DataFormat::JSON))?;
    println!("  Converted to quantum register with {} qubits", quantum_register.size());
    println!("  Information loss: {:.6}", metadata.info_loss_metric);
    
    // Convert back to classical
    let recovered_data = qcbp.quantum_to_classical(&quantum_register, &metadata, Some(DataFormat::JSON))?;
    println!("  Recovered data: {}", String::from_utf8_lossy(&recovered_data));
    
    // Verify data integrity
    let is_equal = test_data == recovered_data.as_slice();
    println!("  Data integrity: {}", if is_equal { "PRESERVED" } else { "COMPROMISED" });
    
    // Show the node's activity logs
    node_a.print_logs();
    
    println!("\nIntegrated QCBP Example completed successfully!");
    Ok(())
} 