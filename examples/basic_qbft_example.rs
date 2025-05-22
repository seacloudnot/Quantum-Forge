use quantum_protocols::prelude::*;

#[tokio::main]
async fn main() {
    println!("Quantum Byzantine Fault Tolerance (QBFT) Example");
    println!("-----------------------------------------------");
    
    // Create a network of nodes
    let node_ids = vec![
        "node1".to_string(),
        "node2".to_string(),
        "node3".to_string(),
        "node4".to_string(),
    ];
    
    println!("Network with {} nodes: {:?}", node_ids.len(), node_ids);
    println!();
    
    // Create a QBFT instance for the primary node (node1)
    println!("Creating QBFT instance for primary node (node1)...");
    let mut qbft = QBFT::new(node_ids[0].clone(), node_ids.clone());
    
    // Disable quantum verification for this basic example
    println!("Configuring QBFT (disabling quantum verification for example)...");
    let config = QBFTConfig {
        use_quantum_verification: false,
        ..QBFTConfig::default()
    };
    qbft.set_config(config.clone());
    
    // Create a test value for consensus
    let test_value = b"example transaction data: transfer 10 tokens from Alice to Bob";
    println!("Test value for consensus: {:?}", String::from_utf8_lossy(test_value));
    println!();
    
    // Explain the consensus phases
    println!("QBFT Consensus Phases:");
    println!("1. Pre-prepare: Primary proposes the value");
    println!("2. Prepare: Nodes verify and vote on the proposal");
    println!("3. Commit: Nodes commit to the agreed value");
    println!("4. Quantum verification: Optional quantum verification step");
    println!();
    
    // Propose the value and run the consensus protocol
    println!("Starting consensus protocol...");
    println!("Primary node is proposing the test value...");
    let result = qbft.propose(test_value).await;
    
    // Print consensus result
    println!();
    println!("Consensus complete!");
    println!("Result:");
    println!("  Consensus reached: {}", result.consensus_reached);
    println!("  Participants: {}", result.participants);
    println!("  Agreements: {}", result.agreements);
    
    if let Some(value) = result.value {
        println!("  Agreed value: {:?}", String::from_utf8_lossy(&value));
    } else {
        println!("  No agreed value");
    }
    println!();
    
    // Create a QBFT instance for a non-primary node (node2)
    println!("Creating QBFT instance for non-primary node (node2)...");
    let mut node2_qbft = QBFT::new(node_ids[1].clone(), node_ids.clone());
    node2_qbft.set_config(config);
    
    // Try to propose a value from a non-primary node (should fail)
    println!("Non-primary node (node2) is trying to propose a value...");
    let test_value2 = b"this proposal should fail";
    let result2 = node2_qbft.propose(test_value2).await;
    
    // Print result (should fail)
    println!("Result of non-primary proposal:");
    println!("  Consensus reached: {}", result2.consensus_reached);
    println!();
    
    // Demonstrate a view change
    println!("Simulating view change to make node2 the primary...");
    // Advance the view number to make node2 the primary
    node2_qbft.set_view(1);
    
    // Verify the primary has changed
    println!("After view change, node2 is now primary: {}", node2_qbft.is_primary());
    
    // Now node2 can propose successfully
    println!("New primary (node2) is proposing a value after view change...");
    let test_value3 = b"proposal after view change";
    let result3 = node2_qbft.propose(test_value3).await;
    
    // Print result
    println!("Result of proposal after view change:");
    println!("  Consensus reached: {}", result3.consensus_reached);
    if let Some(value) = result3.value {
        println!("  Agreed value: {:?}", String::from_utf8_lossy(&value));
    }
    
    println!();
    println!("QBFT example completed!");
} 