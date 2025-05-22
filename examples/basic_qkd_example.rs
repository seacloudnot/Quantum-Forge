use quantum_protocols::prelude::*;
use quantum_protocols::QuantumKeyDistribution;

fn main() {
    println!("Quantum Key Distribution (QKD) Example");
    println!("-------------------------------------");
    
    // Create QKD instances for Alice and Bob
    println!("Creating QKD instances for Alice and Bob...");
    let mut alice_qkd = QKD::new("alice".to_string());
    let mut bob_qkd = QKD::new("bob".to_string());
    
    // Set up authentication tokens
    println!("Setting up authentication...");
    let alice_token = "alice-token";
    let bob_token = "bob-token";
    
    alice_qkd.store_auth_token("bob".to_string(), bob_token.to_string());
    bob_qkd.store_auth_token("alice".to_string(), alice_token.to_string());
    
    // Explain the QKD protocol
    println!();
    println!("BB84 Protocol Steps:");
    println!("1. Alice prepares qubits in random bases with random bits");
    println!("2. Alice sends qubits to Bob");
    println!("3. Bob measures each qubit in a randomly chosen basis");
    println!("4. Alice and Bob publicly compare the bases they used (not the bits)");
    println!("5. They keep only the bits where they happened to use the same basis");
    println!("6. They use these bits as their shared secret key");
    println!();
    
    // Alice initiates a session with Bob
    println!("Alice initiates a QKD session with Bob...");
    let session_id = alice_qkd.init_session("bob").unwrap();
    println!("Session created: {session_id}");
    
    // In a real system, Alice would tell Bob the session ID through a classical channel
    // Here we simulate Bob also creating a session
    println!("Bob also initiates a session with Alice...");
    let bob_session_id = bob_qkd.init_session("alice").unwrap();
    println!("Bob's session: {bob_session_id}");
    println!();
    
    // Set up a small number of qubits for testing
    let qubit_count = 20;
    println!("Using {qubit_count} qubits for key exchange...");
    
    // Alice prepares and sends qubits
    println!("\nStep 1-2: Alice prepares and sends qubits...");
    let _alice_measurements = alice_qkd.exchange_qubits(&session_id, qubit_count).unwrap();
    
    // Print some of Alice's information
    let alice_session = alice_qkd.get_session(&session_id).unwrap();
    println!("Alice's bits to encode (first 10): {:?}", &alice_session.random_bits[0..10.min(alice_session.random_bits.len())]);
    println!("Alice's sending bases (first 10):");
    for i in 0..10.min(alice_session.sending_basis.len()) {
        let basis = match alice_session.sending_basis.get(i) {
            Some(b) => match b {
                QKDBasis::Standard => "+",
                QKDBasis::Diagonal => "x",
            },
            None => continue,
        };
        print!("{basis} ");
    }
    println!();
    
    // Bob also prepares qubits to send to Alice
    println!("\nStep 3: Bob measures the received qubits...");
    let _bob_measurements = bob_qkd.exchange_qubits(&bob_session_id, qubit_count).unwrap();
    
    // Print some of Bob's information
    let bob_session = bob_qkd.get_session(&bob_session_id).unwrap();
    println!("Bob's measuring bases (first 10):");
    for i in 0..10.min(bob_session.measuring_basis.len()) {
        let basis = match bob_session.measuring_basis.get(i) {
            Some(b) => match b {
                QKDBasis::Standard => "+",
                QKDBasis::Diagonal => "x",
            },
            None => continue,
        };
        print!("{basis} ");
    }
    println!();
    println!("Bob's measurements (first 10): {:?}", &bob_session.measurement_results[0..10.min(bob_session.measurement_results.len())]);
    println!();
    
    // Exchange basis information
    println!("Step 4: Alice and Bob exchange basis information...");
    // Get Alice's basis choices as bytes for transmission
    let alice_bases: Vec<u8> = alice_session.measuring_basis.iter()
        .map(QKDBasis::to_byte)
        .collect();
    
    // Get Bob's basis choices as bytes for transmission
    let bob_bases: Vec<u8> = bob_session.measuring_basis.iter()
        .map(QKDBasis::to_byte)
        .collect();
    
    // Compare bases to find matching positions
    println!("Step 5: Comparing measurement bases...");
    let alice_matching = alice_qkd.compare_bases(&session_id, &bob_bases).unwrap();
    let _bob_matching = bob_qkd.compare_bases(&bob_session_id, &alice_bases).unwrap();
    
    println!("Matching bases indices: {alice_matching:?}");
    println!("Matching count: {} out of {} qubits", alice_matching.len(), qubit_count);
    println!();
    
    // Derive keys
    println!("Step 6: Deriving shared secret key...");
    let key_size = alice_matching.len().min(16); // Use at most 16 bits (2 bytes) for example
    
    println!("Using {key_size} bits for the key...");
    let alice_key = alice_qkd.derive_key(&session_id, key_size).unwrap();
    let bob_key = bob_qkd.derive_key(&bob_session_id, key_size).unwrap();
    
    println!("Alice's key: {alice_key:?}");
    println!("Bob's key:   {bob_key:?}");
    
    // Check if keys match
    if alice_key == bob_key {
        println!("Success! Alice and Bob have the same key.");
    } else {
        println!("Error: Keys don't match!");
    }
    
    // Check for eavesdropping
    println!();
    println!("Checking for eavesdroppers...");
    let updated_alice_session = alice_qkd.get_session(&session_id).unwrap();
    let qber = updated_alice_session.estimate_error_rate();
    println!("Quantum bit error rate (QBER): {:.2}%", qber * 100.0);
    
    if qber > 0.1 {
        println!("Warning: High error rate might indicate an eavesdropper!");
    } else {
        println!("No eavesdropper detected!");
    }
    
    println!();
    println!("QKD example completed! Alice and Bob now share a secure key.");
    
    // Demonstrate eavesdropper detection
    println!("\n--- BONUS: Eavesdropper Simulation ---");
    println!("Creating QKD instances with simulated Eve (eavesdropper)...");
    
    // Create instances with high noise to simulate Eve
    let mut alice_qkd_with_eve = QKD::with_config(
        "alice".to_string(),
        QKDConfig {
            noise_level: 0.2, // High noise to simulate Eve
            qber_threshold: 0.1,
            ..QKDConfig::default()
        }
    );
    
    let mut bob_qkd_with_eve = QKD::with_config(
        "bob".to_string(),
        QKDConfig {
            noise_level: 0.2, // High noise to simulate Eve
            qber_threshold: 0.1,
            ..QKDConfig::default()
        }
    );
    
    // Setup authentication
    alice_qkd_with_eve.store_auth_token("bob".to_string(), "token".to_string());
    bob_qkd_with_eve.store_auth_token("alice".to_string(), "token".to_string());
    
    // Create sessions
    let eve_session_id = alice_qkd_with_eve.init_session("bob").unwrap();
    let bob_eve_session_id = bob_qkd_with_eve.init_session("alice").unwrap();
    
    // Exchange qubits
    println!("Exchanging qubits with Eve present...");
    let _ = alice_qkd_with_eve.exchange_qubits(&eve_session_id, qubit_count).unwrap();
    let _ = bob_qkd_with_eve.exchange_qubits(&bob_eve_session_id, qubit_count).unwrap();
    
    // Exchange basis
    let alice_eve_session = alice_qkd_with_eve.get_session(&eve_session_id).unwrap();
    let alice_eve_bases: Vec<u8> = alice_eve_session.measuring_basis.iter()
        .map(quantum_protocols::prelude::QKDBasis::to_byte)
        .collect();
    
    let bob_eve_session = bob_qkd_with_eve.get_session(&bob_eve_session_id).unwrap();
    let bob_eve_bases: Vec<u8> = bob_eve_session.measuring_basis.iter()
        .map(quantum_protocols::prelude::QKDBasis::to_byte)
        .collect();
    
    // Compare bases
    let _ = alice_qkd_with_eve.compare_bases(&eve_session_id, &bob_eve_bases).unwrap();
    let _ = bob_qkd_with_eve.compare_bases(&bob_eve_session_id, &alice_eve_bases).unwrap();
    
    // Check QBER
    let qber_with_eve = alice_qkd_with_eve.get_session(&eve_session_id).unwrap().estimate_error_rate();
    println!("QBER with Eve present: {:.2}%", qber_with_eve * 100.0);
    
    // Try to derive key
    println!("Attempting to derive key with Eve present...");
    match alice_qkd_with_eve.derive_key(&eve_session_id, 8) {
        Ok(_) => println!("Key derived successfully (Eve was not detected)"),
        Err(e) => println!("Error deriving key: {e} (Eve was detected!)"),
    }
    
    println!("\nQKD example with eavesdropper detection completed!");
} 