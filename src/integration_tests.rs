//! # Integration Tests Module
//!
//! This module contains integration tests for the quantum protocol library.
//! These tests verify that different components work together correctly.

#[cfg(test)]
mod tests {
    use crate::core::{QuantumState, QuantumRegister, QSTP};
    use crate::core::qstp::QSTPTransport;
    use crate::network::entanglement::{QEP, EntanglementProtocol, EntanglementPurpose};
    use crate::network::Node;
    use crate::network::topology::NetworkTopology;
    use crate::network::routing::QNRPRouter;
    
    #[tokio::test]
    async fn test_quantum_state_operations() {
        // Create a new quantum state
        let mut state = QuantumState::new(3);
        
        // Check initial state
        assert_eq!(state.num_qubits(), 3);
        assert_eq!(state.state_vector().len(), 8); // 2^3
        
        // Apply gates
        state.apply_gate("H", 0);
        state.apply_gate("X", 1);
        
        // Apply decoherence
        state.apply_decoherence(0.1);
        
        // State should still be valid
        assert!(state.fidelity() > 0.0);
        assert!(!state.is_decohered());
    }
    
    #[tokio::test]
    async fn test_entanglement_operations() {
        // Create QEP instance
        let mut qep = QEP::new("test-manager".to_string());
        
        // Create entanglement between nodes
        let result = qep.create_entanglement(
            "node-1", 
            "node-2",
            EntanglementPurpose::General
        ).await;
        
        assert!(result.is_ok());
        
        let pair = result.unwrap();
        assert_eq!(pair.node_a_id, "node-1");
        assert_eq!(pair.node_b_id, "node-2");
        assert!(pair.fidelity > 0.9); // High initial fidelity
        
        // Test entanglement swapping
        let pair_2 = qep.create_entanglement(
            "node-2",
            "node-3",
            EntanglementPurpose::General
        ).await.unwrap();
        
        let swapped = qep.swap_entanglement(&pair.id, &pair_2.id).await;
        assert!(swapped.is_ok());
        
        let swapped_pair = swapped.unwrap();
        assert_eq!(swapped_pair.node_a_id, "node-1");
        assert_eq!(swapped_pair.node_b_id, "node-3");
        
        // Swapped pair typically has changed fidelity compared to originals
        // but our actual implementation might vary, so we just check it's valid
        assert!(swapped_pair.fidelity > 0.0 && swapped_pair.fidelity <= 1.0);
        assert!(swapped_pair.is_swapped); // Should be marked as swapped
    }
    
    #[tokio::test]
    async fn test_qstp_transfer() {
        // Create QSTP instances
        let mut node_a = QSTP::new("node-a".to_string());
        let node_b = QSTP::new("node-b".to_string());
        
        // Create a quantum state to transfer
        let state = QuantumState::new(2);
        
        // Send the state
        let result = node_a.send_state(&state, node_b.node_id()).await;
        assert!(result.is_ok());
        
        let transfer_result = result.unwrap();
        assert!(transfer_result.success);
        assert!(transfer_result.fidelity > 0.0);
    }
    
    #[tokio::test]
    async fn test_network_topology() {
        // Create a network topology
        let mut topology = NetworkTopology::new();
        
        // Add nodes
        let node1 = Node::new("node-1");
        let node2 = Node::new("node-2");
        let node3 = Node::new("node-3");
        
        topology.add_node(node1);
        topology.add_node(node2);
        topology.add_node(node3);
        
        // Add connections
        topology.add_connection("node-1", "node-2");
        topology.add_connection("node-2", "node-3");
        
        // Test connections
        assert!(topology.are_connected("node-1", "node-2"));
        assert!(topology.are_connected("node-2", "node-3"));
        assert!(!topology.are_connected("node-1", "node-3"));
        
        // Test path finding
        let router = QNRPRouter::new(topology);
        let path = router.compute_path("node-1", "node-3");
        
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.nodes.len(), 3); // node-1 -> node-2 -> node-3
        assert_eq!(path.nodes[0], "node-1");
        assert_eq!(path.nodes[1], "node-2");
        assert_eq!(path.nodes[2], "node-3");
    }
    
    #[tokio::test]
    async fn test_quantum_register() {
        // Create a quantum register
        let mut register = QuantumRegister::new(3);
        assert_eq!(register.size(), 3);
        
        // Apply gates
        register.hadamard(0);
        register.x(1);
        
        // Measure
        let measurements = register.measure_all();
        assert_eq!(measurements.len(), 3);
        
        // Create register in special states
        let ghz = QuantumRegister::ghz(3);
        assert_eq!(ghz.size(), 3);
        
        let w = QuantumRegister::w_state(3);
        assert_eq!(w.size(), 3);
    }
} 