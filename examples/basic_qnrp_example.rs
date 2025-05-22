use quantum_protocols::prelude::*;
use quantum_protocols::network::routing::QuantumRouter;

#[tokio::main]
async fn main() {
    println!("Quantum Network Routing Protocol (QNRP) Example");
    println!("-----------------------------------------------");
    
    // Create a simple network topology
    let mut topology = NetworkTopology::new();
    
    // Add nodes to the topology
    topology.add_node(Node::new("A"));
    topology.add_node(Node::new("B"));
    topology.add_node(Node::new("C"));
    topology.add_node(Node::new("D"));
    topology.add_node(Node::new("E"));
    
    // Add links between nodes with distance and quality
    topology.add_link("A", "B", 1.0, 0.95); // High quality, short
    topology.add_link("B", "C", 1.0, 0.9);
    topology.add_link("C", "D", 1.0, 0.85);
    topology.add_link("D", "E", 1.0, 0.8);
    topology.add_link("A", "E", 4.0, 0.7); // Lower quality, long
    
    // Create a router with the topology
    let mut router = QNRPRouter::new(topology);
    
    // Print the network
    println!("Network Topology:");
    println!("A -- B -- C -- D -- E");
    println!("\\               /");
    println!(" \\-------------/");
    println!();
    
    // Find path from A to E
    println!("Finding path from A to E...");
    let path_a_to_e = router.find_path("A", "E").await.unwrap();
    
    println!("Path found: {:?}", path_a_to_e.nodes);
    println!("Distance: {:.2}", path_a_to_e.distance);
    println!("Minimum fidelity: {:.2}", path_a_to_e.min_fidelity);
    println!("Hop count: {}", path_a_to_e.hop_count());
    println!();
    
    // Find path from E to A
    println!("Finding path from E to A...");
    let path_e_to_a = router.find_path("E", "A").await.unwrap();
    println!("Path found: {:?}", path_e_to_a.nodes);
    println!();
    
    // Find all paths between A and E
    println!("Finding all paths between A and E...");
    let all_paths = router.get_all_paths("A", "E").await.unwrap();
    
    for (i, path) in all_paths.iter().enumerate() {
        println!("Path {}: {:?} (fidelity: {:.2}, distance: {:.2})",
            i + 1, path.nodes, path.min_fidelity, path.distance);
    }
    println!();
    
    // Estimate fidelity between nodes
    println!("Estimating fidelity from A to E...");
    let est_fidelity = router.estimate_path_fidelity("A", "E").await.unwrap();
    println!("Estimated fidelity: {est_fidelity:.2}");
    println!();
    
    // Next hop determination
    println!("Computing next hop from C to E...");
    let next_hop = router.next_hop("C", "E").await.unwrap();
    println!("Next hop: {next_hop}");
    println!();
    
    // Update the topology (simulate changing conditions)
    println!("Simulating link degradation from B to C...");
    
    // Update the link quality between B and C
    router.topology_mut().set_link_quality("B", "C", 0.5); // Degraded quality
    
    // Update routes
    println!("Updating routes after link degradation...");
    router.update_routes().await.unwrap();
    
    // Find new path
    println!("Finding new path from A to E after link degradation...");
    let new_path = router.find_path("A", "E").await.unwrap();
    println!("New path found: {:?}", new_path.nodes);
    println!("New minimum fidelity: {:.2}", new_path.min_fidelity);
    println!("New distance: {:.2}", new_path.distance);
    
    println!("QNRP example completed!");
} 