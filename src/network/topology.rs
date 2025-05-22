// Network Topology Implementation
//
// This file implements a topology representation for a quantum network.

use crate::network::node::Node;
use crate::network::entanglement::EntanglementPair;
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Represents the topology of a quantum network
#[derive(Clone)]
pub struct NetworkTopology {
    /// Nodes in the network by ID
    nodes: HashMap<String, Node>,
    
    /// Direct connections between nodes (adjacency list)
    connections: HashMap<String, HashSet<String>>,
    
    /// Entanglement pairs in the network
    entanglements: HashMap<String, EntanglementPair>,
}

impl NetworkTopology {
    /// Create a new empty network topology
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            connections: HashMap::new(),
            entanglements: HashMap::new(),
        }
    }
    
    /// Add a node to the network
    pub fn add_node(&mut self, node: Node) {
        let id = node.id().to_string();
        self.nodes.insert(id.clone(), node);
        self.connections.entry(id).or_default();
    }
    
    /// Add a node to the network by ID
    pub fn add_node_by_id(&mut self, node_id: String) {
        if !self.nodes.contains_key(&node_id) {
            let node = Node::new(&node_id);
            self.nodes.insert(node_id.clone(), node);
            self.connections.entry(node_id).or_default();
        }
    }
    
    /// Add a direct connection between nodes
    pub fn add_connection(&mut self, node_a: &str, node_b: &str) {
        if self.nodes.contains_key(node_a) && self.nodes.contains_key(node_b) {
            self.connections.entry(node_a.to_string())
                .or_default()
                .insert(node_b.to_string());
                
            self.connections.entry(node_b.to_string())
                .or_default()
                .insert(node_a.to_string());
        }
    }
    
    /// Add an entanglement pair to the network
    pub fn add_entanglement(&mut self, pair: EntanglementPair) {
        self.entanglements.insert(pair.id.clone(), pair);
    }
    
    /// Get a reference to a node by ID
    pub fn get_node(&self, node_id: &str) -> Option<&Node> {
        self.nodes.get(node_id)
    }
    
    /// Get a mutable reference to a node by ID
    pub fn get_node_mut(&mut self, node_id: &str) -> Option<&mut Node> {
        self.nodes.get_mut(node_id)
    }
    
    /// Get all nodes in the network
    pub fn nodes(&self) -> &HashMap<String, Node> {
        &self.nodes
    }
    
    /// Get connections for a node
    pub fn connections_for(&self, node_id: &str) -> Vec<&str> {
        self.connections.get(node_id)
            .map(|connections| connections.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }
    
    /// Check if two nodes are directly connected
    pub fn are_connected(&self, node_a: &str, node_b: &str) -> bool {
        self.connections.get(node_a)
            .map(|connections| connections.contains(node_b))
            .unwrap_or(false)
    }
    
    /// Get all entanglement pairs in the network
    pub fn entanglements(&self) -> &HashMap<String, EntanglementPair> {
        &self.entanglements
    }
    
    /// Get entanglement pairs between two nodes
    pub fn entanglements_between(&self, node_a: &str, node_b: &str) -> Vec<&EntanglementPair> {
        self.entanglements.values()
            .filter(|pair| {
                (pair.node_a_id == node_a && pair.node_b_id == node_b) ||
                (pair.node_a_id == node_b && pair.node_b_id == node_a)
            })
            .collect()
    }
    
    /// Find the shortest path between two nodes
    pub fn find_path(&self, start_node: &str, end_node: &str) -> Option<Vec<String>> {
        // Simple breadth-first search
        if !self.nodes.contains_key(start_node) || !self.nodes.contains_key(end_node) {
            return None;
        }
        
        if start_node == end_node {
            return Some(vec![start_node.to_string()]);
        }
        
        let mut queue = std::collections::VecDeque::new();
        let mut visited = HashSet::new();
        let mut prev: HashMap<String, String> = HashMap::new();
        
        queue.push_back(start_node.to_string());
        visited.insert(start_node.to_string());
        
        while let Some(current) = queue.pop_front() {
            if current == end_node {
                // Build path
                let mut path = Vec::new();
                let mut current_node = end_node.to_string();
                
                while current_node != start_node {
                    path.push(current_node.clone());
                    current_node = prev.get(&current_node).unwrap().clone();
                }
                
                path.push(start_node.to_string());
                path.reverse();
                
                return Some(path);
            }
            
            if let Some(connections) = self.connections.get(&current) {
                for neighbor in connections {
                    if !visited.contains(neighbor) {
                        queue.push_back(neighbor.clone());
                        visited.insert(neighbor.clone());
                        prev.insert(neighbor.clone(), current.clone());
                    }
                }
            }
        }
        
        None
    }
    
    /// Get all node IDs in the network
    pub fn get_all_nodes(&self) -> Vec<String> {
        self.nodes.keys().cloned().collect()
    }
    
    /// Get neighbors of a node
    pub fn get_neighbors(&self, node_id: &str) -> Vec<String> {
        self.connections
            .get(node_id)
            .map_or(Vec::new(), |neighbors| neighbors.iter().cloned().collect())
    }
    
    /// Get the distance between two nodes
    pub fn get_distance(&self, node_a: &str, node_b: &str) -> f64 {
        if let (Some(node_a), Some(node_b)) = (self.nodes.get(node_a), self.nodes.get(node_b)) {
            node_a.distance_to(node_b).unwrap_or(f64::MAX)
        } else {
            f64::MAX
        }
    }
    
    /// Get the link quality between two nodes
    pub fn get_link_quality(&self, node_a: &str, node_b: &str) -> f64 {
        if let (Some(node_a), Some(node_b)) = (self.nodes.get(node_a), self.nodes.get(node_b)) {
            node_a.expected_entanglement_fidelity(node_b)
        } else {
            0.0
        }
    }
    
    /// Check if two nodes have an entanglement
    pub fn has_entanglement(&self, node_a: &str, node_b: &str) -> bool {
        self.entanglements.values().any(|pair| 
            (pair.node_a_id == node_a && pair.node_b_id == node_b) ||
            (pair.node_a_id == node_b && pair.node_b_id == node_a)
        )
    }
    
    /// Remove a connection between nodes
    pub fn remove_link(&mut self, node_a: &str, node_b: &str) {
        if let Some(connections_a) = self.connections.get_mut(node_a) {
            connections_a.remove(node_b);
        }
        
        if let Some(connections_b) = self.connections.get_mut(node_b) {
            connections_b.remove(node_a);
        }
    }
    
    /// Add link with quality information
    pub fn add_link(&mut self, node_a: &str, node_b: &str, distance: f64, quality: f64) {
        self.add_connection(node_a, node_b);
        
        // First get coordinates from node_a
        let coords_a = if let Some(node_a_obj) = self.nodes.get(node_a) {
            node_a_obj.coordinates()
        } else {
            None
        };
        
        // Set coordinates for node_b if needed
        if let Some(coords_a) = coords_a {
            if let Some(node_b_obj) = self.nodes.get_mut(node_b) {
                // If we don't have coordinates, set them based on the distance
                if node_b_obj.coordinates().is_none() {
                    // Simple 1D coordinates for illustration
                    node_b_obj.set_coordinates(coords_a.0 + distance, coords_a.1, coords_a.2);
                }
                
                // Set the link quality metadata
                node_b_obj.set_expected_entanglement_fidelity(node_a, quality);
            }
        } else {
            // Set default coordinates for node_a first
            if let Some(node_a_obj) = self.nodes.get_mut(node_a) {
                node_a_obj.set_coordinates(0.0, 0.0, 0.0);
            }
            
            // Then set coordinates for node_b in a separate step
            if let Some(node_b_obj) = self.nodes.get_mut(node_b) {
                if node_b_obj.coordinates().is_none() {
                    node_b_obj.set_coordinates(distance, 0.0, 0.0);
                }
                
                // Set the link quality metadata
                node_b_obj.set_expected_entanglement_fidelity(node_a, quality);
            }
        }
        
        // Finally set quality for node_a to node_b direction
        if let Some(node_a_obj) = self.nodes.get_mut(node_a) {
            node_a_obj.set_expected_entanglement_fidelity(node_b, quality);
        }
    }
    
    /// Set the quality of the link between two nodes
    pub fn set_link_quality(&mut self, node_a: &str, node_b: &str, quality: f64) {
        if let Some(node_a_obj) = self.nodes.get_mut(node_a) {
            node_a_obj.set_expected_entanglement_fidelity(node_b, quality);
            
            // Update the reverse direction too for bidirectional consistency
            if let Some(node_b_obj) = self.nodes.get_mut(node_b) {
                node_b_obj.set_expected_entanglement_fidelity(node_a, quality);
            }
        }
    }
}

impl fmt::Display for NetworkTopology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Network Topology with {} nodes, {} connections, {} entanglements",
            self.nodes.len(),
            self.connections.values().map(|v| v.len()).sum::<usize>() / 2,
            self.entanglements.len()
        )
    }
}

impl Default for NetworkTopology {
    fn default() -> Self {
        Self::new()
    }
} 