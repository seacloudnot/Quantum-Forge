// Quantum Memory Protocol (QMEM)
//
// This protocol defines interfaces for storing and managing quantum states
// with features like coherence maintenance, defragmentation, and efficient
// allocation of quantum memory resources.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

use crate::core::QuantumState;
use crate::error::{Result, Error, QuantumStateError};

/// Configuration for the QMEM protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMEMConfig {
    /// Maximum number of qubits that can be stored
    pub max_capacity: usize,

    /// Decoherence check interval in milliseconds
    pub decoherence_check_ms: u64,

    /// Enable automatic defragmentation
    pub auto_defrag: bool,

    /// Defragmentation threshold (percentage of fragmentation to trigger automatic defrag)
    pub defrag_threshold: f64,

    /// Whether to use error correction on stored states
    pub use_error_correction: bool,
}

impl Default for QMEMConfig {
    fn default() -> Self {
        Self {
            max_capacity: 100,
            decoherence_check_ms: 1000, // Check every second
            auto_defrag: true,
            defrag_threshold: 0.3, // 30% fragmentation
            use_error_correction: false,
        }
    }
}

/// Allocation strategy for quantum memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// First fit: allocate at the first available slot
    FirstFit,
    
    /// Best fit: allocate at the smallest available slot that fits
    BestFit,
    
    /// Worst fit: allocate at the largest available slot
    WorstFit,
    
    /// Contiguous: only allocate if contiguous qubits are available
    Contiguous,
}

/// Information about a stored quantum state
#[derive(Debug, Clone)]
struct StoredStateInfo {
    /// The quantum state itself
    state: QuantumState,
    
    /// When the state was last accessed
    last_accessed: Instant,
    
    /// When the state was stored
    stored_at: Instant,
    
    /// Coherence maintenance information (e.g., refresh count)
    refresh_count: usize,
}

/// Quantum Memory Protocol implementation
pub struct QMEM {
    /// Configuration
    config: QMEMConfig,
    
    /// Stored quantum states
    states: HashMap<String, StoredStateInfo>,
    
    /// Memory allocation map (qubit index -> state ID)
    allocation_map: Vec<Option<String>>,
    
    /// Last time decoherence was checked
    last_decoherence_check: Instant,
    
    /// Allocation strategy
    allocation_strategy: AllocationStrategy,
}

impl Default for QMEM {
    fn default() -> Self {
        Self::new()
    }
}

impl QMEM {
    /// Create a new QMEM instance with default configuration
    pub fn new() -> Self {
        Self::with_config(QMEMConfig::default())
    }
    
    /// Create a new QMEM instance with custom configuration
    pub fn with_config(config: QMEMConfig) -> Self {
        let mut allocation_map = Vec::new();
        allocation_map.resize(config.max_capacity, None);
        
        Self {
            config,
            states: HashMap::new(),
            allocation_map,
            last_decoherence_check: Instant::now(),
            allocation_strategy: AllocationStrategy::FirstFit,
        }
    }
    
    /// Set the allocation strategy
    pub fn set_allocation_strategy(&mut self, strategy: AllocationStrategy) {
        self.allocation_strategy = strategy;
    }
    
    /// Get the current allocation strategy
    pub fn allocation_strategy(&self) -> AllocationStrategy {
        self.allocation_strategy
    }
    
    /// Store a quantum state in memory
    pub fn store(&mut self, state: QuantumState) -> Result<String> {
        // Check if we would exceed capacity
        let num_qubits = state.num_qubits();
        if num_qubits > self.available_capacity() {
            return Err(Error::QuantumState(QuantumStateError::ResourceExhausted(
                format!("Not enough quantum memory: need {}, have {}", 
                        num_qubits, self.available_capacity())
            )));
        }
        
        // Find an allocation based on strategy
        let allocation = self.find_allocation(num_qubits)?;
        
        // Store the state
        let state_id = state.id().to_string();
        let info = StoredStateInfo {
            state,
            last_accessed: Instant::now(),
            stored_at: Instant::now(),
            refresh_count: 0,
        };
        
        self.states.insert(state_id.clone(), info);
        
        // Update allocation map
        for i in allocation..(allocation + num_qubits) {
            self.allocation_map[i] = Some(state_id.clone());
        }
        
        // Check if we need decoherence maintenance
        self.check_decoherence();
        
        // Check if we need defragmentation
        if self.config.auto_defrag && self.fragmentation_ratio() > self.config.defrag_threshold {
            self.defragment()?;
        }
        
        Ok(state_id)
    }
    
    /// Retrieve a quantum state from memory
    pub fn retrieve(&mut self, state_id: &str) -> Result<QuantumState> {
        let info = self.states.get_mut(state_id)
            .ok_or_else(|| Error::QuantumState(QuantumStateError::StateNotFound(
                format!("State not found: {state_id}")
            )))?;
        
        // Update access time
        info.last_accessed = Instant::now();
        
        // Return a clone of the state
        Ok(info.state.clone())
    }
    
    /// Remove a quantum state from memory
    pub fn remove(&mut self, state_id: &str) -> Result<QuantumState> {
        // Get the state info
        let info = self.states.remove(state_id)
            .ok_or_else(|| Error::QuantumState(QuantumStateError::StateNotFound(
                format!("State not found: {state_id}")
            )))?;
        
        // Clear allocation map
        for i in 0..self.allocation_map.len() {
            if let Some(id) = &self.allocation_map[i] {
                if id == state_id {
                    self.allocation_map[i] = None;
                }
            }
        }
        
        Ok(info.state)
    }
    
    /// Find the starting index for allocation based on current strategy
    fn find_allocation(&self, num_qubits: usize) -> Result<usize> {
        match self.allocation_strategy {
            AllocationStrategy::FirstFit => self.find_first_fit(num_qubits),
            AllocationStrategy::BestFit => self.find_best_fit(num_qubits),
            AllocationStrategy::WorstFit => self.find_worst_fit(num_qubits),
            AllocationStrategy::Contiguous => self.find_contiguous(num_qubits),
        }
    }
    
    /// Find the first available slot that fits the required qubits
    fn find_first_fit(&self, num_qubits: usize) -> Result<usize> {
        let mut current_run = 0;
        let mut start_idx = 0;
        
        for (i, slot) in self.allocation_map.iter().enumerate() {
            if slot.is_none() {
                if current_run == 0 {
                    // Start of a new run
                    start_idx = i;
                }
                current_run += 1;
                
                if current_run >= num_qubits {
                    return Ok(start_idx);
                }
            } else {
                current_run = 0;
            }
        }
        
        Err(Error::QuantumState(QuantumStateError::ResourceExhausted(
            format!("Could not find a suitable allocation for {num_qubits} qubits")
        )))
    }
    
    /// Find the smallest available slot that fits the required qubits
    fn find_best_fit(&self, num_qubits: usize) -> Result<usize> {
        let mut best_size = usize::MAX;
        let mut best_start = 0;
        let mut found = false;
        
        let mut current_run = 0;
        let mut start_idx = 0;
        
        for (i, slot) in self.allocation_map.iter().enumerate() {
            if slot.is_none() {
                if current_run == 0 {
                    // Start of a new run
                    start_idx = i;
                }
                current_run += 1;
            } else {
                if current_run >= num_qubits && current_run < best_size {
                    best_size = current_run;
                    best_start = start_idx;
                    found = true;
                }
                current_run = 0;
            }
        }
        
        // Check the last run
        if current_run >= num_qubits && current_run < best_size {
            best_start = start_idx;
            found = true;
        }
        
        if found {
            Ok(best_start)
        } else {
            Err(Error::QuantumState(QuantumStateError::ResourceExhausted(
                format!("Could not find a suitable allocation for {num_qubits} qubits")
            )))
        }
    }
    
    /// Find the largest available slot
    fn find_worst_fit(&self, num_qubits: usize) -> Result<usize> {
        let mut worst_size = 0;
        let mut worst_start = 0;
        let mut found = false;
        
        let mut current_run = 0;
        let mut start_idx = 0;
        
        for (i, slot) in self.allocation_map.iter().enumerate() {
            if slot.is_none() {
                if current_run == 0 {
                    // Start of a new run
                    start_idx = i;
                }
                current_run += 1;
            } else {
                if current_run >= num_qubits && current_run > worst_size {
                    worst_size = current_run;
                    worst_start = start_idx;
                    found = true;
                }
                current_run = 0;
            }
        }
        
        // Check the last run
        if current_run >= num_qubits && current_run > worst_size {
            worst_start = start_idx;
            found = true;
        }
        
        if found {
            Ok(worst_start)
        } else {
            Err(Error::QuantumState(QuantumStateError::ResourceExhausted(
                format!("Could not find a suitable allocation for {num_qubits} qubits")
            )))
        }
    }
    
    /// Find a contiguous block of qubits
    fn find_contiguous(&self, num_qubits: usize) -> Result<usize> {
        // For this simple implementation, contiguous is the same as first fit
        self.find_first_fit(num_qubits)
    }
    
    /// Check for decoherence and perform maintenance
    fn check_decoherence(&mut self) {
        let now = Instant::now();
        let check_interval = Duration::from_millis(self.config.decoherence_check_ms);
        
        if now.duration_since(self.last_decoherence_check) < check_interval {
            return;
        }
        
        self.last_decoherence_check = now;
        
        // Check each state for decoherence
        let mut decohered_states = Vec::new();
        
        for (id, info) in &mut self.states {
            if info.state.is_decohered() {
                decohered_states.push(id.clone());
            } else {
                // Apply decoherence effect based on time
                let time_stored = now.duration_since(info.stored_at).as_secs_f64();
                let decoherence_factor = 0.01 * time_stored; // Simple model: 1% per second
                
                if decoherence_factor > 0.0 {
                    info.state.apply_decoherence(decoherence_factor);
                    info.refresh_count += 1;
                }
                
                // Apply error correction if enabled
                if self.config.use_error_correction {
                    // In a real implementation, this would apply error correction
                    // For simulation, we can just slightly improve the state
                    info.state.apply_decoherence(-0.005); // Negative means improve coherence
                }
            }
        }
        
        // Remove decohered states
        for id in decohered_states {
            // Just remove from our state map, but keep allocation to avoid fragmentation
            // until defragmentation happens
            self.states.remove(&id);
            
            // Clear allocation map
            for i in 0..self.allocation_map.len() {
                if let Some(stored_id) = &self.allocation_map[i] {
                    if stored_id == &id {
                        self.allocation_map[i] = None;
                    }
                }
            }
        }
    }
    
    /// Defragment the quantum memory
    pub fn defragment(&mut self) -> Result<usize> {
        let mut moves = 0;
        
        // Create a new allocation map
        let mut new_map = vec![None; self.allocation_map.len()];
        let mut next_free = 0;
        
        // For each state, find its current allocation and move it to a contiguous space
        for (id, info) in &self.states {
            let num_qubits = info.state.num_qubits();
            
            // Find all current allocations for this state
            let mut current_allocations = Vec::new();
            for (i, slot) in self.allocation_map.iter().enumerate() {
                if let Some(stored_id) = slot {
                    if stored_id == id {
                        current_allocations.push(i);
                    }
                }
            }
            
            // Skip if no allocations found (shouldn't happen in normal operation)
            if current_allocations.is_empty() {
                continue;
            }
            
            // If the allocations are not already contiguous, we need to move
            let is_contiguous = current_allocations.len() == num_qubits &&
                current_allocations.first().unwrap() + current_allocations.len() - 1 == 
                *current_allocations.last().unwrap();
                
            if !is_contiguous || current_allocations[0] != next_free {
                // Need to move this state
                moves += 1;
                
                // Allocate in the new map
                for i in 0..num_qubits {
                    new_map[next_free + i] = Some(id.clone());
                }
            } else {
                // Already in the right place, just copy the allocations
                for i in 0..num_qubits {
                    new_map[next_free + i] = Some(id.clone());
                }
            }
            
            // Update next free position
            next_free += num_qubits;
        }
        
        // Replace the old map with the new one
        self.allocation_map = new_map;
        
        Ok(moves)
    }
    
    /// Get the current fragmentation ratio (0.0 = no fragmentation, 1.0 = completely fragmented)
    pub fn fragmentation_ratio(&self) -> f64 {
        let mut free_blocks = 0;
        let mut in_free_block = false;
        
        for slot in &self.allocation_map {
            if slot.is_none() {
                if !in_free_block {
                    free_blocks += 1;
                    in_free_block = true;
                }
            } else {
                in_free_block = false;
            }
        }
        
        // If there are no free blocks or just one, there's no fragmentation
        if free_blocks <= 1 {
            return 0.0;
        }
        
        // Calculate the ratio based on how many free blocks we have compared to ideal (1)
        let used_slots = self.allocation_map.iter().filter(|s| s.is_some()).count();
        let total_slots = self.allocation_map.len();
        let free_slots = total_slots - used_slots;
        
        if free_slots == 0 {
            return 0.0;
        }
        
        // Normalize to 0.0-1.0 range
        ((free_blocks - 1) as f64) / (free_slots as f64)
    }
    
    /// Get the total capacity of the quantum memory
    pub fn total_capacity(&self) -> usize {
        self.allocation_map.len()
    }
    
    /// Get the currently used capacity
    pub fn used_capacity(&self) -> usize {
        self.allocation_map.iter().filter(|s| s.is_some()).count()
    }
    
    /// Get the available capacity
    pub fn available_capacity(&self) -> usize {
        self.total_capacity() - self.used_capacity()
    }
    
    /// Get the number of stored states
    pub fn state_count(&self) -> usize {
        self.states.len()
    }
    
    /// List all stored state IDs
    pub fn list_states(&self) -> Vec<String> {
        self.states.keys().cloned().collect()
    }
    
    /// Check if a state exists in memory
    pub fn has_state(&self, state_id: &str) -> bool {
        self.states.contains_key(state_id)
    }
    
    /// Get information about memory usage
    pub fn memory_info(&self) -> QMEMInfo {
        QMEMInfo {
            total_capacity: self.total_capacity(),
            used_capacity: self.used_capacity(),
            available_capacity: self.available_capacity(),
            stored_states: self.state_count(),
            fragmentation_ratio: self.fragmentation_ratio(),
        }
    }
}

/// Information about quantum memory usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMEMInfo {
    /// Total capacity in qubits
    pub total_capacity: usize,
    
    /// Used capacity in qubits
    pub used_capacity: usize,
    
    /// Available capacity in qubits
    pub available_capacity: usize,
    
    /// Number of stored quantum states
    pub stored_states: usize,
    
    /// Fragmentation ratio (0.0-1.0)
    pub fragmentation_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_storage() {
        let mut qmem = QMEM::new();
        
        // Create a test state
        let state = QuantumState::new(3);
        let id = qmem.store(state.clone()).unwrap();
        
        // Retrieve the state
        let retrieved = qmem.retrieve(&id).unwrap();
        
        // Check that it's the same state
        assert_eq!(retrieved.num_qubits(), state.num_qubits());
        assert_eq!(retrieved.id(), state.id());
        
        // Remove the state
        let removed = qmem.remove(&id).unwrap();
        assert_eq!(removed.id(), state.id());
        
        // Try to retrieve again (should fail)
        assert!(qmem.retrieve(&id).is_err());
    }
    
    #[test]
    fn test_allocation_strategies() {
        let mut qmem = QMEM::with_config(QMEMConfig {
            max_capacity: 10,
            ..Default::default()
        });
        
        // Store a 2-qubit state
        let state1 = QuantumState::new(2);
        let id1 = qmem.store(state1).unwrap();
        
        // Store a 3-qubit state
        let state2 = QuantumState::new(3);
        let _id2 = qmem.store(state2).unwrap();
        
        // Remove the first state to create fragmentation
        qmem.remove(&id1).unwrap();
        
        // Test different allocation strategies
        qmem.set_allocation_strategy(AllocationStrategy::FirstFit);
        let state3 = QuantumState::new(2);
        let _id3 = qmem.store(state3).unwrap();
        
        // Check memory info
        let info = qmem.memory_info();
        assert_eq!(info.total_capacity, 10);
        assert!(info.used_capacity >= 5); // At least 5 qubits used
        
        // Test defragmentation
        let _moves = qmem.defragment().unwrap();
        
        // After defragmentation, fragmentation should be reduced or remain the same
        // if it was already optimal
        let post_defrag_ratio = qmem.fragmentation_ratio();
        assert!(post_defrag_ratio <= info.fragmentation_ratio + 0.001, 
                "Post-defrag ratio ({}) should be less than or equal to original ratio ({})",
                post_defrag_ratio, info.fragmentation_ratio);
    }
} 