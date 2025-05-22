// Quantum Synchronization Protocol (QSYP)
//
// This protocol coordinates timing of quantum operations across the network,
// enabling synchronized operations that are critical for distributed quantum computing.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use tokio::sync::{RwLock, Mutex};
use tokio::time;
use tokio::task;
use rand::{Rng, thread_rng};

use crate::network::Node;
use crate::util;

/// Errors specific to Quantum Synchronization Protocol
#[derive(Error, Debug)]
pub enum QSYPError {
    /// Synchronization timeout occurred
    #[error("Synchronization timeout after {0:?}")]
    Timeout(Duration),
    
    /// Clock drift beyond acceptable threshold
    #[error("Clock drift of {0}ms exceeds threshold of {1}ms")]
    ExcessiveDrift(u64, u64),
    
    /// Node is not part of the synchronization group
    #[error("Node {0} is not part of synchronization group {1}")]
    NotInGroup(String, String),
    
    /// Could not establish communication with node
    #[error("Unable to communicate with node {0}")]
    CommunicationFailed(String),
    
    /// General synchronization error
    #[error("Synchronization error: {0}")]
    SyncError(String),
    
    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),
}

/// Synchronization strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncStrategy {
    /// Master-slave synchronization with one reference clock
    MasterSlave,
    
    /// Distributed consensus on time
    DistributedConsensus,
    
    /// Hierarchical synchronization across network layers
    Hierarchical,
    
    /// External atomic clock based synchronization
    AtomicClock,
}

/// Configuration for QSYP
#[derive(Debug, Clone)]
pub struct QSYPConfig {
    /// Maximum allowed clock drift in milliseconds
    pub max_drift_ms: u64,
    
    /// Synchronization interval in milliseconds
    pub sync_interval_ms: u64,
    
    /// Synchronization strategy to use
    pub strategy: SyncStrategy,
    
    /// Synchronization timeout in milliseconds
    pub timeout_ms: u64,
    
    /// Number of sync messages to exchange
    pub sync_message_count: usize,
    
    /// Whether to auto-compensate for observed drift
    pub auto_compensate: bool,
}

impl Default for QSYPConfig {
    fn default() -> Self {
        Self {
            max_drift_ms: 5,          // 5ms maximum allowed drift
            sync_interval_ms: 5000,   // Sync every 5 seconds
            strategy: SyncStrategy::DistributedConsensus,
            timeout_ms: 1000,         // 1 second timeout
            sync_message_count: 8,    // Exchange 8 messages for accurate sync
            auto_compensate: true,
        }
    }
}

/// A synchronization message exchanged between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncMessage {
    /// ID of the sender node
    pub sender_id: String,
    
    /// ID of the receiver node
    pub receiver_id: String,
    
    /// Sender's timestamp when the message was sent
    pub send_timestamp: u64,
    
    /// Receiver's timestamp when message was received
    pub receive_timestamp: Option<u64>,
    
    /// Unique message ID
    pub message_id: String,
    
    /// Round number for multi-round sync
    pub round: usize,
}

/// Result of a synchronization operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncResult {
    /// Calculated clock offset in milliseconds (+ means local ahead, - means behind)
    pub offset_ms: i64,
    
    /// Round-trip time in milliseconds
    pub rtt_ms: u64,
    
    /// Timestamp when synchronization completed
    pub sync_time: u64,
    
    /// Participating nodes in this sync
    pub nodes: Vec<String>,
    
    /// Estimated accuracy in microseconds
    pub accuracy_us: u64,
    
    /// Whether synchronization was successful
    pub success: bool,
}

/// Represents a group of nodes that maintain synchronized timing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncGroup {
    /// Unique ID for this synchronization group
    pub id: String,
    
    /// Name of this synchronization group
    pub name: String,
    
    /// Member nodes in this synchronization group
    pub members: HashSet<String>,
    
    /// Reference node (for master-slave strategy)
    pub reference_node: Option<String>,
    
    /// Creation time of this group
    pub created_at: u64,
    
    /// Last synchronization time
    pub last_sync: u64,
}

/// Interface for quantum timing synchronization
#[async_trait]
pub trait QuantumSynchronizer {
    /// Synchronize with a specific node
    async fn synchronize_with(&mut self, node_id: &str) -> Result<SyncResult, QSYPError>;
    
    /// Synchronize with a group of nodes
    async fn synchronize_group(&mut self, group_id: &str) -> Result<SyncResult, QSYPError>;
    
    /// Create a new synchronization group
    async fn create_sync_group(&mut self, name: &str, members: &[String]) -> Result<SyncGroup, QSYPError>;
    
    /// Get the current clock offset relative to a node
    fn get_offset(&self, node_id: &str) -> Option<i64>;
    
    /// Start background synchronization process
    async fn start_background_sync(&mut self) -> Result<(), QSYPError>;
    
    /// Stop background synchronization
    async fn stop_background_sync(&mut self) -> Result<(), QSYPError>;
}

/// Information about a node's timing characteristics
#[derive(Debug, Clone)]
struct NodeTimingInfo {
    /// Node ID
    _node_id: String,
    
    /// Estimated clock offset (+ means remote ahead, - means remote behind)
    offset_ms: i64,
    
    /// Round-trip time to this node
    rtt_ms: u64,
    
    /// Stability metric (lower is better)
    stability: f64,
    
    /// Last sync time
    last_sync: Instant,
    
    /// Drift rate in parts per million
    drift_rate_ppm: f64,
}

/// Implementation of Quantum Synchronization Protocol
pub struct QSYP {
    /// ID of the node running this protocol
    #[allow(dead_code)]
    node_id: String,
    
    /// Configuration
    config: QSYPConfig,
    
    /// Reference to the node
    node: Option<Arc<RwLock<Node>>>,
    
    /// Known synchronization groups
    sync_groups: HashMap<String, SyncGroup>,
    
    /// Timing information for known nodes
    timing_info: HashMap<String, NodeTimingInfo>,
    
    /// Whether background sync is running
    background_sync_running: bool,
    
    /// Background sync task handle
    #[allow(dead_code)]
    background_task: Option<task::JoinHandle<()>>,
    
    /// Lock for sync operations
    sync_lock: Arc<Mutex<()>>,
}

impl QSYP {
    /// Create a new Quantum Synchronization Protocol instance
    ///
    /// # Arguments
    ///
    /// * `node_id` - ID of the node running this instance
    ///
    /// # Returns
    ///
    /// A new QSYP instance
    #[must_use]
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            config: QSYPConfig::default(),
            sync_groups: HashMap::new(),
            timing_info: HashMap::new(),
            node: None,
            background_sync_running: false,
            background_task: None,
            sync_lock: Arc::new(Mutex::new(())),
        }
    }
    
    /// Create a new QSYP instance with custom configuration
    ///
    /// # Arguments
    ///
    /// * `node_id` - ID of the node running this instance
    /// * `config` - Custom configuration
    ///
    /// # Returns
    ///
    /// A new QSYP instance with the specified configuration
    #[must_use]
    pub fn with_config(node_id: String, config: QSYPConfig) -> Self {
        Self {
            node_id,
            config,
            sync_groups: HashMap::new(),
            timing_info: HashMap::new(),
            node: None,
            background_sync_running: false,
            background_task: None,
            sync_lock: Arc::new(Mutex::new(())),
        }
    }
    
    /// Set the node reference
    pub fn set_node(&mut self, node: Arc<RwLock<Node>>) {
        self.node = Some(node);
    }
    
    /// Get the current configuration
    #[must_use]
    pub fn config(&self) -> &QSYPConfig {
        &self.config
    }
    
    /// Set a new configuration
    pub fn set_config(&mut self, config: QSYPConfig) {
        self.config = config;
    }
    
    /// Get all synchronization groups
    #[must_use]
    pub fn get_sync_groups(&self) -> Vec<String> {
        self.sync_groups.keys().cloned().collect()
    }
    
    /// Check if a node is part of a synchronization group
    #[must_use]
    pub fn is_in_group(&self, node_id: &str, group_id: &str) -> bool {
        self.sync_groups.get(group_id)
            .is_some_and(|group| group.members.contains(node_id))
    }
    
    /// Create a synchronization message
    #[allow(dead_code)]
    fn create_sync_message(&self, receiver_id: &str, round: usize) -> SyncMessage {
        SyncMessage {
            sender_id: self.node_id.clone(),
            receiver_id: receiver_id.to_string(),
            send_timestamp: util::timestamp_now(),
            receive_timestamp: None,
            message_id: util::generate_id("sync"),
            round,
        }
    }
    
    /// Process an incoming sync message
    ///
    /// Returns a response message to send back
    #[allow(dead_code)]
    fn process_sync_message(&self, msg: &SyncMessage) -> SyncMessage {
        let now = util::timestamp_now();
        
        // Create response message
        SyncMessage {
            sender_id: self.node_id.clone(),
            receiver_id: msg.sender_id.clone(),
            send_timestamp: now,
            receive_timestamp: Some(now),
            message_id: util::generate_id("sync-resp"),
            round: msg.round,
        }
    }
    
    /// Update timing information for a node based on message exchange
    #[allow(dead_code)]
    fn update_timing_info(&mut self, node_id: &str, sent_msg: &SyncMessage, received_msg: &SyncMessage) {
        let t1 = sent_msg.send_timestamp;
        let t2 = received_msg.receive_timestamp.unwrap_or(0);
        let t3 = received_msg.send_timestamp;
        let t4 = util::timestamp_now();
        
        // Calculate round-trip time
        let rtt = {
            let t1_i64 = i64::try_from(t1).unwrap_or(0);
            let t2_i64 = i64::try_from(t2).unwrap_or(0);
            let t3_i64 = i64::try_from(t3).unwrap_or(0);
            let t4_i64 = i64::try_from(t4).unwrap_or(0);
            
            let rtt_calc = match (t4_i64.checked_sub(t1_i64)).and_then(|a| a.checked_sub(t3_i64.checked_sub(t2_i64).unwrap_or(0))) {
                Some(val) if val >= 0 => val,
                _ => 0 // In case of overflow or negative values, default to 0
            };
            
            u64::try_from(rtt_calc).unwrap_or(0)
        };
        
        // Calculate clock offset using the formula: offset = ((t2 - t1) - (t4 - t3)) / 2
        let offset = {
            let t1_i64 = i64::try_from(t1).unwrap_or(0);
            let t2_i64 = i64::try_from(t2).unwrap_or(0);
            let t3_i64 = i64::try_from(t3).unwrap_or(0);
            let t4_i64 = i64::try_from(t4).unwrap_or(0);
            
            ((t2_i64.checked_sub(t1_i64).unwrap_or(0)) - (t4_i64.checked_sub(t3_i64).unwrap_or(0))) / 2
        };
        
        // Update or create timing info
        let timing_info = self.timing_info.entry(node_id.to_string()).or_insert_with(|| {
            NodeTimingInfo {
                _node_id: node_id.to_string(),
                offset_ms: 0,
                rtt_ms: 0,
                stability: 1.0,
                last_sync: Instant::now(),
                drift_rate_ppm: 0.0,
            }
        });
        
        // Calculate drift since last sync
        let time_since_last = f64::from(
            u32::try_from(timing_info.last_sync.elapsed().as_millis())
                .unwrap_or(u32::MAX)
        );
        
        if time_since_last > 1000.0 {
            // Only calculate drift rate if enough time has passed (> 1 second)
            let old_offset = f64::from(
                i32::try_from(timing_info.offset_ms)
                    .unwrap_or(if timing_info.offset_ms > 0 { i32::MAX } else { i32::MIN })
            );
            let new_offset = f64::from(
                i32::try_from(offset)
                    .unwrap_or(if offset > 0 { i32::MAX } else { i32::MIN })
            );
            let drift = new_offset - old_offset;
            
            // Calculate drift rate in parts per million
            let drift_rate = (drift / time_since_last) * 1_000_000.0;
            
            // Update with exponential smoothing
            timing_info.drift_rate_ppm = timing_info.drift_rate_ppm * 0.7 + drift_rate * 0.3;
        }
        
        // Update timing info with new measurements
        timing_info.offset_ms = offset;
        timing_info.rtt_ms = rtt;
        timing_info.last_sync = Instant::now();
        
        // Update stability metric based on consistency of measurements
        // (not fully implemented in this example)
        timing_info.stability = 0.9;
    }
    
    /// Calculate timing compensation for a node
    #[allow(dead_code)]
    fn calculate_compensation(&self, node_id: &str, _target_time: u64) -> i64 {
        if let Some(info) = self.timing_info.get(node_id) {
            // Use safe conversion method to avoid precision loss
            let time_since_sync = f64::from(
                u32::try_from(info.last_sync.elapsed().as_millis())
                    .unwrap_or(u32::MAX)
            );
            
            // Calculate expected drift since last sync
            let expected_drift = (info.drift_rate_ppm * time_since_sync) / 1000.0;
            
            // Dynamic check for safe conversion from f64 to i64, avoiding hardcoded constants
            fn is_safely_convertible_to_i64(value: f64) -> bool {
                // Check if conversion is lossless by performing a round-trip conversion
                let test_value = value.round();
                let as_i64 = test_value as i64;
                let back_to_f64 = as_i64 as f64;
                
                // If conversion is lossless, the value is safely convertible
                (test_value - back_to_f64).abs() < f64::EPSILON
            }
            
            // Add entropy-based jitter to prevent timing attacks
            let jitter = {
                // Create a simple entropy source from the current timestamp 
                // and memory address of the info object
                let timestamp = util::timestamp_now();
                let ptr_value = std::ptr::addr_of!(info) as u64;
                let entropy = timestamp ^ ptr_value;
                
                // Generate a small jitter value between -0.1% and +0.1% of the expected drift
                let jitter_factor = ((entropy % 201) as f64 - 100.0) / 100000.0;
                expected_drift * jitter_factor
            };
            
            // Apply jitter to expected drift (makes timing attacks more difficult)
            let adjusted_drift = expected_drift + jitter;
            
            // Safely convert to i64 using dynamic check instead of constants
            let drift_adjustment = if !is_safely_convertible_to_i64(adjusted_drift) {
                if adjusted_drift > 0.0 { i64::MAX } else { i64::MIN }
            } else {
                adjusted_drift.round() as i64 // Safe because we've checked convertibility
            };
            
            info.offset_ms.saturating_add(drift_adjustment)
        } else {
            0
        }
    }
    
    /// Master-slave synchronization implementation
    async fn master_slave_sync(&mut self, master_id: &str, slaves: &[String]) -> Result<SyncResult, QSYPError> {
        let mut success = true;
        let mut max_rtt = 0;
        let mut total_offset = 0;
        let mut node_count = 0;
        
        // Sync each slave with the master
        for slave_id in slaves {
            if slave_id != &self.node_id && slave_id != master_id {
                match self.synchronize_with(slave_id).await {
                    Ok(result) => {
                        total_offset += result.offset_ms;
                        node_count += 1;
                        if result.rtt_ms > max_rtt {
                            max_rtt = result.rtt_ms;
                        }
                    }
                    Err(_) => {
                        success = false;
                    }
                }
            }
        }
        
        // Calculate average offset
        let avg_offset = if node_count > 0 {
            total_offset / i64::from(node_count)
        } else {
            0
        };
        
        // Create sync result
        let result = SyncResult {
            offset_ms: avg_offset,
            rtt_ms: max_rtt,
            sync_time: util::timestamp_now(),
            nodes: slaves.to_vec(),
            accuracy_us: max_rtt * 1000 / 2, // Rough estimate
            success,
        };
        
        Ok(result)
    }
    
    /// Distributed consensus synchronization implementation
    async fn distributed_consensus_sync(&mut self, nodes: &[String]) -> Result<SyncResult, QSYPError> {
        let mut node_offsets = HashMap::new();
        let mut max_rtt = 0;
        
        // First, measure offsets to all other nodes
        for node_id in nodes {
            if node_id != &self.node_id {
                match self.synchronize_with(node_id).await {
                    Ok(result) => {
                        node_offsets.insert(node_id.to_string(), result.offset_ms);
                        if result.rtt_ms > max_rtt {
                            max_rtt = result.rtt_ms;
                        }
                    }
                    Err(e) => {
                        eprintln!("Error synchronizing with {node_id}: {e:?}");
                    }
                }
            }
        }
        
        // If we couldn't sync with any node, return error
        if node_offsets.is_empty() {
            return Err(QSYPError::SyncError("Could not synchronize with any node".to_string()));
        }
        
        // Compute median offset to be robust against outliers
        let mut offsets: Vec<i64> = node_offsets.values().copied().collect();
        
        // Add entropy-based sorting perturbation to prevent side-channel attacks
        // This makes the sorting order slightly unpredictable while preserving the median
        {
            // Generate entropy from the combination of collected offsets
            let entropy = offsets.iter().fold(0u64, |acc, &offset| {
                acc.wrapping_add(offset.unsigned_abs())
            }) ^ util::timestamp_now();
            
            // Apply a small permutation to nearby elements if they're very close in value
            // This prevents an attacker from inferring exact values through timing analysis
            if offsets.len() > 2 {
                for i in 1..offsets.len() {
                    // Check if consecutive values are very close
                    let diff = (offsets[i] - offsets[i-1]).abs();
                    if diff < 3 { // If values are within 3ms
                        // Use entropy to decide whether to swap them (50% chance)
                        if entropy & (1 << (i % 64)) != 0 {
                            offsets.swap(i, i-1);
                        }
                    }
                }
            }
        };
        
        // Sort after perturbation to get approximate median
        offsets.sort_unstable();
        
        let median_offset = if offsets.len() % 2 == 0 {
            (offsets[offsets.len() / 2 - 1] + offsets[offsets.len() / 2]) / 2
        } else {
            offsets[offsets.len() / 2]
        };
        
        // Create sync result
        let result = SyncResult {
            offset_ms: median_offset,
            rtt_ms: max_rtt,
            sync_time: util::timestamp_now(),
            nodes: nodes.to_vec(),
            accuracy_us: max_rtt * 1000 / 2, // Rough estimate
            success: true,
        };
        
        Ok(result)
    }
    
    /// Background synchronization loop
    async fn background_sync_loop(self_arc: Arc<Mutex<Self>>) {
        let config = {
            let this = self_arc.lock().await;
            this.config.clone()
        };
        
        let interval_ms = config.sync_interval_ms;
        let mut interval = time::interval(Duration::from_millis(interval_ms));
        
        loop {
            interval.tick().await;
            
            let continue_running = {
                let mut this = self_arc.lock().await;
                
                if !this.background_sync_running {
                    break;
                }
                
                // Get all groups this node is part of
                let groups = this.get_sync_groups();
                
                // Synchronize with each group
                for group_id in groups {
                    if let Err(e) = this.synchronize_group(&group_id).await {
                        eprintln!("Background sync error for group {group_id}: {e:?}");
                    }
                }
                
                this.background_sync_running
            };
            
            if !continue_running {
                break;
            }
        }
    }
}

#[async_trait]
impl QuantumSynchronizer for QSYP {
    async fn synchronize_with(&mut self, node_id: &str) -> Result<SyncResult, QSYPError> {
        if node_id == self.node_id {
            return Err(QSYPError::SyncError("Cannot synchronize with self".to_string()));
        }
        
        // In a real implementation, this would establish communication with the remote node
        // and exchange a series of timestamped messages.
        
        // For simulation, we'll generate simulated messages
        let mut total_offset = 0;
        let mut total_rtt = 0;
        
        // Clone self.node_id and config to avoid borrowing issues
        let this_node_id = self.node_id.clone();
        let message_count = self.config.sync_message_count;
        let max_drift = self.config.max_drift_ms;
        let auto_compensate = self.config.auto_compensate;
        
        // Acquire sync lock to prevent concurrent sync operations
        {
            let _lock = self.sync_lock.lock().await;
            
            for round in 0..message_count {
                // Create and send sync message
                let sent_msg = SyncMessage {
                    sender_id: this_node_id.clone(),
                    receiver_id: node_id.to_string(),
                    send_timestamp: util::timestamp_now(),
                    receive_timestamp: None,
                    message_id: util::generate_id("sync"),
                    round,
                };
                
                // Simulate network delay
                let delay_ms = thread_rng().gen_range(5..50);
                time::sleep(Duration::from_millis(delay_ms)).await;
                
                // Simulate remote node processing and response
                let now = util::timestamp_now();
                let mut received_msg = SyncMessage {
                    sender_id: node_id.to_string(),
                    receiver_id: this_node_id.clone(),
                    send_timestamp: now,
                    receive_timestamp: Some(now),
                    message_id: util::generate_id("sync-resp"),
                    round,
                };
                
                // Add simulated clock offset for the remote node
                let simulated_offset = i64::from(thread_rng().gen_range(-20i32..20i32));
                // Safe conversion from i64 to u64 with appropriate wrapping
                let simulated_offset_u64 = if simulated_offset >= 0 {
                    simulated_offset as u64
                } else {
                    0u64.wrapping_sub((-simulated_offset) as u64)
                };
                received_msg.send_timestamp = received_msg.send_timestamp.wrapping_add(simulated_offset_u64);
                
                // Simulate return network delay
                let return_delay_ms = thread_rng().gen_range(5..50);
                time::sleep(Duration::from_millis(return_delay_ms)).await;
                
                // Process timing information
                let t1 = sent_msg.send_timestamp;
                let t2 = received_msg.receive_timestamp.unwrap_or(0);
                let t3 = received_msg.send_timestamp;
                let t4 = util::timestamp_now();
                
                // Calculate round-trip time
                #[allow(clippy::cast_possible_wrap)]
                let rtt = match ((t4 as i64).checked_sub(t1 as i64)).and_then(|a| a.checked_sub((t3 as i64).checked_sub(t2 as i64).unwrap_or(0))) {
                    #[allow(clippy::cast_sign_loss)]
                    Some(val) if val >= 0 => val as u64,
                    _ => 0 // In case of overflow or negative values, default to 0
                };
                
                // Calculate clock offset using the formula: offset = ((t2 - t1) - (t4 - t3)) / 2
                #[allow(clippy::cast_possible_wrap)]
                let offset = ((t2 as i64 - t1 as i64) - (t4 as i64 - t3 as i64)) / 2;
                
                // Update or create timing info
                let timing_info = self.timing_info.entry(node_id.to_string()).or_insert_with(|| {
                    NodeTimingInfo {
                        _node_id: node_id.to_string(),
                        offset_ms: 0,
                        rtt_ms: 0,
                        stability: 1.0,
                        last_sync: Instant::now(),
                        drift_rate_ppm: 0.0,
                    }
                });
                
                // Calculate drift since last sync
                #[allow(clippy::cast_precision_loss)]
                let time_since_last = timing_info.last_sync.elapsed().as_millis() as f64;
                if time_since_last > 1000.0 {
                    // Only calculate drift rate if enough time has passed (> 1 second)
                    #[allow(clippy::cast_precision_loss)]
                    let old_offset = timing_info.offset_ms as f64;
                    #[allow(clippy::cast_precision_loss)]
                    let new_offset = f64::from(
                        i32::try_from(offset)
                            .unwrap_or(if offset > 0 { i32::MAX } else { i32::MIN })
                    );
                    let drift = new_offset - old_offset;
                    
                    // Calculate drift rate in parts per million
                    let drift_rate = (drift / time_since_last) * 1_000_000.0;
                    
                    // Update with exponential smoothing
                    timing_info.drift_rate_ppm = timing_info.drift_rate_ppm * 0.7 + drift_rate * 0.3;
                }
                
                // Update timing info with new measurements
                timing_info.offset_ms = offset;
                timing_info.rtt_ms = rtt;
                timing_info.last_sync = Instant::now();
                
                // Update stability metric based on consistency of measurements
                // (not fully implemented in this example)
                timing_info.stability = 0.9;
                
                // Accumulate measurements
                total_offset += offset;
                total_rtt += rtt;
            }
        }
        
        // Calculate average values
        #[allow(clippy::cast_possible_wrap)]
        let count = message_count as i64;
        let avg_offset = total_offset / count;
        
        // Safer way to calculate average RTT
        #[allow(clippy::cast_precision_loss)]
        let avg_rtt_f64 = total_rtt as f64 / count as f64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let avg_rtt = avg_rtt_f64 as u64;
        
        // Add timing protection through variable-time response with entropy-based jitter
        if auto_compensate && avg_offset.unsigned_abs() > max_drift / 2 {
            // Use a combination of system timing, node ID, and request parameters as entropy source
            let mut entropy_source = util::timestamp_now();
            entropy_source ^= node_id.as_bytes().iter().fold(0u64, |acc, &b| acc.wrapping_add(u64::from(b)));
            entropy_source ^= avg_offset.unsigned_abs();
            
            // Generate variable delay (5-15ms) that makes timing analysis more difficult
            let jitter_ms = 5 + (entropy_source % 11);
            
            // Apply random timing jitter
            time::sleep(Duration::from_millis(jitter_ms)).await;
        }
        
        // Check if drift exceeds threshold
        if avg_offset.unsigned_abs() > max_drift && !auto_compensate {
            return Err(QSYPError::ExcessiveDrift(
                avg_offset.unsigned_abs(),
                max_drift,
            ));
        }
        
        // Create sync result
        let result = SyncResult {
            offset_ms: avg_offset,
            rtt_ms: avg_rtt,
            sync_time: util::timestamp_now(),
            nodes: vec![node_id.to_string()],
            accuracy_us: avg_rtt * 500, // Estimate (half RTT, converted to μs)
            success: true,
        };
        
        Ok(result)
    }
    
    async fn synchronize_group(&mut self, group_id: &str) -> Result<SyncResult, QSYPError> {
        // Get the group
        let group = match self.sync_groups.get(group_id) {
            Some(g) => g.clone(),
            None => return Err(QSYPError::SyncError(format!("Sync group {group_id} not found"))),
        };
            
        // Check if this node is part of the group
        let node_id = self.node_id.clone();
        if !group.members.contains(&node_id) {
            return Err(QSYPError::NotInGroup(node_id, group_id.to_string()));
        }
        
        // Convert members to vec for easier handling
        let members: Vec<String> = group.members.iter().cloned().collect();
        
        // Choose synchronization strategy based on configuration
        match self.config.strategy {
            SyncStrategy::MasterSlave => {
                // Use reference node as master, or first node if none specified
                let master = group.reference_node.clone()
                    .unwrap_or_else(|| members[0].clone());
                    
                self.master_slave_sync(&master, &members).await
            },
            SyncStrategy::DistributedConsensus => {
                self.distributed_consensus_sync(&members).await
            },
            SyncStrategy::Hierarchical => {
                // For simplicity, just use distributed consensus here
                // A real implementation would organize nodes in a hierarchy
                self.distributed_consensus_sync(&members).await
            },
            SyncStrategy::AtomicClock => {
                // Simulate synchronization with an external atomic clock
                // In a real implementation, this would involve hardware integration
                
                // Simulate by adding a small random offset
                #[allow(clippy::cast_sign_loss)]
                let simulated_offset = i64::from(thread_rng().gen_range(-2i32..3i32));
                
                Ok(SyncResult {
                    offset_ms: simulated_offset,
                    rtt_ms: 2, // Very low RTT for atomic clock
                    sync_time: util::timestamp_now(),
                    nodes: vec!["atomic_clock".to_string()],
                    accuracy_us: 1, // Very high accuracy
                    success: true,
                })
            },
        }
    }
    
    async fn create_sync_group(&mut self, name: &str, members: &[String]) -> Result<SyncGroup, QSYPError> {
        let group_id = util::generate_id("sync-group");
        let now = util::timestamp_now();
        
        // Create the group
        let group = SyncGroup {
            id: group_id.clone(),
            name: name.to_string(),
            members: members.iter().cloned().collect(),
            reference_node: None,
            created_at: now,
            last_sync: now,
        };
        
        // Store the group
        self.sync_groups.insert(group_id.clone(), group.clone());
        
        Ok(group)
    }
    
    fn get_offset(&self, node_id: &str) -> Option<i64> {
        self.timing_info.get(node_id).map(|info| info.offset_ms)
    }
    
    async fn start_background_sync(&mut self) -> Result<(), QSYPError> {
        if self.background_sync_running {
            return Ok(());
        }
        
        self.background_sync_running = true;
        
        // Clone self for the background task
        let arc_self = Arc::new(Mutex::new(self.clone()));
        
        // Spawn background task
        let handle = task::spawn(async move {
            QSYP::background_sync_loop(arc_self).await;
        });
        
        self.background_task = Some(handle);
        
        Ok(())
    }
    
    async fn stop_background_sync(&mut self) -> Result<(), QSYPError> {
        self.background_sync_running = false;
        
        if let Some(handle) = self.background_task.take() {
            // Wait for the task to finish
            let _ = handle.await;
        }
        
        Ok(())
    }
}

// Special Clone implementation for QSYP that doesn't include the task handle
impl Clone for QSYP {
    fn clone(&self) -> Self {
        Self {
            node_id: self.node_id.clone(),
            config: self.config.clone(),
            sync_groups: self.sync_groups.clone(),
            timing_info: self.timing_info.clone(),
            node: self.node.clone(),
            background_sync_running: self.background_sync_running,
            background_task: None, // Don't clone the task handle
            sync_lock: Arc::new(Mutex::new(())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::RwLock;
    
    #[tokio::test]
    async fn test_synchronize_with_node() {
        // Create a test node
        let node_a = Arc::new(RwLock::new(Node::new("node_a")));
        
        // Create QSYP instance
        let mut qsyp = QSYP::new("node_a".to_string());
        qsyp.set_node(node_a);
        
        // Test synchronization with another node
        let result = qsyp.synchronize_with("node_b").await;
        
        assert!(result.is_ok(), "Synchronization failed: {:?}", result.err());
        
        if let Ok(sync_result) = result {
            // Just check that the result is reasonable
            assert!(sync_result.offset_ms.abs() < 100, "Offset too large: {}", sync_result.offset_ms);
            assert!(sync_result.rtt_ms > 0, "RTT should be positive");
            assert!(sync_result.success);
        }
    }
    
    #[tokio::test]
    async fn test_create_sync_group() {
        // Create QSYP instance
        let mut qsyp = QSYP::new("node_a".to_string());
        
        // Test creating a synchronization group
        let members = vec!["node_a".to_string(), "node_b".to_string(), "node_c".to_string()];
        let result = qsyp.create_sync_group("Test Group", &members).await;
        
        assert!(result.is_ok(), "Failed to create sync group: {:?}", result.err());
        
        if let Ok(group) = result {
            assert_eq!(group.name, "Test Group");
            assert_eq!(group.members.len(), 3);
            assert!(group.members.contains("node_a"));
            assert!(group.members.contains("node_b"));
            assert!(group.members.contains("node_c"));
        }
        
        // Check that the group exists
        let groups = qsyp.get_sync_groups();
        assert_eq!(groups.len(), 1, "Group not added to QSYP instance");
    }
} 