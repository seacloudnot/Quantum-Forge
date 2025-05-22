// Expose QKD Methods for Examples
//
// This module re-exports QKD methods with public visibility for use in examples.

use crate::security::qkd::QKD;
use crate::security::QuantumKeyDistribution;

pub trait QKDExampleMethods {
    /// Initialize a QKD session with the specified peer
    ///
    /// # Arguments
    ///
    /// * `peer_id` - The ID of the peer node
    ///
    /// # Returns
    ///
    /// The session ID if successful
    ///
    /// # Errors
    ///
    /// Returns an error if the peer is not authenticated or initialization fails
    fn init_session_example(&mut self, peer_id: &str) -> Result<String, String>;

    /// Compare measurement bases with a peer
    ///
    /// # Arguments
    ///
    /// * `session_id` - The session ID
    /// * `peer_bases` - The peer's measurement bases
    ///
    /// # Returns
    ///
    /// Indices where bases match if successful
    ///
    /// # Errors
    ///
    /// Returns an error if the session doesn't exist or is in an invalid state
    fn compare_bases_example(&mut self, session_id: &str, peer_bases: &[u8]) -> Result<Vec<usize>, String>;
}

impl QKDExampleMethods for QKD {
    fn init_session_example(&mut self, peer_id: &str) -> Result<String, String> {
        self.init_session(peer_id)
    }
    
    fn compare_bases_example(&mut self, session_id: &str, peer_bases: &[u8]) -> Result<Vec<usize>, String> {
        self.compare_bases(session_id, peer_bases)
            .map_err(|e| e.to_string())
    }
} 