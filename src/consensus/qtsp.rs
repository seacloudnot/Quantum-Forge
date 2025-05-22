// Quantum Threshold Signature Protocol (QTSP)
//
// This protocol enables group-based quantum signatures with threshold reconstruction
// for secure distributed consensus operations.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use tokio::sync::RwLock;
use rand::Rng;

use crate::network::Node;
use crate::security::qkd::QKD;
use crate::security::qrng::{QRNG, QRNGConfig};
use crate::util;
use crate::error;

/// Errors specific to QTSP
#[derive(Error, Debug)]
pub enum QTSPError {
    /// Key sharing error
    #[error("Key sharing error: {0}")]
    KeySharingError(String),
    
    /// Threshold error
    #[error("Threshold error: {0}")]
    ThresholdError(String),
    
    /// Signature verification error
    #[error("Signature verification error: {0}")]
    VerificationError(String),
    
    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),
    
    /// Timeout error
    #[error("Operation timed out after {0:?}")]
    Timeout(Duration),
    
    /// Missing key share
    #[error("Missing key share for node {0}")]
    MissingKeyShare(String),
    
    /// Invalid share
    #[error("Invalid share provided by node {0}")]
    InvalidShare(String),
    
    /// Invalid signature
    #[error("Invalid signature")]
    InvalidSignature,
    
    /// Unauthorized node
    #[error("Node {0} not authorized to participate")]
    UnauthorizedNode(String),
    
    /// General error
    #[error("General error: {0}")]
    General(String),
    
    /// Not enough entanglements
    #[error("Not enough entanglements")]
    NotEnoughEntanglements,
    
    /// Node not found
    #[error("Node not found: {0}")]
    NodeNotFound(String),
    
    /// Session not found
    #[error("Session not found: {0}")]
    SessionNotFound(String),
    
    /// Share not found
    #[error("Share not found")]
    ShareNotFound,
    
    /// Signing failed
    #[error("Signing failed: {0}")]
    SigningFailed(String),
}

// Implement From<error::Error> for QTSPError
impl From<error::Error> for QTSPError {
    fn from(err: error::Error) -> Self {
        QTSPError::General(err.to_string())
    }
}

/// Signature scheme to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignatureScheme {
    /// Classical threshold signatures with quantum key sharing
    ClassicalThreshold,
    
    /// Quantum one-time signatures
    QuantumOneTime,
    
    /// Quantum many-worlds signatures
    QuantumManyWorlds,
    
    /// Lattice-based post-quantum signatures
    LatticeBased,
    
    /// Hash-based signatures
    HashBased,
}

/// Configuration for QTSP
#[derive(Debug, Clone)]
pub struct QTSPConfig {
    /// Signature scheme to use
    pub signature_scheme: SignatureScheme,
    
    /// Threshold (k) out of n nodes required to create a signature
    pub threshold: usize,
    
    /// Timeout for operations in milliseconds
    pub timeout_ms: u64,
    
    /// Key size in bits
    pub key_size: usize,
    
    /// Signature validity period in seconds
    pub validity_period_seconds: u64,
    
    /// Whether to use rotating keys
    pub use_rotating_keys: bool,
    
    /// Rotation period in seconds (if rotating keys are used)
    pub key_rotation_period_seconds: u64,
}

impl Default for QTSPConfig {
    fn default() -> Self {
        Self {
            signature_scheme: SignatureScheme::QuantumOneTime,
            threshold: 3,
            timeout_ms: 30000, // 30 seconds
            key_size: 256,
            validity_period_seconds: 3600, // 1 hour
            use_rotating_keys: true,
            key_rotation_period_seconds: 86400, // 24 hours
        }
    }
}

/// A key share for threshold signatures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyShare {
    /// ID of this key share
    pub id: String,
    
    /// Index of this share (1-based)
    pub index: usize,
    
    /// Node that owns this share
    pub node_id: String,
    
    /// The share value (encrypted)
    pub share_value: Vec<u8>,
    
    /// When this share was created
    pub timestamp: u64,
    
    /// Validity period end time
    pub valid_until: u64,
    
    /// Key generation ID this share belongs to
    pub key_gen_id: String,
}

/// A quantum threshold signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSignature {
    /// Unique signature ID
    pub id: String,
    
    /// Message that was signed (hash)
    pub message_hash: Vec<u8>,
    
    /// The actual signature value
    pub signature: Vec<u8>,
    
    /// List of nodes that participated in signing
    pub signer_node_ids: Vec<String>,
    
    /// When the signature was created
    pub timestamp: u64,
    
    /// Validity period end time
    pub valid_until: u64,
    
    /// Key generation ID used for this signature
    pub key_gen_id: String,
    
    /// Signature scheme used
    pub scheme: SignatureScheme,
}

/// A key generation session
#[derive(Debug, Clone)]
pub struct KeyGenSession {
    /// Unique ID for this key generation
    pub id: String,
    
    /// When the key generation was started
    pub timestamp: u64,
    
    /// Participating nodes
    pub participants: HashSet<String>,
    
    /// Committed shares by node
    pub committed_shares: HashMap<String, bool>,
    
    /// Whether the key generation is complete
    pub is_complete: bool,
    
    /// The public verification key
    pub public_key: Option<Vec<u8>>,
    
    /// When this key expires
    pub valid_until: u64,
}

/// A signing session
#[derive(Debug, Clone)]
pub struct SigningSession {
    /// Unique ID for this signing session
    pub id: String,
    
    /// Message to sign (usually a hash)
    pub message_hash: Vec<u8>,
    
    /// When the signing was initiated
    pub timestamp: u64,
    
    /// Key generation ID to use
    pub key_gen_id: String,
    
    /// Nodes that have submitted shares
    pub submitted_shares: HashMap<String, Vec<u8>>,
    
    /// Whether signing is complete
    pub is_complete: bool,
    
    /// The resulting signature (if complete)
    pub signature: Option<Vec<u8>>,
}

/// QTSP implementation
pub struct QTSP {
    /// Node ID running this QTSP instance
    node_id: String,
    
    /// Configuration
    config: QTSPConfig,
    
    /// Reference to the node
    node: Option<Arc<RwLock<Node>>>,
    
    /// Authorized nodes for key generation and signing
    authorized_nodes: HashSet<String>,
    
    /// Key shares owned by this node
    my_key_shares: HashMap<String, KeyShare>,
    
    /// Active key generation sessions
    key_gen_sessions: HashMap<String, KeyGenSession>,
    
    /// Active signing sessions
    signing_sessions: HashMap<String, SigningSession>,
    
    /// Verified signatures
    verified_signatures: HashMap<String, QuantumSignature>,
    
    /// Public verification keys by key generation ID
    verification_keys: HashMap<String, Vec<u8>>,
    
    /// QKD instance for secure communication
    #[allow(dead_code)]
    qkd: QKD,
    
    /// QRNG instance for randomness
    qrng: QRNG,
}

impl QTSP {
    /// Create a new QTSP instance
    ///
    /// # Arguments
    ///
    /// * `node_id` - The ID of the node running this instance
    ///
    /// # Returns
    ///
    /// A new QTSP instance
    #[must_use]
    pub fn new(node_id: &str) -> Self {
        let node_id_clone = node_id.to_string();
        Self {
            node_id: node_id_clone.clone(),
            config: QTSPConfig::default(),
            node: None,
            authorized_nodes: HashSet::new(),
            my_key_shares: HashMap::new(),
            key_gen_sessions: HashMap::new(),
            signing_sessions: HashMap::new(),
            verified_signatures: HashMap::new(),
            verification_keys: HashMap::new(),
            qkd: QKD::new(node_id_clone),
            qrng: QRNG::new(QRNGConfig::default()),
        }
    }
    
    /// Create a new QTSP instance with the specified configuration
    ///
    /// # Arguments
    ///
    /// * `node_id` - The ID of the node running this instance
    /// * `config` - The configuration to use
    ///
    /// # Returns
    ///
    /// A new QTSP instance with the specified configuration
    #[must_use]
    pub fn with_config(node_id: &str, config: QTSPConfig) -> Self {
        let node_id_clone = node_id.to_string();
        Self {
            node_id: node_id_clone.clone(),
            config,
            node: None,
            authorized_nodes: HashSet::new(),
            my_key_shares: HashMap::new(),
            key_gen_sessions: HashMap::new(),
            signing_sessions: HashMap::new(),
            verified_signatures: HashMap::new(),
            verification_keys: HashMap::new(),
            qkd: QKD::new(node_id_clone),
            qrng: QRNG::new(QRNGConfig::default()),
        }
    }
    
    /// Set the node reference
    pub fn set_node(&mut self, node: Arc<RwLock<Node>>) {
        self.node = Some(node);
    }
    
    /// Add an authorized node
    pub fn add_authorized_node(&mut self, node_id: String) {
        self.authorized_nodes.insert(node_id);
    }
    
    /// Remove an authorized node
    pub fn remove_authorized_node(&mut self, node_id: &str) {
        self.authorized_nodes.remove(node_id);
    }
    
    /// Get the set of authorized nodes in the threshold group
    #[must_use]
    pub fn authorized_nodes(&self) -> &HashSet<String> {
        &self.authorized_nodes
    }
    
    /// Initialize key generation
    ///
    /// # Returns
    ///
    /// Session ID or error
    ///
    /// # Errors
    ///
    /// Returns error if threshold not met or other generation issues
    #[must_use = "Contains the key generation session ID"]
    pub fn init_key_generation(&mut self) -> Result<String, QTSPError> {
        // Check if we have enough authorized nodes
        if self.authorized_nodes.len() < self.config.threshold {
            return Err(QTSPError::ThresholdError(format!(
                "Not enough authorized nodes. Have {}, need {}",
                self.authorized_nodes.len(),
                self.config.threshold
            )));
        }
        
        // Generate a unique session ID
        let session_id = util::generate_id("qtsp_key");
        
        // Create a key generation session
        let session = KeyGenSession {
            id: session_id.clone(),
            timestamp: util::timestamp_now(),
            participants: self.authorized_nodes.clone(),
            committed_shares: HashMap::new(),
            is_complete: false,
            public_key: None,
            valid_until: util::timestamp_now() + (self.config.validity_period_seconds * 1000),
        };
        
        // Store the session
        self.key_gen_sessions.insert(session_id.clone(), session);
        
        // In a real implementation, we would broadcast the session to all participants
        // For now, we'll simulate it with a direct call to generate_key_shares
        self.generate_key_shares(&session_id)?;
        
        Ok(session_id)
    }
    
    /// Generate key shares for a quantum signing session
    ///
    /// For simulation, this creates simple RSA key shares that would be
    /// entangled in a real quantum implementation.
    ///
    /// # Arguments
    ///
    /// * `session_id` - ID of the signing session
    ///
    /// # Returns
    ///
    /// Ok or error if the session is invalid
    fn generate_key_shares(&mut self, session_id: &str) -> Result<(), QTSPError> {
        // Get the session
        let session = self.key_gen_sessions.get_mut(session_id).ok_or_else(|| {
            QTSPError::KeySharingError(format!("Session not found: {session_id}"))
        })?;
        
        // Check if already generated
        if session.committed_shares.len() == session.participants.len() {
            return Ok(());
        }
        
        // Generate a simulated quantum signing key
        // In a real implementation, this would use quantum key distribution (QKD)
        // or quantum key encapsulation mechanisms (QKEM)
        
        // In a real implementation, we would use quantum-resistant cryptography
        // For simulation, we'll create simplified key shares
        let bits = 2048;
        let mut rng = rand::thread_rng();
        
        // Generate simulated key material
        let private_key_material = (0..bits/8).map(|_| rng.gen::<u8>()).collect::<Vec<u8>>();
        let public_key_material = private_key_material.iter().map(|&b| !b).collect::<Vec<u8>>();
        
        // Store public key in the session
        session.public_key = Some(public_key_material.clone());
        
        // Store the public key in verification_keys map for verification
        self.verification_keys.insert(session_id.to_string(), public_key_material.clone());
        
        // Create threshold key shares
        let threshold = session.participants.len();
        let mut index = 1;
        
        for node_id in &session.participants {
            // Create a unique key share for each node
            // This isn't proper threshold cryptography, just a simulation
            let share_id = format!("share-{session_id}-{node_id}");
            
            // Create a unique share by XORing with the node ID
            let mut share = private_key_material.clone();
            for (j, byte) in node_id.bytes().enumerate() {
                if j < share.len() {
                    share[j] ^= byte;
                }
            }
            
            // Instead of real quantum key sharing, just store simulated key material
            session.committed_shares.insert(node_id.clone(), true);
            
            let share_value = self.qrng.generate_bytes(self.config.key_size / 8)?;
            
            let share = KeyShare {
                id: share_id,
                index,
                node_id: node_id.clone(),
                share_value,
                timestamp: util::timestamp_now(),
                valid_until: session.valid_until,
                key_gen_id: session_id.to_string(),
            };
            
            // If this is our node, store the share
            if node_id == &self.node_id {
                self.my_key_shares.insert(session_id.to_string(), share);
            }
            
            index += 1;
        }
        
        // Check if all participants have committed
        if session.committed_shares.len() == threshold {
            session.is_complete = true;
        }
        
        Ok(())
    }
    
    /// Sign a message using threshold signature
    ///
    /// # Arguments
    ///
    /// * `message` - Message to sign
    ///
    /// # Returns
    ///
    /// Quantum signature or error
    ///
    /// # Errors
    ///
    /// Returns error if no valid key generation exists or other signing issues
    #[must_use = "Contains the quantum signature for the message"]
    pub fn sign_message(&mut self, message: &[u8]) -> Result<QuantumSignature, QTSPError> {
        // Find a valid key generation to use
        let key_gen_id = self.find_valid_key_gen()?;
        
        // Get our share for the key generation
        let key_gen = self.key_gen_sessions.get(&key_gen_id)
            .ok_or_else(|| QTSPError::SessionNotFound(key_gen_id.clone()))?;
        
        // Check if this node has a share
        let node_share = *key_gen.committed_shares.get(&self.node_id)
            .ok_or(QTSPError::ShareNotFound)?;
        
        // Calculate message hash
        let message_hash = util::hash_bytes(message);
        
        // Create our signature share
        let partial_signature = Self::create_signature_share(&message_hash, node_share);
        
        // Create the signature request
        let signature_id = util::generate_id("qtsp_sig");
        
        let signature_session = SigningSession {
            id: signature_id.clone(),
            message_hash: message_hash.clone(),
            timestamp: util::timestamp_now(),
            key_gen_id: key_gen_id.clone(),
            submitted_shares: HashMap::new(),
            is_complete: false,
            signature: None,
        };
        
        // Store signature session
        self.signing_sessions.insert(signature_id.clone(), signature_session);
        
        // Add our partial signature
        self.submit_signature_share(
            &signature_id,
            partial_signature
        )?;
        
        // For simulation, generate signatures from other nodes
        self.simulate_other_nodes_signatures(&signature_id)?;
        
        // Try to combine signatures
        self.combine_signatures(&signature_id)?;
        
        // At this point, the signature should be in verified_signatures
        // Find the signature that matches this key_gen_id and message_hash
        for signature in self.verified_signatures.values() {
            if signature.key_gen_id == key_gen_id && 
               signature.message_hash == message_hash {
                return Ok(signature.clone());
            }
        }
        
        Err(QTSPError::SigningFailed("Failed to create signature".into()))
    }
    
    /// Find a valid key generation
    fn find_valid_key_gen(&self) -> Result<String, QTSPError> {
        let now = util::timestamp_now();
        
        // Find the most recent valid key generation
        let valid_key_gen = self.key_gen_sessions.iter()
            .filter(|(_, session)| session.is_complete && session.valid_until > now)
            .max_by_key(|(_, session)| session.timestamp);
            
        if let Some((key_gen_id, _)) = valid_key_gen {
            return Ok(key_gen_id.clone());
        }
        
        // No valid key generation found
        Err(QTSPError::KeySharingError("No valid key generation available".to_string()))
    }
    
    /// Submit a signature share from a node
    ///
    /// In a real implementation, this would involve quantum secret sharing
    /// and protected quantum channels. For simulation, we're just collecting
    /// signature shares.
    ///
    /// # Arguments
    ///
    /// * `session_id` - ID of the signing session
    /// * `share_value` - Signature share value
    /// * `node_id` - ID of the node submitting the share
    ///
    /// # Returns
    ///
    /// Ok or error if the session or node is invalid
    fn submit_signature_share(
        &mut self,
        session_id: &str,
        share_value: Vec<u8>,
    ) -> Result<(), QTSPError> {
        // Get the session
        let session = self.signing_sessions.get_mut(session_id).ok_or_else(|| {
            QTSPError::VerificationError(format!("Signing session not found: {session_id}"))
        })?;
        
        // Submit our share
        session.submitted_shares.insert(self.node_id.clone(), share_value);
        
        // Check if we have enough shares to complete signing
        if session.submitted_shares.len() >= self.config.threshold {
            // We would normally call complete_signing here, but we'll let the caller do it
            // for simulation simplicity
        }
        
        Ok(())
    }
    
    /// Submit a share from another node (internal helper)
    ///
    /// This is a simplified version for simulation; in a real implementation,
    /// this would involve quantum secure channels and verification.
    ///
    /// # Arguments
    ///
    /// * `session_id` - ID of the signing session
    /// * `node_id` - ID of the node submitting
    /// * `data` - Data being signed
    ///
    /// # Returns
    ///
    /// Ok or error if the session or node is invalid
    fn submit_share_from_node(
        &mut self,
        session_id: &str,
        node_id: String,
        data: Vec<u8>
    ) -> Result<(), QTSPError> {
        // Verify the node is authorized
        if !self.authorized_nodes.contains(&node_id) {
            return Err(QTSPError::UnauthorizedNode(node_id));
        }
        
        // Get the session
        let session = self.signing_sessions.get_mut(session_id).ok_or_else(|| {
            QTSPError::VerificationError(format!("Signing session not found: {session_id}"))
        })?;
        
        // Submit the share
        session.submitted_shares.insert(node_id, data);
        
        Ok(())
    }
    
    /// Complete a signing session, combining shares into a final signature
    ///
    /// In a real quantum implementation, this would involve quantum multi-party
    /// computation and entanglement-based signature aggregation.
    ///
    /// # Arguments
    ///
    /// * `session_id` - ID of the signing session
    ///
    /// # Returns
    ///
    /// The combined quantum signature or error if insufficient shares
    #[allow(dead_code)]
    fn complete_signing(&mut self, session_id: &str) -> Result<QuantumSignature, QTSPError> {
        // Get the session
        let session = self.signing_sessions.get_mut(session_id).ok_or_else(|| {
            QTSPError::VerificationError(format!("Signing session not found: {session_id}"))
        })?;
        
        // Check if we have enough shares
        if session.submitted_shares.len() < self.config.threshold {
            return Err(QTSPError::ThresholdError(format!(
                "Not enough shares: {} < {}",
                session.submitted_shares.len(),
                self.config.threshold
            )));
        }
        
        // In a real implementation, we would combine the shares to reconstruct the signature
        // For simulation, we'll just concatenate the first threshold number of shares
        
        let mut combined_signature = Vec::new();
        let mut signer_node_ids = Vec::new();
        
        for (node_id, share) in session.submitted_shares.iter().take(self.config.threshold) {
            combined_signature.extend_from_slice(share);
            signer_node_ids.push(node_id.clone());
        }
        
        // Create signature object
        let signature = QuantumSignature {
            id: util::generate_id("qtsp-sig"),
            message_hash: session.message_hash.clone(),
            signature: combined_signature,
            signer_node_ids,
            timestamp: util::timestamp_now(),
            valid_until: util::timestamp_now() + (self.config.validity_period_seconds * 1000),
            key_gen_id: session.key_gen_id.clone(),
            scheme: self.config.signature_scheme,
        };
        
        // Store signature
        self.verified_signatures.insert(signature.id.clone(), signature.clone());
        
        // Mark session as complete
        session.is_complete = true;
        session.signature = Some(signature.signature.clone());
        
        Ok(signature)
    }
    
    /// Verify a quantum threshold signature
    ///
    /// # Arguments
    ///
    /// * `message` - Message that was signed
    /// * `signature` - Signature to verify
    ///
    /// # Returns
    ///
    /// Result containing true if signature is valid, false otherwise
    ///
    /// # Errors
    ///
    /// * `QTSPError::InvalidSignature` - If signature format is invalid
    /// * `QTSPError::VerificationError` - If verification process fails
    pub fn verify_signature(
        &self,
        message: &[u8],
        signature: &QuantumSignature,
    ) -> Result<bool, QTSPError> {
        // Get public verification key for this key generation
        let _public_key = self.verification_keys.get(&signature.key_gen_id).ok_or_else(|| {
            QTSPError::VerificationError(format!(
                "No verification key for key generation {}",
                signature.key_gen_id
            ))
        })?;
        
        // Check signature validity period
        let now = util::timestamp_now();
        if signature.valid_until < now {
            return Err(QTSPError::VerificationError("Signature has expired".to_string()));
        }
        
        // Compute message hash
        let computed_hash = util::hash_bytes(message);
        
        // Verify hash matches
        if computed_hash != signature.message_hash {
            return Err(QTSPError::VerificationError("Message hash mismatch".to_string()));
        }
        
        // In a real implementation, we would verify the signature cryptographically
        // For simulation, we'll check that:
        // 1. There are at least threshold signers
        // 2. All signers are authorized
        
        if signature.signer_node_ids.len() < self.config.threshold {
            return Err(QTSPError::ThresholdError(format!(
                "Not enough signers: {} < {}",
                signature.signer_node_ids.len(),
                self.config.threshold
            )));
        }
        
        for node_id in &signature.signer_node_ids {
            if !self.authorized_nodes.contains(node_id) {
                return Err(QTSPError::UnauthorizedNode(node_id.clone()));
            }
        }
        
        // For simulation, all valid signatures are accepted
        Ok(true)
    }
    
    /// Rotate keys by generating a new key
    ///
    /// # Returns
    ///
    /// New key generation ID or error
    ///
    /// # Errors
    ///
    /// Returns error if key generation fails
    #[must_use = "Contains the ID of the newly generated key"]
    pub fn rotate_keys(&mut self) -> Result<String, QTSPError> {
        // Initialize a new key generation
        let new_key_gen_id = self.init_key_generation()?;
        
        // In a real implementation, we might want to invalidate old keys
        // or add logic to transition between key generations
        
        Ok(new_key_gen_id)
    }
    
    /// Get a signature by ID
    #[must_use]
    pub fn get_signature(&self, signature_id: &str) -> Option<&QuantumSignature> {
        self.verified_signatures.get(signature_id)
    }
    
    /// Get my key shares
    #[must_use]
    pub fn my_key_shares(&self) -> &HashMap<String, KeyShare> {
        &self.my_key_shares
    }
    
    /// Get all verification keys
    #[must_use]
    pub fn verification_keys(&self) -> &HashMap<String, Vec<u8>> {
        &self.verification_keys
    }

    /// Create a signature share using a share value and message hash
    ///
    /// # Arguments
    ///
    /// * `message_hash` - Hash of the message being signed
    /// * `share_value` - Key share value to use for signing
    ///
    /// # Returns
    ///
    /// A partial signature derived from the key share
    #[must_use]
    fn create_signature_share(message_hash: &[u8], share_value: bool) -> Vec<u8> {
        // Create a byte array with hash and share
        let mut signature = Vec::with_capacity(message_hash.len() + 1);
        
        // Add the hash
        signature.extend_from_slice(message_hash);
        
        // Add the share value as a byte
        let share_byte = u8::from(share_value);
        signature.push(share_byte);
        
        signature
    }
    
    /// Simulate other nodes submitting signature shares
    ///
    /// # Arguments
    ///
    /// * `session_id` - ID of the signature session
    ///
    /// # Returns
    ///
    /// Ok(()) on success or appropriate error
    ///
    /// # Errors
    ///
    /// Returns error if the session is invalid or simulation fails
    fn simulate_other_nodes_signatures(&mut self, session_id: &str) -> Result<(), QTSPError> {
        // Get the session
        let signature_session = self.signing_sessions.get(session_id).ok_or_else(|| {
            QTSPError::SessionNotFound(session_id.to_string())
        })?;
        
        // Get the key generation ID
        let key_gen_id = &signature_session.key_gen_id;
        
        // Find all nodes from the key generation that aren't us
        if let Some(key_gen_session) = self.key_gen_sessions.get(key_gen_id) {
            // Get other nodes who participated in key generation
            let other_nodes: Vec<String> = key_gen_session.participants.iter()
                .filter(|id| **id != self.node_id)
                .cloned()
                .collect();
            
            // For simulation, we'll have just enough nodes sign to meet threshold
            let nodes_needed = (self.config.threshold - 1).min(other_nodes.len());
            
            for i in 0..nodes_needed {
                if i < other_nodes.len() {
                    let node_id = &other_nodes[i];
                    
                    // Create a simulated share value
                    let share_data = self.qrng.generate_bytes(32)?;
                    
                    // Submit this node's share
                    self.submit_share_from_node(session_id, node_id.clone(), share_data)?;
                }
            }
            
            Ok(())
        } else {
            Err(QTSPError::SessionNotFound(key_gen_id.clone()))
        }
    }
    
    /// Combine partial signatures into a complete signature
    ///
    /// # Arguments
    ///
    /// * `session_id` - ID of the signature session
    ///
    /// # Returns
    ///
    /// Ok(()) on success or appropriate error
    ///
    /// # Errors
    ///
    /// Returns error if combination fails or threshold not met
    fn combine_signatures(&mut self, session_id: &str) -> Result<(), QTSPError> {
        // Check if we have enough signatures to combine
        let session = self.signing_sessions.get(session_id).ok_or_else(|| {
            QTSPError::SessionNotFound(session_id.to_string())
        })?;
        
        if session.submitted_shares.len() < self.config.threshold {
            return Err(QTSPError::ThresholdError(format!(
                "Not enough signature shares: {} < {}",
                session.submitted_shares.len(),
                self.config.threshold
            )));
        }
        
        // In a real implementation, this would combine shares to reconstruct the signature
        // For simulation, we'll create a complete signature that can be verified
        
        // Collect all the shares and contributing node IDs
        let mut combined_signature = Vec::new();
        let mut signer_node_ids = Vec::new();
        
        let key_gen_id = session.key_gen_id.clone();
        let message_hash = session.message_hash.clone();
        let timestamp = util::timestamp_now();
        let valid_until = timestamp + (self.config.validity_period_seconds * 1000);
        
        // Take threshold number of shares
        for (node_id, share) in session.submitted_shares.iter().take(self.config.threshold) {
            // Add to the combined signature data
            combined_signature.extend_from_slice(share);
            signer_node_ids.push(node_id.clone());
        }
        
        // Create the signature
        let signature = QuantumSignature {
            id: util::generate_id("sig"),
            message_hash,
            signature: combined_signature,
            signer_node_ids,
            timestamp,
            valid_until,
            key_gen_id,
            scheme: self.config.signature_scheme,
        };
        
        // Store in verified signatures
        self.verified_signatures.insert(signature.id.clone(), signature);
        
        // Update session
        let session = self.signing_sessions.get_mut(session_id).ok_or_else(|| {
            QTSPError::SessionNotFound(session_id.to_string())
        })?;
        
        session.is_complete = true;
        session.signature = Some(session.submitted_shares.values().next().unwrap().clone());
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_key_generation() {
        let mut qtsp = QTSP::new("test_node");
        
        // Add authorized nodes
        qtsp.add_authorized_node("test_node".to_string());
        qtsp.add_authorized_node("node1".to_string());
        qtsp.add_authorized_node("node2".to_string());
        qtsp.add_authorized_node("node3".to_string());
        
        // Initialize key generation
        let key_gen_id = qtsp.init_key_generation().unwrap();
        
        // Check that key generation completed
        let session = qtsp.key_gen_sessions.get(&key_gen_id).unwrap();
        assert!(session.is_complete, "Key generation should be complete");
        assert!(session.public_key.is_some(), "Public key should be generated");
        
        // Check that we have our share
        assert!(qtsp.my_key_shares.contains_key(&key_gen_id), "Should have our key share");
    }
    
    #[tokio::test]
    async fn test_signing() {
        let mut qtsp = QTSP::new("test_node");
        
        // Add authorized nodes
        qtsp.add_authorized_node("test_node".to_string());
        qtsp.add_authorized_node("node1".to_string());
        qtsp.add_authorized_node("node2".to_string());
        qtsp.add_authorized_node("node3".to_string());
        
        // Initialize key generation
        let key_gen_id = qtsp.init_key_generation().unwrap();
        
        // Sign a message
        let message = b"test message";
        let signature = qtsp.sign_message(message).unwrap();
        
        // Verify signature properties
        assert_eq!(signature.key_gen_id, key_gen_id, "Should use the correct key gen ID");
        assert_eq!(signature.signer_node_ids.len(), qtsp.config.threshold, 
                  "Should have threshold number of signers");
        assert!(signature.signer_node_ids.contains(&qtsp.node_id), 
               "Our node should be included in signers");
        
        // Verify the signature
        let verification = qtsp.verify_signature(message, &signature).unwrap();
        assert!(verification, "Signature should verify correctly");
    }
    
    #[tokio::test]
    async fn test_key_rotation() {
        let mut qtsp = QTSP::new("test_node");
        
        // Add authorized nodes
        qtsp.add_authorized_node("test_node".to_string());
        qtsp.add_authorized_node("node1".to_string());
        qtsp.add_authorized_node("node2".to_string());
        qtsp.add_authorized_node("node3".to_string());
        
        // Initialize first key generation
        let key_gen_id1 = qtsp.init_key_generation().unwrap();
        
        // Rotate keys - no longer async
        let key_gen_id2 = qtsp.rotate_keys().unwrap();
        
        // Check that we have two different key generations
        assert_ne!(key_gen_id1, key_gen_id2, "Should generate different key IDs");
        assert!(qtsp.key_gen_sessions.contains_key(&key_gen_id1), "First key gen should exist");
        assert!(qtsp.key_gen_sessions.contains_key(&key_gen_id2), "Second key gen should exist");
        
        // Check that we have shares for both
        assert!(qtsp.my_key_shares.contains_key(&key_gen_id1), "Should have share for first key gen");
        assert!(qtsp.my_key_shares.contains_key(&key_gen_id2), "Should have share for second key gen");
    }
} 