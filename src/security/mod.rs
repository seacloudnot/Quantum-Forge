// Security Module
//
// This module provides quantum-resistant cryptography and security protocols.

/// Enumeration of supported quantum-resistant cryptography algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantumResistantAlgorithm {
    /// Lattice-based cryptography (NTRU)
    NTRU,
    
    /// Hash-based cryptography (SPHINCS+)
    SPHINCS,
    
    /// Code-based cryptography (`McEliece`)
    McEliece,
    
    /// Multivariate cryptography
    Multivariate,
    
    /// Isogeny-based cryptography (SIKE)
    SIKE,
}

/// Trait for quantum key distribution
pub trait QuantumKeyDistribution {
    /// Initialize a session with a peer
    ///
    /// # Errors
    ///
    /// Returns an error if session initialization fails
    fn init_session(&mut self, peer_id: &str) -> Result<String, String>;
    
    /// Exchange qubits with a peer
    ///
    /// # Errors
    ///
    /// Returns an error if qubit exchange fails
    fn exchange_qubits(&mut self, session_id: &str, count: usize) -> Result<Vec<u8>, String>;
    
    /// Measure qubits in specific basis
    ///
    /// # Errors
    ///
    /// Returns an error if measurement fails
    fn measure_qubits(&mut self, session_id: &str, basis: &[u8]) -> Result<Vec<u8>, String>;
    
    /// Compare bases with peer
    ///
    /// # Errors
    ///
    /// Returns an error if comparison fails
    fn compare_bases(&mut self, session_id: &str, peer_bases: &[u8]) -> Result<Vec<usize>, String>;
    
    /// Derive shared secret key
    ///
    /// # Errors
    ///
    /// Returns an error if key derivation fails
    fn derive_key(&mut self, session_id: &str, size: usize) -> Result<Vec<u8>, String>;
}

/// Trait for post-quantum cryptography
pub trait PostQuantumCrypto {
    /// Generate a quantum-resistant keypair
    ///
    /// # Errors
    ///
    /// Returns an error if keypair generation fails
    fn generate_keypair(&self, algorithm: QuantumResistantAlgorithm) -> Result<(Vec<u8>, Vec<u8>), String>;
    
    /// Sign data with a private key
    ///
    /// # Errors
    ///
    /// Returns an error if signing fails
    fn sign(&self, private_key: &[u8], data: &[u8]) -> Result<Vec<u8>, String>;
    
    /// Verify a signature
    ///
    /// # Errors
    ///
    /// Returns an error if verification fails
    fn verify(&self, public_key: &[u8], data: &[u8], signature: &[u8]) -> Result<bool, String>;
    
    /// Encrypt data with a public key
    ///
    /// # Errors
    ///
    /// Returns an error if encryption fails
    fn encrypt(&self, public_key: &[u8], data: &[u8]) -> Result<Vec<u8>, String>;
    
    /// Decrypt data with a private key
    ///
    /// # Errors
    ///
    /// Returns an error if decryption fails
    fn decrypt(&self, private_key: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>, String>;
}

pub mod qkd;
pub mod qrng;
pub mod qspp;
pub mod pqc;
pub mod expose_qkd_methods;

pub use qkd::QKD;
pub use qrng::QRNG;
pub use qspp::QSPP;
pub use pqc::PQC; 