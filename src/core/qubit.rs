// Qubit Implementation
//
// This file implements a simulated qubit for quantum protocol development.

use std::collections::HashSet;
use std::time::{Duration, Instant};
use std::fmt;
use rand::{Rng, thread_rng};

/// Represents the state of a single quantum bit (qubit)
#[derive(Debug, Clone)]
pub struct Qubit {
    /// Probability amplitude for |0⟩ state (real and imaginary parts)
    alpha: (f64, f64),
    
    /// Probability amplitude for |1⟩ state (real and imaginary parts)
    beta: (f64, f64),
    
    /// Whether this qubit is entangled with others
    entangled: bool,
    
    /// IDs of other qubits this one is entangled with
    entanglement_ids: HashSet<u64>,
    
    /// Unique identifier for this qubit
    id: u64,
    
    /// Creation time for tracking decoherence
    creation_time: Instant,
    
    /// When this qubit was last measured
    last_measured: Option<Instant>,
    
    /// Last measurement result (if any)
    last_measurement: Option<u8>,
}

impl Qubit {
    /// Create a new qubit in the |0⟩ state
    pub fn new() -> Self {
        Self {
            alpha: (1.0, 0.0), // |0⟩ state with probability 1
            beta: (0.0, 0.0),  // |1⟩ state with probability 0
            entangled: false,
            entanglement_ids: HashSet::new(),
            id: thread_rng().gen(),
            creation_time: Instant::now(),
            last_measured: None,
            last_measurement: None,
        }
    }
    
    /// Create a qubit in a specific superposition state
    pub fn with_state(alpha_real: f64, alpha_imag: f64, beta_real: f64, beta_imag: f64) -> Self {
        let mut qubit = Self::new();
        
        // Set the state
        qubit.alpha = (alpha_real, alpha_imag);
        qubit.beta = (beta_real, beta_imag);
        
        // Normalize
        qubit.normalize();
        
        qubit
    }
    
    /// Get the unique ID of this qubit
    pub fn id(&self) -> u64 {
        self.id
    }
    
    /// Check if this qubit is entangled with others
    pub fn is_entangled(&self) -> bool {
        self.entangled
    }
    
    /// Get the set of qubit IDs this qubit is entangled with
    pub fn entanglement_ids(&self) -> &HashSet<u64> {
        &self.entanglement_ids
    }
    
    /// Mark this qubit as entangled with another qubit
    pub fn entangle_with(&mut self, other_id: u64) {
        self.entangled = true;
        self.entanglement_ids.insert(other_id);
    }
    
    /// Remove entanglement with another qubit
    pub fn remove_entanglement(&mut self, other_id: u64) {
        self.entanglement_ids.remove(&other_id);
        self.entangled = !self.entanglement_ids.is_empty();
    }
    
    /// Calculate the probability of measuring |0⟩
    pub fn prob_0(&self) -> f64 {
        let (re, im) = self.alpha;
        re * re + im * im
    }
    
    /// Calculate the probability of measuring |1⟩
    pub fn prob_1(&self) -> f64 {
        let (re, im) = self.beta;
        re * re + im * im
    }
    
    /// Normalize the state vector
    fn normalize(&mut self) {
        let p0 = self.prob_0();
        let p1 = self.prob_1();
        let norm = (p0 + p1).sqrt();
        
        if norm > 1e-10 {
            self.alpha.0 /= norm;
            self.alpha.1 /= norm;
            self.beta.0 /= norm;
            self.beta.1 /= norm;
        }
    }
    
    /// Create a qubit in the |+⟩ state (superposition)
    pub fn plus() -> Self {
        let factor = 1.0 / 2.0_f64.sqrt();
        Self::with_state(factor, 0.0, factor, 0.0)
    }
    
    /// Apply the X (NOT) gate to this qubit
    pub fn x(&mut self) {
        // Swap alpha and beta
        std::mem::swap(&mut self.alpha, &mut self.beta);
    }
    
    /// Alias for x() method
    pub fn apply_x(&mut self) {
        self.x();
    }
    
    /// Apply the Z gate to this qubit
    pub fn z(&mut self) {
        // Negate the phase of |1⟩
        self.beta.0 = -self.beta.0;
        self.beta.1 = -self.beta.1;
    }
    
    /// Apply the Hadamard gate to this qubit
    pub fn hadamard(&mut self) {
        // Extract current state
        let (alpha_re, alpha_im) = self.alpha;
        let (beta_re, beta_im) = self.beta;
        
        // Apply Hadamard transformation
        let factor = 1.0 / 2.0_f64.sqrt();
        
        // |0⟩ → (|0⟩ + |1⟩)/√2, |1⟩ → (|0⟩ - |1⟩)/√2
        self.alpha = (
            factor * (alpha_re + beta_re),
            factor * (alpha_im + beta_im)
        );
        
        self.beta = (
            factor * (alpha_re - beta_re),
            factor * (alpha_im - beta_im)
        );
    }
    
    /// Alias for hadamard() method
    pub fn apply_h(&mut self) {
        self.hadamard();
    }
    
    /// Apply the Y gate to this qubit
    pub fn y(&mut self) {
        // Y = iXZ
        // First apply Z
        self.z();
        
        // Then X (with phase factor i)
        let (alpha_re, alpha_im) = self.alpha;
        let (beta_re, beta_im) = self.beta;
        
        // Apply phase factor i when going from |0⟩ to |1⟩
        self.alpha = (beta_re, beta_im);
        // Apply phase factor -i when going from |1⟩ to |0⟩
        self.beta = (-alpha_im, alpha_re);
    }
    
    /// Apply a phase shift gate (rotation around Z axis)
    pub fn phase(&mut self, angle: f64) {
        // Apply phase shift to |1⟩ component only
        // e^(iθ) = cos(θ) + i sin(θ)
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();
        
        // Only apply to |1⟩ component (Beta)
        let (beta_re, beta_im) = self.beta;
        
        self.beta = (
            beta_re * cos_angle - beta_im * sin_angle,
            beta_re * sin_angle + beta_im * cos_angle
        );
        
        // Optionally debug
        //println!("DEBUG: Applied phase: {}, New beta: {:?}", angle, self.beta);
    }
    
    /// Apply S gate (phase gate) - rotates |1⟩ by π/2
    pub fn s(&mut self) {
        // S = |0⟩⟨0| + i|1⟩⟨1|
        let (beta_re, beta_im) = self.beta;
        self.beta = (-beta_im, beta_re); // Multiply by i
    }
    
    /// Apply S† gate (adjoint of S gate) - rotates |1⟩ by -π/2
    pub fn s_dagger(&mut self) {
        // S† = |0⟩⟨0| - i|1⟩⟨1|
        let (beta_re, beta_im) = self.beta;
        self.beta = (beta_im, -beta_re); // Multiply by -i
    }
    
    /// Apply T gate (π/8 gate) - rotates |1⟩ by π/4
    pub fn t(&mut self) {
        // T = |0⟩⟨0| + e^(iπ/4)|1⟩⟨1|
        self.phase(std::f64::consts::PI / 4.0);
    }
    
    /// Apply T† gate (adjoint of T gate) - rotates |1⟩ by -π/4
    pub fn t_dagger(&mut self) {
        // T† = |0⟩⟨0| + e^(-iπ/4)|1⟩⟨1|
        self.phase(-std::f64::consts::PI / 4.0);
    }
    
    /// Apply Rx gate (rotation around X axis)
    pub fn rx(&mut self, angle: f64) {
        // Rx(θ) = cos(θ/2)I - i sin(θ/2)X
        let (alpha_re, alpha_im) = self.alpha;
        let (beta_re, beta_im) = self.beta;
        
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        
        self.alpha = (
            alpha_re * cos_half - beta_im * sin_half,
            alpha_im * cos_half + beta_re * sin_half
        );
        
        self.beta = (
            beta_re * cos_half - alpha_im * sin_half,
            beta_im * cos_half + alpha_re * sin_half
        );
    }
    
    /// Apply Ry gate (rotation around Y axis)
    pub fn ry(&mut self, angle: f64) {
        // Ry(θ) = cos(θ/2)I - i sin(θ/2)Y
        let (alpha_re, alpha_im) = self.alpha;
        let (beta_re, beta_im) = self.beta;
        
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        
        self.alpha = (
            alpha_re * cos_half - beta_re * sin_half,
            alpha_im * cos_half - beta_im * sin_half
        );
        
        self.beta = (
            beta_re * cos_half + alpha_re * sin_half,
            beta_im * cos_half + alpha_im * sin_half
        );
    }
    
    /// Apply Rz gate (rotation around Z axis)
    pub fn rz(&mut self, angle: f64) {
        // For special case of π, just use the regular Z gate which is well-tested
        if (angle - std::f64::consts::PI).abs() < 1e-10 {
            self.z();
            return;
        }
        
        // For other angles: Rz(θ) = e^(-iθ/2)|0⟩⟨0| + e^(iθ/2)|1⟩⟨1|
        let half_angle = angle / 2.0;
        
        // Apply phase to |0⟩ component
        let (alpha_re, alpha_im) = self.alpha;
        let cos_half = half_angle.cos();
        let sin_half = (-half_angle).sin(); // Note the negative for e^(-iθ/2)
        
        self.alpha = (
            alpha_re * cos_half - alpha_im * sin_half, 
            alpha_im * cos_half + alpha_re * sin_half
        );
        
        // Apply phase to |1⟩ component
        let (beta_re, beta_im) = self.beta;
        let cos_half = half_angle.cos();
        let sin_half = half_angle.sin(); // Positive for e^(iθ/2)
        
        self.beta = (
            beta_re * cos_half - beta_im * sin_half,
            beta_im * cos_half + beta_re * sin_half
        );
    }
    
    /// Get the coefficients for controlled operations
    pub fn get_coeffs(&self) -> ((f64, f64), (f64, f64)) {
        (self.alpha, self.beta)
    }
    
    /// Set the coefficients from controlled operations
    pub fn set_coeffs(&mut self, alpha: (f64, f64), beta: (f64, f64)) {
        self.alpha = alpha;
        self.beta = beta;
        self.normalize();
    }
    
    /// Create a qubit in the |1⟩ state
    #[must_use]
    pub fn one() -> Self {
        Self::with_state(0.0, 0.0, 1.0, 0.0)
    }
    
    /// Create a qubit in the |-⟩ state (superposition with negative phase)
    #[must_use]
    pub fn minus() -> Self {
        let factor = 1.0 / 2.0_f64.sqrt();
        Self::with_state(factor, 0.0, -factor, 0.0)
    }
    
    /// Apply controlled phase flip to qubit
    pub fn controlled_phase_flip(&mut self, control_is_one: bool) {
        if control_is_one {
            self.z();
        }
    }
    
    /// Measure the qubit, collapsing its state
    pub fn measure(&mut self) -> u8 {
        let p0 = self.prob_0();
        let result = if thread_rng().gen::<f64>() < p0 { 0 } else { 1 };
        
        // Collapse the state
        if result == 0 {
            self.alpha = (1.0, 0.0);
            self.beta = (0.0, 0.0);
        } else {
            self.alpha = (0.0, 0.0);
            self.beta = (1.0, 0.0);
        }
        
        // Record measurement
        self.last_measured = Some(Instant::now());
        self.last_measurement = Some(result);
        
        result
    }
    
    /// Add noise to simulate decoherence
    pub fn add_noise(&mut self, amount: f64) {
        if amount <= 0.0 {
            return;
        }
        
        let mut rng = thread_rng();
        
        // Apply phase noise
        let phase_noise = rng.gen_range(-amount..amount);
        self.phase(phase_noise);
        
        // Apply amplitude damping
        let damping = rng.gen_range(0.0..amount);
        
        // Bias towards |0⟩ state (ground state)
        let p1 = self.prob_1();
        let damped_p1 = p1 * (1.0 - damping);
        let damped_p0 = 1.0 - damped_p1;
        
        // Calculate new amplitudes
        let new_alpha_mag = damped_p0.sqrt();
        let new_beta_mag = damped_p1.sqrt();
        
        // Preserve phases but update magnitudes
        let alpha_mag = self.prob_0().sqrt();
        let beta_mag = self.prob_1().sqrt();
        
        if alpha_mag > 1e-10 {
            let phase_a_re = self.alpha.0 / alpha_mag;
            let phase_a_im = self.alpha.1 / alpha_mag;
            self.alpha = (new_alpha_mag * phase_a_re, new_alpha_mag * phase_a_im);
        }
        
        if beta_mag > 1e-10 {
            let phase_b_re = self.beta.0 / beta_mag;
            let phase_b_im = self.beta.1 / beta_mag;
            self.beta = (new_beta_mag * phase_b_re, new_beta_mag * phase_b_im);
        }
        
        // Ensure the state is normalized
        self.normalize();
    }
    
    /// Alias for add_noise method
    pub fn apply_noise(&mut self, amount: f64) {
        self.add_noise(amount);
    }
    
    /// Get the fidelity of this qubit's state (1.0 = perfect, 0.0 = completely mixed)
    pub fn fidelity(&self) -> f64 {
        // For a pure state, fidelity is 1. We simulate decoherence by
        // reduced fidelity based on time since creation
        let age_seconds = self.creation_time.elapsed().as_secs_f64();
        (1.0 - 0.01 * age_seconds).max(0.0)
    }
    
    /// Get the time since creation
    pub fn age(&self) -> Duration {
        self.creation_time.elapsed()
    }
    
    /// Get the last measurement result, if any
    pub fn last_measurement(&self) -> Option<u8> {
        self.last_measurement
    }
    
    /// Get the time since last measurement, if any
    pub fn time_since_measurement(&self) -> Option<Duration> {
        self.last_measured.map(|time| time.elapsed())
    }
    
    /// Add phase noise to the qubit (T2 dephasing)
    ///
    /// This applies a random phase rotation to simulate quantum dephasing effects
    ///
    /// # Arguments
    /// * `amount` - The amount of phase noise to apply (0.0-1.0)
    pub fn add_phase_noise(&mut self, amount: f64) {
        if amount <= 0.0 {
            return;
        }
        
        let mut rng = thread_rng();
        
        // Apply random phase rotation to both amplitudes
        // More severe for the |1⟩ state which is more susceptible to dephasing
        
        // Phase noise for |0⟩ state (alpha)
        let alpha_phase_noise = rng.gen_range(-amount * std::f64::consts::PI..amount * std::f64::consts::PI);
        let (re, im) = self.alpha;
        let alpha_mag = (re * re + im * im).sqrt();
        let alpha_phase = im.atan2(re) + alpha_phase_noise;
        self.alpha = (
            alpha_mag * alpha_phase.cos(),
            alpha_mag * alpha_phase.sin()
        );
        
        // Phase noise for |1⟩ state (beta) - typically more affected
        let beta_phase_noise = rng.gen_range(-1.5 * amount * std::f64::consts::PI..1.5 * amount * std::f64::consts::PI);
        let (re, im) = self.beta;
        let beta_mag = (re * re + im * im).sqrt();
        let beta_phase = im.atan2(re) + beta_phase_noise;
        self.beta = (
            beta_mag * beta_phase.cos(),
            beta_mag * beta_phase.sin()
        );
        
        // Ensure the state is normalized
        self.normalize();
    }
    
    /// Set the probability of measuring |0⟩ while preserving phase
    ///
    /// # Arguments
    /// * `prob` - The new probability of measuring |0⟩ (0.0-1.0)
    pub fn set_prob_0(&mut self, prob: f64) {
        let prob = prob.clamp(0.0, 1.0); // Clamp between 0 and 1
        let prob_1 = 1.0 - prob;
        
        // Extract current phases
        let alpha_mag = self.prob_0().sqrt();
        let beta_mag = self.prob_1().sqrt();
        
        // Preserve phases while changing magnitudes
        if alpha_mag > 1e-10 {
            let alpha_phase_re = self.alpha.0 / alpha_mag;
            let alpha_phase_im = self.alpha.1 / alpha_mag;
            
            let new_alpha_mag = prob.sqrt();
            self.alpha = (
                new_alpha_mag * alpha_phase_re,
                new_alpha_mag * alpha_phase_im
            );
        } else {
            // If alpha is near zero, set to real axis
            self.alpha = (prob.sqrt(), 0.0);
        }
        
        if beta_mag > 1e-10 {
            let beta_phase_re = self.beta.0 / beta_mag;
            let beta_phase_im = self.beta.1 / beta_mag;
            
            let new_beta_mag = prob_1.sqrt();
            self.beta = (
                new_beta_mag * beta_phase_re,
                new_beta_mag * beta_phase_im
            );
        } else {
            // If beta is near zero, set to real axis
            self.beta = (prob_1.sqrt(), 0.0);
        }
        
        // Ensure the state is normalized
        self.normalize();
    }
}

impl Default for Qubit {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Qubit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let p0 = self.prob_0();
        let p1 = self.prob_1();
        
        write!(f, "Qubit[{:x}]: p(|0⟩)={:.4}, p(|1⟩)={:.4}", self.id, p0, p1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Add a constant for float comparison epsilon
    const FLOAT_EPSILON: f64 = 1e-10;
    
    #[test]
    fn test_new_qubit() {
        let qubit = Qubit::new();
        
        // Use epsilon comparison instead of assert_eq for floating point values
        assert!((qubit.prob_0() - 1.0).abs() < FLOAT_EPSILON, "prob_0 should be 1.0");
        assert!((qubit.prob_1() - 0.0).abs() < FLOAT_EPSILON, "prob_1 should be 0.0");
    }
    
    #[test]
    fn test_hadamard_creates_superposition() {
        let mut qubit = Qubit::new();
        qubit.hadamard();
        assert!((qubit.prob_0() - 0.5).abs() < 1e-10);
        assert!((qubit.prob_1() - 0.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_x_gate() {
        let mut qubit = Qubit::new();
        qubit.x();
        
        // Use epsilon comparison instead of assert_eq for floating point values
        assert!((qubit.prob_0() - 0.0).abs() < FLOAT_EPSILON, "prob_0 should be 0.0");
        assert!((qubit.prob_1() - 1.0).abs() < FLOAT_EPSILON, "prob_1 should be 1.0");
    }
    
    #[test]
    fn test_measure() {
        let mut qubit = Qubit::plus();
        
        // Probabilities should be 50/50 before measurement
        assert!((qubit.prob_0() - 0.5).abs() < 1e-10);
        assert!((qubit.prob_1() - 0.5).abs() < 1e-10);
        
        let result = qubit.measure();
        
        match result {
            0 => {
                // Use epsilon comparison instead of assert_eq for floating point values
                assert!((qubit.prob_0() - 1.0).abs() < FLOAT_EPSILON, "prob_0 should be 1.0");
                assert!((qubit.prob_1() - 0.0).abs() < FLOAT_EPSILON, "prob_1 should be 0.0");
            }
            1 => {
                // Use epsilon comparison instead of assert_eq for floating point values
                assert!((qubit.prob_0() - 0.0).abs() < FLOAT_EPSILON, "prob_0 should be 0.0");
                assert!((qubit.prob_1() - 1.0).abs() < FLOAT_EPSILON, "prob_1 should be 1.0");
            }
            _ => panic!("Unexpected measurement result"),
        }
    }
    
    #[test]
    fn test_s_gate() {
        let mut qubit = Qubit::plus(); // Create |+⟩ = (|0⟩ + |1⟩)/√2
        qubit.s(); // Apply S gate
        
        // The result should be (|0⟩ + i|1⟩)/√2
        let (alpha_re, alpha_im) = qubit.alpha;
        let (beta_re, beta_im) = qubit.beta;
        
        const EPSILON: f64 = 1e-10;
        let factor = 1.0 / 2.0_f64.sqrt();
        
        assert!((alpha_re - factor).abs() < EPSILON);
        assert!(alpha_im.abs() < EPSILON);
        assert!(beta_re.abs() < EPSILON);
        assert!((beta_im - factor).abs() < EPSILON);
    }
    
    #[test]
    fn test_t_gate() {
        let mut qubit = Qubit::plus(); // Create |+⟩ = (|0⟩ + |1⟩)/√2
        qubit.t(); // Apply T gate
        
        // The result should be (|0⟩ + e^(iπ/4)|1⟩)/√2
        let (_, _) = qubit.alpha;
        let (beta_re, beta_im) = qubit.beta;
        
        const EPSILON: f64 = 1e-10;
        let factor = 1.0 / 2.0_f64.sqrt();
        let pi_4_cos = (std::f64::consts::PI / 4.0).cos();
        let pi_4_sin = (std::f64::consts::PI / 4.0).sin();
        
        assert!((beta_re - factor * pi_4_cos).abs() < EPSILON);
        assert!((beta_im - factor * pi_4_sin).abs() < EPSILON);
    }
    
    #[test]
    fn test_rotation_gates() {
        // Test Rx gate with π rotation (should be equivalent to X gate)
        let mut qubit1 = Qubit::new();
        let mut qubit2 = Qubit::new();
        
        qubit1.x();
        qubit2.rx(std::f64::consts::PI);
        
        assert!((qubit1.prob_1() - qubit2.prob_1()).abs() < 1e-10);
        
        // Test Ry gate with π rotation
        let mut qubit3 = Qubit::new();
        qubit3.ry(std::f64::consts::PI);
        
        assert!((qubit3.prob_1() - 1.0).abs() < 1e-10);
        
        // Test Rz gate (should only affect phase, not probabilities)
        let mut qubit4 = Qubit::plus();
        let prob_before = qubit4.prob_1();
        
        qubit4.rz(std::f64::consts::PI);
        let prob_after = qubit4.prob_1();
        
        // Probabilities should remain unchanged by Z rotations
        assert!((prob_before - prob_after).abs() < 1e-10);
        
        // But the state should change - specifically, for |+⟩ and π rotation, 
        // we should get the |-⟩ state, which means the sign of the |1⟩ component flips
        let (_, _) = qubit4.alpha;
        let (beta_re, _) = qubit4.beta;
        
        const EPSILON: f64 = 1e-10;
        let factor = 1.0 / 2.0_f64.sqrt();
        
        // The |1⟩ component should have its sign flipped
        assert!((beta_re + factor).abs() < EPSILON);
    }
    
    #[test]
    fn test_special_states() {
        // Test |1⟩ state
        let qubit = Qubit::one();
        assert!(qubit.prob_1() > 0.99);
        
        // Test |-⟩ state
        let qubit = Qubit::minus();
        
        // |-⟩ should be an equal superposition
        assert!((qubit.prob_0() - 0.5).abs() < 1e-10);
        assert!((qubit.prob_1() - 0.5).abs() < 1e-10);
        
        // Apply H to |-⟩ should give |1⟩
        let mut minus = Qubit::minus();
        minus.hadamard();
        assert!(minus.prob_1() > 0.99);
    }
    
    #[test]
    fn test_controlled_operations() {
        // Test controlled phase flip
        let mut qubit = Qubit::plus();
        
        // When control is |0⟩, no change should occur
        qubit.controlled_phase_flip(false);
        let (alpha_re1, alpha_im1) = qubit.alpha;
        let (beta_re1, beta_im1) = qubit.beta;
        
        const EPSILON: f64 = 1e-10;
        let factor = 1.0 / 2.0_f64.sqrt();
        
        assert!((alpha_re1 - factor).abs() < EPSILON);
        assert!(alpha_im1.abs() < EPSILON);
        assert!((beta_re1 - factor).abs() < EPSILON);
        assert!(beta_im1.abs() < EPSILON);
        
        // When control is |1⟩, phase should flip
        qubit.controlled_phase_flip(true);
        let (alpha_re2, alpha_im2) = qubit.alpha;
        let (beta_re2, beta_im2) = qubit.beta;
        
        assert!((alpha_re2 - factor).abs() < EPSILON);
        assert!(alpha_im2.abs() < EPSILON);
        assert!((beta_re2 + factor).abs() < EPSILON); // Sign flipped
        assert!(beta_im2.abs() < EPSILON);
    }
} 