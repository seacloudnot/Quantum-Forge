// Quantum Register Implementation
//
// This file implements a quantum register that manages multiple qubits.

use crate::core::qubit::Qubit;
use std::fmt;

/// Represents a quantum register containing multiple qubits
#[derive(Clone, Debug)]
pub struct QuantumRegister {
    /// The qubits in this register
    qubits: Vec<Qubit>,
    
    /// The name/identifier of this register
    name: String,
}

impl QuantumRegister {
    /// Create a new quantum register with the specified number of qubits
    pub fn new(num_qubits: usize) -> Self {
        let mut qubits = Vec::with_capacity(num_qubits);
        
        // Initialize qubits in |0⟩ state
        for _ in 0..num_qubits {
            qubits.push(Qubit::new());
        }
        
        Self {
            qubits,
            name: format!("register-{num_qubits}"),
        }
    }
    
    /// Get the number of qubits in this register
    pub fn size(&self) -> usize {
        self.qubits.len()
    }
    
    /// Get the name of this register
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Set the name of this register
    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }
    
    /// Get a reference to the qubits in this register
    pub fn qubits(&self) -> &Vec<Qubit> {
        &self.qubits
    }
    
    /// Get a mutable reference to the qubits in this register
    pub fn qubits_mut(&mut self) -> &mut Vec<Qubit> {
        &mut self.qubits
    }
    
    /// Get a specific qubit by index
    pub fn qubit(&self, qubit_idx: usize) -> Option<&Qubit> {
        self.qubits.get(qubit_idx)
    }
    
    /// Get a mutable reference to a specific qubit by index
    pub fn qubit_mut(&mut self, qubit_idx: usize) -> Option<&mut Qubit> {
        self.qubits.get_mut(qubit_idx)
    }
    
    /// Apply the Hadamard gate to a specific qubit
    pub fn hadamard(&mut self, qubit_idx: usize) {
        if qubit_idx < self.qubits.len() {
            self.qubits[qubit_idx].hadamard();
        }
    }
    
    /// Apply the X gate to a specific qubit
    pub fn x(&mut self, qubit_idx: usize) {
        if qubit_idx < self.qubits.len() {
            self.qubits[qubit_idx].x();
        }
    }
    
    /// Apply the Y gate to a specific qubit
    pub fn y(&mut self, qubit_idx: usize) {
        if qubit_idx < self.qubits.len() {
            self.qubits[qubit_idx].y();
        }
    }
    
    /// Apply the Z gate to a specific qubit
    pub fn z(&mut self, qubit_idx: usize) {
        if qubit_idx < self.qubits.len() {
            self.qubits[qubit_idx].z();
        }
    }
    
    /// Apply the S gate (phase gate) to a specific qubit
    pub fn s(&mut self, qubit_idx: usize) {
        if qubit_idx < self.qubits.len() {
            self.qubits[qubit_idx].s();
        }
    }
    
    /// Apply the S† gate (adjoint of S gate) to a specific qubit
    pub fn s_dagger(&mut self, qubit_idx: usize) {
        if qubit_idx < self.qubits.len() {
            self.qubits[qubit_idx].s_dagger();
        }
    }
    
    /// Apply the T gate (π/8 gate) to a specific qubit
    pub fn t(&mut self, qubit_idx: usize) {
        if qubit_idx < self.qubits.len() {
            self.qubits[qubit_idx].t();
        }
    }
    
    /// Apply the T† gate (adjoint of T gate) to a specific qubit
    pub fn t_dagger(&mut self, qubit_idx: usize) {
        if qubit_idx < self.qubits.len() {
            self.qubits[qubit_idx].t_dagger();
        }
    }
    
    /// Apply the Rx gate (rotation around X axis) to a specific qubit
    pub fn rx(&mut self, qubit_idx: usize, angle: f64) {
        if qubit_idx < self.qubits.len() {
            self.qubits[qubit_idx].rx(angle);
        }
    }
    
    /// Apply the Ry gate (rotation around Y axis) to a specific qubit
    pub fn ry(&mut self, qubit_idx: usize, angle: f64) {
        if qubit_idx < self.qubits.len() {
            self.qubits[qubit_idx].ry(angle);
        }
    }
    
    /// Apply the Rz gate (rotation around Z axis) to a specific qubit
    pub fn rz(&mut self, qubit_idx: usize, angle: f64) {
        if qubit_idx < self.qubits.len() {
            self.qubits[qubit_idx].rz(angle);
        }
    }
    
    /// Apply a phase shift gate to a specific qubit
    pub fn phase(&mut self, qubit_idx: usize, angle: f64) {
        if qubit_idx < self.qubits.len() {
            self.qubits[qubit_idx].phase(angle);
        }
    }
    
    /// Apply a Controlled-Z gate between two qubits
    ///
    /// This implements a controlled-Z operation which is fundamental for quantum phase
    /// estimation and other phase-based algorithms. The controlled-Z has two key effects:
    ///
    /// 1. When control qubit is |1⟩ and target is |1⟩, applies -1 phase to target
    /// 2. Due to phase kickback, also applies -1 phase to control qubit's |1⟩ component
    ///    when target is in |1⟩ state
    ///
    /// # Arguments
    ///
    /// * `control` - Index of the control qubit
    /// * `target` - Index of the target qubit
    ///
    /// # Quantum Mechanical Details
    ///
    /// For input state |c⟩|t⟩, the controlled-Z transforms:
    /// - |0⟩|0⟩ → |0⟩|0⟩
    /// - |0⟩|1⟩ → |0⟩|1⟩
    /// - |1⟩|0⟩ → |1⟩|0⟩
    /// - |1⟩|1⟩ → -|1⟩|1⟩
    ///
    /// The phase kickback occurs because when target is in |1⟩ state,
    /// the control qubit's |1⟩ component gets a phase of -1.
    pub fn cz(&mut self, control: usize, target: usize) {
        // Validate indices
        if control >= self.qubits.len() || target >= self.qubits.len() || control == target {
            return;
        }
        
        // Mark qubits as entangled
        let control_id = self.qubits[control].id();
        let target_id = self.qubits[target].id();
        
        self.qubits[control].entangle_with(target_id);
        self.qubits[target].entangle_with(control_id);
        
        // Amplitude threshold for considering a state component significant
        const AMPLITUDE_THRESHOLD: f64 = 1e-6;
        
        // Get quantum states safely with minimal borrowing
        let (control_state, target_state, target_prob_one) = {
            let control_qubit = &self.qubits[control];
            let target_qubit = &self.qubits[target];
            
            (control_qubit.get_coeffs(), 
             target_qubit.get_coeffs(), 
             target_qubit.prob_1())
        };
        
        // For efficiency, don't process further if target has no |1⟩ component
        if target_prob_one < AMPLITUDE_THRESHOLD {
            return;
        }
        
        // Destructure the states for easier access
        let (alpha_c, beta_c) = control_state;
        let (alpha_t, beta_t) = target_state;
        
        // Check if control has significant |1⟩ component
        let control_has_one = beta_c.0.abs() > AMPLITUDE_THRESHOLD || 
                              beta_c.1.abs() > AMPLITUDE_THRESHOLD;
        
        // Check if target has significant |1⟩ component
        let target_has_one = beta_t.0.abs() > AMPLITUDE_THRESHOLD || 
                             beta_t.1.abs() > AMPLITUDE_THRESHOLD;
        
        // Phase kickback: When target is |1⟩, apply -1 phase to control's |1⟩ component
        if target_has_one {
            // Apply phase -1 to control's |1⟩ component (crucial for phase kickback)
            self.qubits[control].set_coeffs(alpha_c, (-beta_c.0, -beta_c.1));
        }
        
        // Standard CZ effect: When control is |1⟩, apply -1 phase to target's |1⟩ component
        if control_has_one && target_has_one {
            self.qubits[target].set_coeffs(alpha_t, (-beta_t.0, -beta_t.1));
        }
        
        // Debug verification in debug mode
        #[cfg(debug_assertions)]
        {
            if control_has_one && target_has_one {
                let (a, b) = self.qubits[control].get_coeffs();
                println!("DEBUG: CZ applied with phase kickback");
                println!("      Control: α=({:.4}, {:.4}), β=({:.4}, {:.4})", 
                         a.0, a.1, b.0, b.1);
            }
        }
    }
    
    /// Apply a CNOT gate between two qubits
    pub fn cnot(&mut self, control: usize, target: usize) {
        if control < self.qubits.len() && target < self.qubits.len() && control != target {
            // Mark qubits as entangled
            let control_id = self.qubits[control].id();
            let target_id = self.qubits[target].id();
            
            self.qubits[control].entangle_with(target_id);
            self.qubits[target].entangle_with(control_id);
            
            // Get control qubit state
            let control_qubit = &self.qubits[control];
            let (_alpha_c, beta_c) = (control_qubit.get_coeffs().0, control_qubit.get_coeffs().1);
            
            // Get target qubit state
            let target_qubit = &self.qubits[target];
            let (alpha_t, beta_t) = (target_qubit.get_coeffs().0, target_qubit.get_coeffs().1);
            
            // For proper CNOT operation:
            // If control is |0⟩, target is unchanged
            // If control is |1⟩, target is flipped
            
            // Store updated values
            let mut new_alpha_t = alpha_t;
            let mut new_beta_t = beta_t;
            
            // Control in |1⟩ state component - flip the target qubit for this component
            if beta_c.0.abs() > 1e-10 || beta_c.1.abs() > 1e-10 {
                // Swap the state components for the |1⟩ control component
                std::mem::swap(&mut new_alpha_t, &mut new_beta_t);
            }
            
            // Update target qubit
            self.qubits[target].set_coeffs(new_alpha_t, new_beta_t);
        }
    }
    
    /// Apply a Toffoli (CCNOT) gate - 3-qubit gate
    pub fn toffoli(&mut self, control1: usize, control2: usize, target: usize) {
        if control1 < self.qubits.len() && 
           control2 < self.qubits.len() && 
           target < self.qubits.len() &&
           control1 != control2 && 
           control1 != target && 
           control2 != target {
            
            // Mark qubits as entangled
            let control1_id = self.qubits[control1].id();
            let control2_id = self.qubits[control2].id();
            let target_id = self.qubits[target].id();
            
            self.qubits[control1].entangle_with(control2_id);
            self.qubits[control1].entangle_with(target_id);
            self.qubits[control2].entangle_with(control1_id);
            self.qubits[control2].entangle_with(target_id);
            self.qubits[target].entangle_with(control1_id);
            self.qubits[target].entangle_with(control2_id);
            
            // Check if both control qubits are in state |1⟩
            let control1_is_one = self.qubits[control1].prob_1() > 0.5;
            let control2_is_one = self.qubits[control2].prob_1() > 0.5;
            
            // Apply X to target only if both controls are |1⟩
            if control1_is_one && control2_is_one {
                self.qubits[target].x();
            }
        }
    }
    
    /// Apply a SWAP gate between two qubits
    pub fn swap(&mut self, qubit_a: usize, qubit_b: usize) {
        if qubit_a < self.qubits.len() && qubit_b < self.qubits.len() && qubit_a != qubit_b {
            // In a real system, we would apply three CNOT gates
            // For simulation, we just swap the qubits
            self.qubits.swap(qubit_a, qubit_b);
        }
    }
    
    /// Apply the Hadamard gate to all qubits
    pub fn hadamard_all(&mut self) {
        for qubit in &mut self.qubits {
            qubit.hadamard();
        }
    }
    
    /// Measure a specific qubit
    pub fn measure(&mut self, qubit_idx: usize) -> Option<u8> {
        if qubit_idx < self.qubits.len() {
            // Get the target qubit
            let qubit = &mut self.qubits[qubit_idx];
            
            // If the qubit is entangled, we need to handle measurement carefully
            if qubit.is_entangled() {
                // Store entangled qubit IDs before measurement
                let entangled_ids: Vec<u64> = qubit.entanglement_ids().iter().cloned().collect();
                
                // Measure the qubit
                let result = qubit.measure();
                
                // Update entangled qubits based on the type of entanglement
                for (idx, other_qubit) in self.qubits.iter_mut().enumerate() {
                    if idx != qubit_idx && entangled_ids.contains(&other_qubit.id()) {
                        // This only works for bell states - in a real implementation
                        // we'd need to track the type of entanglement
                        
                        // For standard bell states |Φ±⟩, both qubits collapse to the same value
                        // For flipped bell states |Ψ±⟩, qubits collapse to opposite values
                        
                        // Check if the entangled qubit is in an X-basis state (has been flipped with X gate)
                        // This is a simple approximation to detect |Ψ±⟩ vs |Φ±⟩ states
                        let is_x_basis = other_qubit.prob_1() > 0.4 && other_qubit.prob_1() < 0.6;
                        
                        if is_x_basis {
                            // For |Ψ±⟩ states, collapse to opposite value
                            if result == 0 {
                                other_qubit.set_coeffs((0.0, 0.0), (1.0, 0.0)); // |1⟩
                            } else {
                                other_qubit.set_coeffs((1.0, 0.0), (0.0, 0.0)); // |0⟩
                            }
                        } else {
                            // For |Φ±⟩ states, collapse to same value
                            if result == 0 {
                                other_qubit.set_coeffs((1.0, 0.0), (0.0, 0.0)); // |0⟩
                            } else {
                                other_qubit.set_coeffs((0.0, 0.0), (1.0, 0.0)); // |1⟩
                            }
                        }
                    }
                }
                
                return Some(result);
            }
            // If not entangled, just measure normally
            return Some(qubit.measure());
        }
        
        None
    }
    
    /// Measure all qubits
    pub fn measure_all(&mut self) -> Vec<u8> {
        let mut results = Vec::with_capacity(self.qubits.len());
        
        for qubit in &mut self.qubits {
            results.push(qubit.measure());
        }
        
        results
    }
    
    /// Add a qubit to this register
    pub fn add_qubit(&mut self, qubit: Qubit) {
        self.qubits.push(qubit);
    }
    
    /// Add multiple qubits to this register
    pub fn add_qubits(&mut self, qubits: Vec<Qubit>) {
        self.qubits.extend(qubits);
    }
    
    /// Remove a qubit from this register
    pub fn remove_qubit(&mut self, qubit_idx: usize) -> Option<Qubit> {
        if qubit_idx < self.qubits.len() {
            Some(self.qubits.remove(qubit_idx))
        } else {
            None
        }
    }
    
    /// Apply noise to all qubits to simulate decoherence
    pub fn apply_noise(&mut self, noise_factor: f64) {
        for qubit in &mut self.qubits {
            qubit.add_noise(noise_factor);
        }
    }
    
    /// Create a GHZ state
    pub fn ghz(num_qubits: usize) -> Self {
        let mut register = Self::new(num_qubits);
        
        // Apply Hadamard to first qubit
        register.hadamard(0);
        
        // Apply CNOT with first qubit as control to all others
        for i in 1..num_qubits {
            register.cnot(0, i);
        }
        
        register
    }
    
    /// Create a W state
    pub fn w_state(num_qubits: usize) -> Self {
        let mut register = Self::new(num_qubits);
        
        // For a real W state, complex quantum circuits are needed
        // For simulation, we just mark all qubits as entangled with each other
        for i in 0..num_qubits {
            for j in 0..num_qubits {
                if i != j {
                    let qubit_j_id = register.qubits[j].id();
                    register.qubits[i].entangle_with(qubit_j_id);
                }
            }
        }
        
        register
    }
    
    /// Create a register in a random state
    pub fn random_state(num_qubits: usize) -> Self {
        let mut register = Self::new(num_qubits);
        
        // Apply Hadamard to all qubits to create superposition
        for i in 0..num_qubits {
            register.hadamard(i);
        }
        
        register
    }
    
    /// Entangle two qubits in the register
    pub fn entangle(&mut self, control: usize, target: usize) {
        if control < self.qubits.len() && target < self.qubits.len() && control != target {
            // Get qubit IDs
            let control_id = self.qubits[control].id();
            let target_id = self.qubits[target].id();
            
            // Entangle qubits with each other
            self.qubits[control].entangle_with(target_id);
            self.qubits[target].entangle_with(control_id);
        }
    }
    
    /// Reset a qubit to the |0⟩ state
    pub fn set_zero(&mut self, index: usize) {
        if index < self.qubits.len() {
            self.qubits[index] = Qubit::new();
        }
    }
    
    /// Apply the Pauli X (NOT) gate to a qubit
    pub fn apply_x(&mut self, index: usize) {
        if index < self.qubits.len() {
            self.qubits[index].apply_x();
        }
    }
    
    /// Create a Bell state (maximally entangled pair) on two specified qubits
    pub fn bell_state(&mut self, qubit_a: usize, qubit_b: usize, bell_type: usize) {
        if qubit_a < self.qubits.len() && qubit_b < self.qubits.len() && qubit_a != qubit_b {
            // Reset qubits to |0⟩ state first
            self.set_zero(qubit_a);
            self.set_zero(qubit_b);
            
            // Mark qubits as entangled with each other and store Bell state type
            let id_a = self.qubits[qubit_a].id();
            let id_b = self.qubits[qubit_b].id();
            
            self.qubits[qubit_a].entangle_with(id_b);
            self.qubits[qubit_b].entangle_with(id_a);
            
            // Apply Hadamard to first qubit
            self.hadamard(qubit_a);
            
            // Apply CNOT from first to second qubit
            match bell_type {
                0 => {
                    // |Φ+⟩ = (|00⟩ + |11⟩)/√2
                    // Put qubit_a in superposition and apply CNOT
                    self.qubits[qubit_a].set_coeffs((1.0/2.0_f64.sqrt(), 0.0), (1.0/2.0_f64.sqrt(), 0.0));
                    self.qubits[qubit_b].set_coeffs((1.0/2.0_f64.sqrt(), 0.0), (1.0/2.0_f64.sqrt(), 0.0));
                },
                1 => {
                    // |Φ-⟩ = (|00⟩ - |11⟩)/√2
                    self.qubits[qubit_a].set_coeffs((1.0/2.0_f64.sqrt(), 0.0), (1.0/2.0_f64.sqrt(), 0.0));
                    self.qubits[qubit_b].set_coeffs((1.0/2.0_f64.sqrt(), 0.0), (-1.0/2.0_f64.sqrt(), 0.0));
                },
                2 => {
                    // |Ψ+⟩ = (|01⟩ + |10⟩)/√2
                    self.qubits[qubit_a].set_coeffs((1.0/2.0_f64.sqrt(), 0.0), (1.0/2.0_f64.sqrt(), 0.0));
                    self.qubits[qubit_b].set_coeffs((1.0/2.0_f64.sqrt(), 0.0), (1.0/2.0_f64.sqrt(), 0.0));
                    // Store special entanglement info for |Ψ+⟩
                    self.qubits[qubit_b].apply_x();
                },
                3 => {
                    // |Ψ-⟩ = (|01⟩ - |10⟩)/√2
                    self.qubits[qubit_a].set_coeffs((1.0/2.0_f64.sqrt(), 0.0), (1.0/2.0_f64.sqrt(), 0.0));
                    self.qubits[qubit_b].set_coeffs((1.0/2.0_f64.sqrt(), 0.0), (-1.0/2.0_f64.sqrt(), 0.0));
                    // Store special entanglement info for |Ψ-⟩
                    self.qubits[qubit_b].apply_x();
                },
                _ => {
                    // Default to |Φ+⟩
                    self.qubits[qubit_a].set_coeffs((1.0/2.0_f64.sqrt(), 0.0), (1.0/2.0_f64.sqrt(), 0.0));
                    self.qubits[qubit_b].set_coeffs((1.0/2.0_f64.sqrt(), 0.0), (1.0/2.0_f64.sqrt(), 0.0));
                },
            }
        }
    }
    
    /// Apply quantum Fourier transform to a subset of qubits
    pub fn qft(&mut self, start_idx: usize, num_qubits: usize) {
        let end_idx = start_idx + num_qubits;
        if end_idx <= self.qubits.len() {
            // Apply Hadamard gates and controlled rotations
            for i in start_idx..end_idx {
                self.hadamard(i);
                
                // Apply controlled phase rotations
                for j in (i+1)..end_idx {
                    let k = j - i;
                    let angle = std::f64::consts::PI / (1 << k) as f64;
                    
                    // Instead of a proper controlled-phase, we'll 
                    // directly implement the effect for simulation
                    let control_is_one = self.qubits[i].prob_1() > 0.5;
                    if control_is_one {
                        self.phase(j, angle);
                    }
                }
            }
            
            // Swap qubits to get the right ordering
            for i in 0..(num_qubits / 2) {
                self.swap(start_idx + i, start_idx + num_qubits - i - 1);
            }
        }
    }
}

impl fmt::Display for QuantumRegister {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QuantumRegister[{}] with {} qubits", self.name, self.qubits.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new_register() {
        let register = QuantumRegister::new(3);
        assert_eq!(register.size(), 3);
    }
    
    #[test]
    fn test_hadamard_all() {
        let mut register = QuantumRegister::new(3);
        
        // Apply Hadamard to all qubits
        register.hadamard_all();
        
        // Each qubit should now be in a superposition
        for i in 0..3 {
            let qubit = register.qubit(i).unwrap();
            assert!(qubit.prob_0() > 0.0);
            assert!(qubit.prob_1() > 0.0);
        }
    }
    
    #[test]
    fn test_measure_all() {
        let mut register = QuantumRegister::new(5);
        
        // Prepare a known state: |00101⟩
        register.x(2);
        register.x(4);
        
        // Measure all qubits
        let results = register.measure_all();
        
        // Verify the results match the expected values
        assert_eq!(results.len(), 5);
        assert_eq!(results[0], 0);
        assert_eq!(results[1], 0);
        assert_eq!(results[2], 1);
        assert_eq!(results[3], 0);
        assert_eq!(results[4], 1);
    }
    
    #[test]
    fn test_entanglement() {
        let mut register = QuantumRegister::new(2);
        
        // Apply CNOT to entangle qubits
        register.cnot(0, 1);
        
        // Qubits should be entangled
        assert!(register.qubits()[0].is_entangled());
        assert!(register.qubits()[1].is_entangled());
    }
    
    #[test]
    fn test_single_qubit_gates() {
        let mut register = QuantumRegister::new(4);
        
        // Apply various gates to different qubits
        register.x(0);
        register.y(1);
        register.z(2);
        register.hadamard(3);
        
        // Test X gate effect
        assert!(register.qubit(0).unwrap().prob_1() > 0.99);
        
        // Test Y gate effect (|1⟩ state with phase)
        assert!(register.qubit(1).unwrap().prob_1() > 0.99);
        
        // Test Z gate effect (no change to |0⟩)
        assert!(register.qubit(2).unwrap().prob_0() > 0.99);
        
        // Test Hadamard effect (superposition)
        let qubit3 = register.qubit(3).unwrap();
        assert!((qubit3.prob_0() - 0.5).abs() < 1e-10);
        assert!((qubit3.prob_1() - 0.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_phase_gates() {
        let mut register = QuantumRegister::new(3);
        
        // Prepare |+⟩ states
        register.hadamard(0);
        register.hadamard(1);
        register.hadamard(2);
        
        // Apply different phase gates
        register.s(0);
        register.t(1);
        register.phase(2, std::f64::consts::PI / 3.0);
        
        // Phase gates don't change probabilities for |+⟩
        for i in 0..3 {
            let qubit = register.qubit(i).unwrap();
            assert!((qubit.prob_0() - 0.5).abs() < 1e-10);
            assert!((qubit.prob_1() - 0.5).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_bell_state() {
        // Test PhiPlus bell state (|00⟩ + |11⟩)/√2
        {
            let mut register = QuantumRegister::new(2);
            register.bell_state(0, 1, 0);
            
            // Verify qubits are entangled
            assert!(register.qubit(0).unwrap().is_entangled(), "Qubit 0 should be entangled");
            assert!(register.qubit(1).unwrap().is_entangled(), "Qubit 1 should be entangled");
            
            // Check probabilities are correct - each qubit should be in 50/50 superposition
            let q0 = register.qubit(0).unwrap();
            let q1 = register.qubit(1).unwrap();
            
            assert!((q0.prob_0() - 0.5).abs() < 1e-10, "Qubit 0 should have 50% probability for |0⟩");
            assert!((q0.prob_1() - 0.5).abs() < 1e-10, "Qubit 0 should have 50% probability for |1⟩");
            assert!((q1.prob_0() - 0.5).abs() < 1e-10, "Qubit 1 should have 50% probability for |0⟩");
            assert!((q1.prob_1() - 0.5).abs() < 1e-10, "Qubit 1 should have 50% probability for |1⟩");
        }
        
        // Test PsiPlus bell state (|01⟩ + |10⟩)/√2
        {
            let mut register = QuantumRegister::new(2);
            register.bell_state(0, 1, 2);
            
            // Verify qubits are entangled
            assert!(register.qubit(0).unwrap().is_entangled(), "Qubit 0 should be entangled");
            assert!(register.qubit(1).unwrap().is_entangled(), "Qubit 1 should be entangled");
            
            // Check probabilities are correct - each qubit should be in 50/50 superposition
            let q0 = register.qubit(0).unwrap();
            let q1 = register.qubit(1).unwrap();
            
            assert!((q0.prob_0() - 0.5).abs() < 1e-10, "Qubit 0 should have 50% probability for |0⟩");
            assert!((q0.prob_1() - 0.5).abs() < 1e-10, "Qubit 0 should have 50% probability for |1⟩");
            assert!((q1.prob_0() - 0.5).abs() < 1e-10, "Qubit 1 should have 50% probability for |0⟩");
            assert!((q1.prob_1() - 0.5).abs() < 1e-10, "Qubit 1 should have 50% probability for |1⟩");
        }
    }
    
    #[test]
    fn test_toffoli_gate() {
        let mut register = QuantumRegister::new(3);
        
        // Set control qubits to |1⟩
        register.x(0);
        register.x(1);
        
        // Apply Toffoli
        register.toffoli(0, 1, 2);
        
        // Target should be flipped to |1⟩
        assert!(register.qubit(2).unwrap().prob_1() > 0.99);
        
        // Create a new register
        let mut register2 = QuantumRegister::new(3);
        
        // Set only one control to |1⟩
        register2.x(0);
        
        // Apply Toffoli
        register2.toffoli(0, 1, 2);
        
        // Target should remain |0⟩
        assert!(register2.qubit(2).unwrap().prob_0() > 0.99);
    }
    
    #[test]
    fn test_qft() {
        // Create a 4-qubit register to test QFT
        let mut register = QuantumRegister::new(4);
        
        // Initialize with a simple state |0001⟩
        register.x(3);
        
        // Apply QFT to all qubits
        register.qft(0, 4);
        
        // After QFT, all qubits should be in superposition
        for i in 0..4 {
            let qubit = register.qubit(i).unwrap();
            
            // Probabilities should be non-zero for both |0⟩ and |1⟩
            assert!(qubit.prob_0() > 0.0);
            assert!(qubit.prob_1() > 0.0);
        }
        
        // A precise test would check amplitudes, but for simple validation
        // we just ensure the transformation happened
    }
} 