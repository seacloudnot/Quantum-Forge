impl QuantumRegister {
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
} 