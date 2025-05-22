use criterion::{black_box, criterion_group, criterion_main, Criterion};
use quantum_protocols::core::{QuantumState, Qubit};

fn qubit_operations_benchmark(c: &mut Criterion) {
    c.bench_function("qubit_hadamard", |b| {
        b.iter(|| {
            let mut qubit = Qubit::new();
            qubit.hadamard();
            black_box(qubit)
        })
    });
}

fn quantum_state_benchmark(c: &mut Criterion) {
    c.bench_function("create_quantum_state", |b| {
        b.iter(|| {
            let state = QuantumState::new(3);
            black_box(state)
        })
    });
}

criterion_group!(benches, qubit_operations_benchmark, quantum_state_benchmark);
criterion_main!(benches); 