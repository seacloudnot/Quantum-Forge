use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use quantum_protocols::{
    consensus::{qpcp::QPCP, qtsp::QTSP, qvcp::QVCP},
    core::{QuantumRegister, QuantumState, Qubit},
    error_correction::{qcqp::QCQP, qecc::QECC, CorrectionCode},
    integration::qdap::{EncodingScheme, QDAP},
    security::qrng::QRNG,
};

fn quantum_core_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Quantum Core Operations");

    // Benchmark basic qubit operations
    group.bench_function("qubit_x_gate", |b| {
        b.iter(|| {
            let mut qubit = Qubit::new();
            qubit.x();
            black_box(qubit)
        })
    });

    group.bench_function("qubit_hadamard", |b| {
        b.iter(|| {
            let mut qubit = Qubit::new();
            qubit.hadamard();
            black_box(qubit)
        })
    });

    group.bench_function("qubit_measure", |b| {
        b.iter(|| {
            let mut qubit = Qubit::new();
            qubit.measure();
            black_box(qubit)
        })
    });

    // Benchmark quantum register operations with different sizes
    for size in [2, 4, 8, 16, 32].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("create_register", size), size, |b, &size| {
            b.iter(|| {
                let register = QuantumRegister::new(size);
                black_box(register)
            })
        });

        group.bench_with_input(BenchmarkId::new("hadamard_all", size), size, |b, &size| {
            b.iter(|| {
                let mut register = QuantumRegister::new(size);
                register.hadamard_all();
                black_box(register)
            })
        });

        group.bench_with_input(BenchmarkId::new("measure_all", size), size, |b, &size| {
            b.iter(|| {
                let mut register = QuantumRegister::new(size);
                register.hadamard_all(); // Create superposition first
                let results = register.measure_all();
                black_box(results)
            })
        });
    }

    // Quantum State benchmarks
    for size in [2, 4, 8, 16].iter() {
        group.bench_with_input(BenchmarkId::new("create_quantum_state", size), size, |b, &size| {
            b.iter(|| {
                let state = QuantumState::new(*size);
                black_box(state)
            })
        });
    }

    group.bench_function("create_bell_pair", |b| {
        b.iter(|| {
            let state = QuantumState::bell_pair();
            black_box(state)
        })
    });

    group.bench_function("create_ghz_state", |b| {
        b.iter(|| {
            let state = QuantumState::ghz(3);
            black_box(state)
        })
    });

    group.finish();
}

fn error_correction_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Error Correction");

    // QECC benchmarks
    group.bench_function("qecc_repetition_encode", |b| {
        b.iter(|| {
            let state = QuantumState::new(3);
            let mut qecc = QECC::new();
            qecc.set_state(state.clone());
            black_box(qecc.encode(CorrectionCode::Repetition))
        })
    });

    group.bench_function("qecc_detect_errors", |b| {
        b.iter(|| {
            let state = QuantumState::new(3);
            let mut qecc = QECC::new();
            qecc.set_state(state.clone());
            qecc.encode(CorrectionCode::Repetition);
            black_box(qecc.detect_errors())
        })
    });

    // QCQP benchmarks
    group.bench_function("qcqp_protect_classical_data", |b| {
        b.iter(|| {
            let mut qcqp = QCQP::new_default();
            let data = b"Test data for benchmarking";
            black_box(qcqp.protect_classical_data(data))
        })
    });

    group.finish();
}

fn consensus_protocol_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Consensus Protocols");

    // QPCP initialization and voting
    group.bench_function("qpcp_initialize", |b| {
        b.iter(|| {
            let qpcp = QPCP::new("test_node".to_string());
            black_box(qpcp)
        })
    });

    // QTSP key generation
    group.bench_function("qtsp_initialize", |b| {
        b.iter(|| {
            let qtsp = QTSP::new("test_node");
            black_box(qtsp)
        })
    });

    // QVCP initialization
    group.bench_function("qvcp_initialize", |b| {
        b.iter(|| {
            let qvcp = QVCP::new("test_node".to_string());
            black_box(qvcp)
        })
    });

    group.finish();
}

fn integration_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Integration Components");

    // QDAP encoding benchmarks with different schemes
    group.bench_function("qdap_amplitude_encoding", |b| {
        b.iter(|| {
            let mut qdap = QDAP::new_default();
            let data = b"Test data for QDAP benchmarking";
            black_box(qdap.encode_to_quantum(data, EncodingScheme::AmplitudeEncoding, None))
        })
    });

    group.bench_function("qdap_phase_encoding", |b| {
        b.iter(|| {
            let mut qdap = QDAP::new_default();
            let data = b"Test data for QDAP benchmarking";
            black_box(qdap.encode_to_quantum(data, EncodingScheme::PhaseEncoding, None))
        })
    });

    group.bench_function("qdap_binary_encoding", |b| {
        b.iter(|| {
            let mut qdap = QDAP::new_default();
            let data = b"Test";  // Smaller data for binary encoding (uses more qubits)
            black_box(qdap.encode_to_quantum(data, EncodingScheme::BinaryEncoding, None))
        })
    });

    group.finish();
}

fn security_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Security Operations");

    // QRNG benchmarks
    group.bench_function("qrng_generate_byte", |b| {
        b.iter(|| {
            let mut qrng = QRNG::new_default();
            black_box(qrng.generate_byte())
        })
    });

    for size in [16, 32, 64, 128, 256].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::new("qrng_generate_bytes", size), size, |b, &size| {
            b.iter(|| {
                let mut qrng = QRNG::new_default();
                black_box(qrng.generate_bytes(size))
            })
        });
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(50);
    targets = quantum_core_benchmarks, error_correction_benchmarks, 
              consensus_protocol_benchmarks, integration_benchmarks,
              security_benchmarks
);
criterion_main!(benches); 