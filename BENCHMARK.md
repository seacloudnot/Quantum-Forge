# Quantum Protocols Benchmarking & Performance Profiling

This document provides guidance on running benchmarks and analyzing performance of the Quantum Protocols project.

## Benchmark Components

The project includes two complementary benchmark systems:

1. **Criterion Benchmarks**: Standard Rust benchmarking library for statistical analysis of performance.
2. **Custom Performance Profiling**: A comprehensive profiling system that captures execution time, memory usage, and provides visualization.

## Benchmark Categories

The benchmarks cover all major components of the Quantum Protocols project:

- **Quantum Core Operations**: Basic qubit operations, register manipulations, state transformations
- **Error Correction**: QECC, QCQP, and other error correction protocols
- **Consensus Protocols**: QPCP, QTSP, QVCP implementations
- **Integration Components**: QDAP and other integration protocols
- **Security Operations**: QRNG and other security features

## Running Benchmarks

### Quick Start

#### Windows (PowerShell)

```powershell
.\run_all_benchmarks.ps1
```

#### Linux/macOS

```bash
./run_all_benchmarks.sh
```

This will execute all benchmarks and generate HTML dashboard and CSV reports in the `benchmark_results` directory.

### Individual Benchmark Components

To run specific benchmark components:

#### Standard Criterion Benchmarks

```bash
cargo bench --bench protocol_benchmarks
```

#### Comprehensive Custom Benchmarks

```bash
cargo run --release --bin run_benchmarks
```

#### Memory Profiling (if configured)

```bash
cargo run --release --bin run_benchmarks --features memory_profiling
```

## Benchmark Results

After running benchmarks, you can find the results in:

1. **Performance Dashboard**: `benchmark_results/performance_dashboard.html`
2. **CSV Data**: `benchmark_results/performance_data.csv`
3. **Criterion Reports**: `target/criterion/`

## Using the Performance Profiler in Your Code

The profiling system can be used in your own code to measure performance. Here's how:

### Basic Profiling

```rust
use quantum_protocols::benchmark::{get_profiler, OperationType};

fn my_function() {
    let profiler = get_profiler();
    
    // Profile a specific operation
    profiler.profile("my_operation", OperationType::QubitOperation, || {
        // Your code to profile here
        // The return value will be passed through
    });
}
```

### Using the Profile Guard

For automatic profiling of a code scope:

```rust
use quantum_protocols::benchmark::{get_profiler, OperationType, ProfileGuard};

fn my_function() {
    let profiler = get_profiler();
    
    // Create a profile guard that will automatically measure the scope
    let _guard = ProfileGuard::new("my_scope", OperationType::StateOperation, &profiler);
    
    // All code here will be profiled
    // The guard automatically records time when it goes out of scope
}
```

### Using the Macro

A convenient macro is also provided:

```rust
use quantum_protocols::benchmark::{OperationType};
use quantum_protocols::profile_block;

fn my_function() {
    // Profile this block of code
    profile_block!("my_operation", OperationType::QubitOperation);
    
    // Code to profile here
}
```

## Generating Reports

You can generate performance reports programmatically:

```rust
use quantum_protocols::benchmark::{get_profiler, visualization::{create_performance_dashboard, export_to_csv}};

fn generate_reports() {
    let profiler = get_profiler();
    
    // Generate HTML dashboard
    create_performance_dashboard(&profiler, "performance_dashboard.html").unwrap();
    
    // Export data to CSV for further analysis
    export_to_csv(&profiler, "performance_data.csv").unwrap();
    
    // Print text report
    println!("{}", profiler.generate_report());
}
```

## Performance Tips

When optimizing quantum protocols, consider the following:

1. **Qubit Count**: Minimize the number of qubits used in algorithms
2. **Gate Operations**: Reduce the number of quantum gates applied
3. **Serialization**: Optimize classical/quantum data transfer
4. **Error Correction**: Balance error correction overhead with quantum advantage
5. **Measurement Operations**: Avoid unnecessary measurement operations that collapse quantum states

## Continuous Monitoring

For ongoing projects, consider integrating benchmarks into CI/CD pipelines to track performance changes over time.

```yaml
# Example CI step
benchmark:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v2
    - name: Run benchmarks
      run: ./run_all_benchmarks.sh
    - name: Archive benchmark results
      uses: actions/upload-artifact@v2
      with:
        name: benchmark-results
        path: benchmark_results/
``` 