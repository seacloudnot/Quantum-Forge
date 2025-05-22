#!/bin/bash

# Run benchmark script with a colorful output
print_header() {
    echo -e "\e[1;34m=== $1 ===\e[0m"
    echo -e "\e[1;34m$(printf '=%.0s' $(seq 1 ${#1}))\e[0m"
}

print_step() {
    echo -e "\e[0;36m>> $1\e[0m"
}

print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "\e[0;32m✓ $2\e[0m"
    else
        echo -e "\e[0;31m✗ $2 (Error code: $1)\e[0m"
    fi
}

print_header "Quantum Protocols Benchmarking Suite"
echo "Running comprehensive benchmarks for the Quantum Protocols project"
echo ""

# Create results directory
RESULTS_DIR="benchmark_results"
mkdir -p $RESULTS_DIR

# Step 1: Run Criterion benchmarks
print_step "Running standard benchmarks with Criterion"
cargo bench --bench protocol_benchmarks -- --noplot
RESULT1=$?
print_status $RESULT1 "Standard benchmarks completed"
echo ""

# Step 2: Run comprehensive benchmarks
print_step "Running comprehensive benchmarks"
cargo run --release --bin run_benchmarks
RESULT2=$?
print_status $RESULT2 "Comprehensive benchmarks completed"
echo ""

# Step 3: Run benchmarks with memory profiling if available
print_step "Running memory profiling benchmarks"
cargo run --release --bin run_benchmarks --features memory_profiling
RESULT3=$?
print_status $RESULT3 "Memory profiling benchmarks completed"
echo ""

# Step 4: Run more detailed benchmarks (if they exist)
if [ -f "benches/comprehensive_benchmarks.rs" ]; then
    print_step "Running detailed component benchmarks"
    cargo bench --bench comprehensive_benchmarks -- --noplot
    RESULT4=$?
    print_status $RESULT4 "Detailed component benchmarks completed"
    echo ""
fi

# Print summary
print_header "Benchmark Summary"
echo "All benchmark results are available in the '$RESULTS_DIR' directory"
echo ""
echo "View the performance dashboard at: $RESULTS_DIR/performance_dashboard.html"
echo "CSV data available at: $RESULTS_DIR/performance_data.csv"
echo "Criterion results available at: target/criterion"
echo ""

# Completion
if [ $RESULT1 -eq 0 ] && [ $RESULT2 -eq 0 ]; then
    echo -e "\e[0;32mAll benchmarks completed successfully!\e[0m"
    exit 0
else
    echo -e "\e[0;31mSome benchmarks encountered errors. Check the output above for details.\e[0m"
    exit 1
fi 