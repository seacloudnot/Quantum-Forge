# PowerShell script to run all benchmarks for Quantum Protocols project

function Print-Header {
    param ([string]$text)
    Write-Host ""
    Write-Host "=== $text ===" -ForegroundColor Blue
    Write-Host "==================" -ForegroundColor Blue
}

function Print-Step {
    param ([string]$text)
    Write-Host ">> $text" -ForegroundColor Cyan
}

function Print-Success {
    param ([string]$text)
    Write-Host "✓ $text" -ForegroundColor Green
}

function Print-Error {
    param ([string]$text, [int]$code)
    Write-Host "✗ $text (Error code: $code)" -ForegroundColor Red
}

# Title
Print-Header "Quantum Protocols Benchmarking Suite"
Write-Host "Running comprehensive benchmarks for the Quantum Protocols project"

# Create results directory
$RESULTS_DIR = "benchmark_results"
if (-not (Test-Path $RESULTS_DIR)) {
    New-Item -ItemType Directory -Path $RESULTS_DIR | Out-Null
}

# Step 1: Run Criterion benchmarks
Print-Step "Running standard benchmarks with Criterion"
cargo bench --bench protocol_benchmarks -- --noplot
$RESULT1 = $LASTEXITCODE
if ($RESULT1 -eq 0) {
    Print-Success "Standard benchmarks completed"
} else {
    Print-Error "Standard benchmarks failed" $RESULT1
}

# Step 2: Run comprehensive benchmarks
Print-Step "Running comprehensive benchmarks"
cargo run --release --bin run_benchmarks
$RESULT2 = $LASTEXITCODE
if ($RESULT2 -eq 0) {
    Print-Success "Comprehensive benchmarks completed"
} else {
    Print-Error "Comprehensive benchmarks failed" $RESULT2
}

# Step 3: Run benchmarks with memory profiling if available
Print-Step "Running memory profiling benchmarks"
cargo run --release --bin run_benchmarks --features memory_profiling
$RESULT3 = $LASTEXITCODE
if ($RESULT3 -eq 0) {
    Print-Success "Memory profiling benchmarks completed"
} else {
    Print-Error "Memory profiling benchmarks failed" $RESULT3
}

# Step 4: Run more detailed benchmarks (if they exist)
if (Test-Path "benches/comprehensive_benchmarks.rs") {
    Print-Step "Running detailed component benchmarks"
    cargo bench --bench comprehensive_benchmarks -- --noplot
    $RESULT4 = $LASTEXITCODE
    if ($RESULT4 -eq 0) {
        Print-Success "Detailed component benchmarks completed"
    } else {
        Print-Error "Detailed component benchmarks failed" $RESULT4
    }
}

# Print summary
Print-Header "Benchmark Summary"
Write-Host "All benchmark results are available in the '$RESULTS_DIR' directory"
Write-Host ""
Write-Host "View the performance dashboard at: $RESULTS_DIR/performance_dashboard.html"
Write-Host "CSV data available at: $RESULTS_DIR/performance_data.csv"
Write-Host "Criterion results available at: target/criterion"

# Completion
if (($RESULT1 -eq 0) -and ($RESULT2 -eq 0)) {
    Write-Host "All benchmarks completed successfully!" -ForegroundColor Green
    exit 0 
} else {
    Write-Host "Some benchmarks encountered errors. Check the output above for details." -ForegroundColor Red
    exit 1
} 