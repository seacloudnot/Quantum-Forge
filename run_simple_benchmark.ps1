Write-Host "Running Quantum Protocols benchmarks" -ForegroundColor Blue

# Create results directory
$RESULTS_DIR = "benchmark_results"
if (-not (Test-Path $RESULTS_DIR)) {
    New-Item -ItemType Directory -Path $RESULTS_DIR | Out-Null
}

# Run the benchmark
Write-Host "Running benchmarks..." -ForegroundColor Cyan
cargo run --release --bin run_benchmarks

if ($LASTEXITCODE -eq 0) {
    Write-Host "Benchmarks completed successfully!" -ForegroundColor Green
} else {
    Write-Host "Benchmarks failed with error code: $LASTEXITCODE" -ForegroundColor Red
}

Write-Host "Results are in the $RESULTS_DIR directory" -ForegroundColor Yellow 