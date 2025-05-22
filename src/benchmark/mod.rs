// Performance benchmarking and profiling
//
// This module provides functionality for profiling and benchmarking
// quantum operations.

#![allow(clippy::non_std_lazy_statics)]

// Import necessary modules
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::fmt::Write;
use lazy_static::lazy_static;

pub mod visualization;

// Export key items for easier use
pub use visualization::{ChartType, VisualizationConfig};

/// Types of operations being profiled
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OperationType {
    /// Basic qubit operations (gates, measurement)
    QubitOperation,
    
    /// Quantum register operations
    RegisterOperation,
    
    /// Quantum state manipulations
    StateOperation,
    
    /// Error correction operations
    ErrorCorrection,
    
    /// Consensus protocol operations
    ConsensusProtocol,
    
    /// Classical-quantum translation
    ClassicalQuantumTranslation,
    
    /// Security operations (encryption, RNG)
    SecurityOperation,
    
    /// Network operations
    NetworkOperation,
    
    /// Custom operation type
    Custom(String),
}

impl std::fmt::Display for OperationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperationType::QubitOperation => write!(f, "Qubit Operation"),
            OperationType::RegisterOperation => write!(f, "Register Operation"),
            OperationType::StateOperation => write!(f, "State Operation"),
            OperationType::ErrorCorrection => write!(f, "Error Correction"),
            OperationType::ConsensusProtocol => write!(f, "Consensus Protocol"),
            OperationType::ClassicalQuantumTranslation => write!(f, "Classical-Quantum Translation"),
            OperationType::SecurityOperation => write!(f, "Security Operation"),
            OperationType::NetworkOperation => write!(f, "Network Operation"),
            OperationType::Custom(name) => write!(f, "Custom: {name}"),
        }
    }
}

/// Metrics for a single operation
#[derive(Debug, Clone)]
pub struct OperationMetrics {
    /// Operation name
    pub name: String,
    
    /// Operation type
    pub op_type: String,
    
    /// Duration of the operation
    pub duration: Duration,
    
    /// Additional metadata
    pub metadata: String,
}

impl OperationMetrics {
    /// Create a new operation metrics instance
    #[must_use]
    pub fn new(name: &str, op_type: &str, duration: Duration) -> Self {
        Self {
            name: name.to_string(),
            op_type: op_type.to_string(),
            duration,
            metadata: String::new(),
        }
    }
    
    /// Add metadata to the metrics
    ///
    /// # Arguments
    ///
    /// * `metadata` - Metadata to add
    ///
    /// # Returns
    ///
    /// Self for method chaining
    #[must_use]
    pub fn with_metadata(mut self, metadata: &str) -> Self {
        self.metadata = metadata.to_string();
        self
    }
}

/// Aggregated metrics for multiple operations
#[derive(Debug, Clone, Default)]
pub struct MetricsAggregate {
    /// Number of operations
    pub count: usize,
    
    /// Total duration of all operations
    pub total_duration: Duration,
    
    /// Average duration per operation
    pub avg_duration: Duration,
}

/// Performance profiler for quantum operations
#[derive(Debug, Clone)]
pub struct PerformanceProfiler {
    /// Operation metrics collected
    metrics: Arc<Mutex<Vec<OperationMetrics>>>,
    
    /// Whether profiling is enabled
    enabled: bool,
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    #[must_use]
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(Vec::new())),
            enabled: true,
        }
    }
    
    /// Enable profiling
    pub fn enable(&mut self) {
        self.enabled = true;
    }
    
    /// Disable profiling
    pub fn disable(&mut self) {
        self.enabled = false;
    }
    
    /// Is profiling enabled?
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    /// Reset all collected metrics
    ///
    /// # Panics
    ///
    /// May panic if the mutex is poisoned
    pub fn reset(&self) {
        if self.enabled {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.clear();
        }
    }
    
    /// Profile a function execution and record its performance metrics
    pub fn profile<F, R>(&self, name: &str, op_type: &OperationType, func: F) -> R
    where
        F: FnOnce() -> R,
    {
        if !self.enabled {
            return func();
        }
        
        let start = Instant::now();
        let result = func();
        let duration = start.elapsed();
        
        // Only collect metrics if profiling is enabled
        let _ = self.metrics.lock().map(|mut metrics| {
            metrics.push(OperationMetrics::new(name, &op_type.to_string(), duration));
        });
        
        result
    }
    
    /// Get a copy of all collected metrics
    ///
    /// # Panics
    ///
    /// May panic if the mutex is poisoned
    #[must_use]
    pub fn get_metrics(&self) -> Vec<OperationMetrics> {
        if !self.enabled {
            return Vec::new();
        }
        
        let metrics = self.metrics.lock().unwrap();
        metrics.clone()
    }
    
    /// Generate aggregated metrics by operation type
    ///
    /// # Panics
    ///
    /// May panic if the mutex is poisoned
    #[must_use]
    pub fn aggregate_by_type(&self) -> HashMap<OperationType, MetricsAggregate> {
        if !self.enabled {
            return HashMap::new();
        }
        
        let metrics = self.metrics.lock().unwrap();
        let mut result = HashMap::new();
        
        for metric in metrics.iter() {
            // Convert string operation type to OperationType enum
            let op_type = match metric.op_type.as_str() {
                "Qubit Operation" => OperationType::QubitOperation,
                "Register Operation" => OperationType::RegisterOperation,
                "State Operation" => OperationType::StateOperation,
                "Error Correction" => OperationType::ErrorCorrection,
                "Consensus Protocol" => OperationType::ConsensusProtocol,
                "Classical-Quantum Translation" => OperationType::ClassicalQuantumTranslation,
                "Security Operation" => OperationType::SecurityOperation,
                "Network Operation" => OperationType::NetworkOperation,
                _ => OperationType::Custom(metric.op_type.clone()),
            };
            
            let entry = result.entry(op_type).or_insert_with(MetricsAggregate::default);
            
            entry.count += 1;
            entry.total_duration += metric.duration;
        }
        
        // Calculate averages
        for agg in result.values_mut() {
            if agg.count > 0 {
                agg.avg_duration = agg.total_duration / u32::try_from(agg.count).unwrap_or(1);
            }
        }
        
        result
    }
    
    /// Generate a text report of collected metrics
    ///
    /// # Panics
    ///
    /// May panic if the mutex is poisoned
    #[must_use]
    pub fn generate_report(&self) -> String {
        if !self.enabled {
            return "Profiling disabled".to_string();
        }
        
        let mut report = String::new();
        report.push_str("Quantum Protocols Performance Report\n");
        report.push_str("====================================\n\n");
        
        // Group metrics by operation type
        let aggregated = self.aggregate_by_operation_type();
        
        // Add summary for each operation type
        for (op_type, metrics) in &aggregated {
            writeln!(
                report,
                "{} (count: {})\n  - Avg: {:?}",
                op_type,
                metrics.count,
                metrics.avg_duration
            ).unwrap();
            
            writeln!(report).unwrap();
        }
        
        // Add detailed metrics for slowest operations
        let metrics = self.metrics.lock().unwrap();
        if !metrics.is_empty() {
            let mut sorted_metrics = metrics.clone();
            sorted_metrics.sort_by(|a, b| b.duration.cmp(&a.duration));
            
            writeln!(report, "\nSlowest Operations:").unwrap();
            writeln!(report, "------------------").unwrap();
            
            for (i, metric) in sorted_metrics.iter().take(10).enumerate() {
                writeln!(
                    report,
                    "{}. {} ({}): {:?}",
                    i + 1,
                    metric.name,
                    metric.op_type,
                    metric.duration
                ).unwrap();
                
                if !metric.metadata.is_empty() {
                    writeln!(report, "   Metadata: {}", metric.metadata).unwrap();
                }
            }
        }
        
        report
    }
    
    /// Save performance report to a file
    ///
    /// # Errors
    ///
    /// Returns an error if writing to the file fails
    pub fn save_report_to_file(&self, path: &str) -> std::io::Result<()> {
        let report = self.generate_report();
        std::fs::write(path, report)
    }
    
    /// Profile a function's memory usage and execution time
    ///
    /// # Panics
    ///
    /// This function may panic if the mutex is poisoned when accessing metrics
    #[cfg(feature = "memory_profiling")]
    pub fn profile_memory<F, R>(&self, name: &str, op_type: &OperationType, func: F) -> R
    where
        F: FnOnce() -> R,
    {
        if !self.enabled {
            return func();
        }
        
        let start = Instant::now();
        let result = func();
        let duration = start.elapsed();
        
        let mut metrics = OperationMetrics::new(name, &op_type.to_string(), duration);
        metrics.duration = duration;
        
        // Try to estimate memory using heap allocation
        #[cfg(feature = "memory_profiling")]
        {
            use std::alloc::{GlobalAlloc, Layout, System};
            
            struct MemoryTracker;
            
            static ALLOCATED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            
            #[global_allocator]
            static ALLOCATOR: MemoryTracker = MemoryTracker;
            
            unsafe impl GlobalAlloc for MemoryTracker {
                unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                    let ptr = System.alloc(layout);
                    if !ptr.is_null() {
                        ALLOCATED.fetch_add(layout.size(), std::sync::atomic::Ordering::SeqCst);
                    }
                    ptr
                }
                
                unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                    System.dealloc(ptr, layout);
                    ALLOCATED.fetch_sub(layout.size(), std::sync::atomic::Ordering::SeqCst);
                }
            }
            
            metrics.metadata = format!("Memory: {} bytes", ALLOCATED.load(std::sync::atomic::Ordering::SeqCst));
        }
        
        // Store the metrics
        let mut all_metrics = self.metrics.lock().unwrap();
        all_metrics.push(metrics);
        
        result
    }
    
    /// Stub implementation when memory profiling is not enabled
    #[cfg(not(feature = "memory_profiling"))]
    pub fn profile_memory<F, R>(&self, name: &str, op_type: &OperationType, func: F) -> R
    where
        F: FnOnce() -> R,
    {
        // Without memory profiling, just use regular profiling
        self.profile(name, op_type, func)
    }

    /// Group metrics by operation type and calculate aggregates
    #[must_use]
    pub fn aggregate_by_operation_type(&self) -> HashMap<String, MetricsAggregate> {
        // Create HashMap to hold the aggregates
        let mut aggregated = HashMap::new();
        
        // First try to lock the metrics
        let Ok(metrics_lock) = self.metrics.lock() else { return aggregated };
        
        // Calculate aggregates
        for metric in &*metrics_lock {
            let entry = aggregated.entry(metric.op_type.clone()).or_insert(MetricsAggregate {
                count: 0,
                total_duration: Duration::from_secs(0),
                avg_duration: Duration::from_secs(0),
            });
            
            entry.count += 1;
            entry.total_duration += metric.duration;
        }
        
        // Calculate average durations
        for agg in aggregated.values_mut() {
            if agg.count > 0 {
                agg.avg_duration = agg.total_duration / u32::try_from(agg.count).unwrap_or(1);
            }
        }
        
        aggregated
    }
}

/// Get a global performance profiler instance
#[must_use]
pub fn get_profiler() -> Arc<PerformanceProfiler> {
    lazy_static! {
        static ref PROFILER: Arc<PerformanceProfiler> = Arc::new(PerformanceProfiler::new());
    }
    
    PROFILER.clone()
}

/// A guard that automatically records the duration of its scope
pub struct ProfileGuard<'a> {
    name: String,
    op_type: OperationType,
    start: Instant,
    profiler: &'a PerformanceProfiler,
    metadata: HashMap<String, String>,
}

impl<'a> ProfileGuard<'a> {
    /// Create a new `ProfileGuard` to automatically profile function execution
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the operation
    /// * `op_type` - Type of operation
    /// * `profiler` - Performance profiler to use
    #[must_use]
    pub fn new(name: &str, op_type: OperationType, profiler: &'a PerformanceProfiler) -> Self {
        Self {
            name: name.to_string(),
            op_type,
            start: Instant::now(),
            profiler,
            metadata: HashMap::new(),
        }
    }
    
    /// Add metadata to the profile guard
    pub fn add_metadata(&mut self, key: &str, value: &str) -> &mut Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

impl Drop for ProfileGuard<'_> {
    fn drop(&mut self) {
        if !self.profiler.is_enabled() {
            return;
        }
        
        let duration = self.start.elapsed();
        
        let mut metrics = OperationMetrics::new(&self.name, &self.op_type.to_string(), duration);
        metrics.duration = duration;
        
        // Convert metadata to string using fold instead of map + collect
        metrics.metadata = if self.metadata.is_empty() {
            String::new()
        } else {
            let mut result = String::new();
            let mut first = true;
            for (k, v) in &self.metadata {
                if !first {
                    result.push_str(", ");
                }
                write!(result, "{k}: {v}").unwrap();
                first = false;
            }
            result
        };
        
        // Store the metrics
        if let Ok(mut all_metrics) = self.profiler.metrics.lock() {
            all_metrics.push(metrics);
        }
    }
}

/// Macro to easily profile a block of code
#[macro_export]
macro_rules! profile_block {
    ($name:expr, $op_type:expr) => {
        let _guard = {
            let profiler = $crate::benchmark::get_profiler();
            $crate::benchmark::ProfileGuard::new($name, $op_type, &profiler)
        };
    };
    ($name:expr, $op_type:expr, $profiler:expr) => {
        let _guard = $crate::benchmark::ProfileGuard::new($name, $op_type, $profiler);
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_basic_profiling() {
        let profiler = PerformanceProfiler::new();
        
        // Profile a simple operation
        profiler.profile("test_operation", &OperationType::QubitOperation, || {
            // Simulate some work
            thread::sleep(Duration::from_millis(10));
        });
        
        let metrics = profiler.get_metrics();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].name, "test_operation");
        assert!(metrics[0].duration.as_millis() >= 10);
    }
    
    #[test]
    fn test_profiling_disabled() {
        let mut profiler = PerformanceProfiler::new();
        profiler.disable();
        
        profiler.profile("test_operation", &OperationType::QubitOperation, || {
            // Simulate some work
            thread::sleep(Duration::from_millis(10));
        });
        
        let metrics = profiler.get_metrics();
        assert!(metrics.is_empty());
    }
    
    #[test]
    fn test_profile_guard() {
        let profiler = PerformanceProfiler::new();
        
        {
            let _guard = ProfileGuard::new("guard_test", OperationType::StateOperation, &profiler);
            // Simulate some work
            thread::sleep(Duration::from_millis(10));
        }
        
        let metrics = profiler.get_metrics();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].name, "guard_test");
        assert!(metrics[0].duration.as_millis() >= 10);
    }
    
    #[test]
    fn test_aggregation() {
        let profiler = PerformanceProfiler::new();
        
        // Add different operation types
        profiler.profile("op1", &OperationType::QubitOperation, || {
            thread::sleep(Duration::from_millis(10));
        });
        
        profiler.profile("op2", &OperationType::QubitOperation, || {
            thread::sleep(Duration::from_millis(20));
        });
        
        profiler.profile("op3", &OperationType::StateOperation, || {
            thread::sleep(Duration::from_millis(15));
        });
        
        let aggregated = profiler.aggregate_by_type();
        assert_eq!(aggregated.len(), 2);
        
        let qubit_metrics = &aggregated[&OperationType::QubitOperation];
        assert_eq!(qubit_metrics.count, 2);
        assert!(qubit_metrics.total_duration.as_millis() >= 30);
        
        let state_metrics = &aggregated[&OperationType::StateOperation];
        assert_eq!(state_metrics.count, 1);
        assert!(state_metrics.total_duration.as_millis() >= 15);
    }
    
    #[test]
    fn test_generate_report() {
        let profiler = PerformanceProfiler::new();
        
        // Add some test operations
        profiler.profile("fast_op", &OperationType::QubitOperation, || {
            thread::sleep(Duration::from_millis(5));
        });
        
        profiler.profile("slow_op", &OperationType::StateOperation, || {
            thread::sleep(Duration::from_millis(30));
        });
        
        let report = profiler.generate_report();
        assert!(report.contains("Quantum Protocols Performance Report"));
        assert!(report.contains("Qubit Operation"));
        assert!(report.contains("State Operation"));
        assert!(report.contains("Slowest Operations"));
    }
} 