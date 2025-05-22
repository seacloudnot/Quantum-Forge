//! Visualization tools for performance metrics
//!
//! This module provides functionality to generate visual representations of
//! performance data collected by the profiler.

use std::collections::HashMap;
use std::fs::File;
use std::fmt::Write;
use crate::benchmark::PerformanceProfiler;
use std::io::Write as IoWrite;

/// Chart type for visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChartType {
    /// Bar chart
    Bar,
    
    /// Line chart
    Line,
    
    /// Pie chart
    Pie,
    
    /// Scatter plot
    Scatter,
}

/// Data visualization configuration
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Chart type
    pub chart_type: ChartType,
    
    /// Chart title
    pub title: String,
    
    /// Chart width in pixels
    pub width: u32,
    
    /// Chart height in pixels
    pub height: u32,
    
    /// Show legend
    pub show_legend: bool,
    
    /// Output format (html, svg, png)
    pub output_format: String,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            chart_type: ChartType::Bar,
            title: "Performance Metrics".to_string(),
            width: 800,
            height: 600,
            show_legend: true,
            output_format: "html".to_string(),
        }
    }
}

/// HTML template for rendering charts using Chart.js
const HTML_TEMPLATE: &str = r"
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src=https://cdn.jsdelivr.net/npm/chart.js></script>
</head>
<body>
    <div style='width: {width}px; height: {height}px; margin: 0 auto;'>
        <canvas id=chart></canvas>
    </div>
    <div id=summary style='margin: 20px auto; width: 80%; padding: 10px; border: 1px solid #ccc; border-radius: 5px;'>
        <h3>Summary</h3>
        <pre>{summary}</pre>
    </div>
    <script>
        const ctx = document.getElementById('chart').getContext('2d');
        const chart = new Chart(ctx, {
            type: '{chart_type}',
            data: {
                labels: {labels},
                datasets: [{
                    label: '{dataset_label}',
                    data: {data},
                    backgroundColor: {colors},
                    borderColor: {border_colors},
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: {show_legend}
                    },
                    title: {
                        display: true,
                        text: '{title}'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: '{y_axis_label}'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: '{x_axis_label}'
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>
";

/// Generate an HTML performance report
///
/// # Arguments
///
/// * `profiler` - The performance profiler containing metrics
/// * `config` - Visualization configuration
///
/// # Returns
///
/// HTML report as a string
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn generate_html_report(profiler: &PerformanceProfiler, config: &VisualizationConfig) -> String {
    let aggregated = profiler.aggregate_by_type();
    if aggregated.is_empty() {
        return "No performance data collected ".to_string();
    }
    
    // Extract data for chart
    let mut labels = Vec::new();
    let mut data = Vec::new();
    let mut colors = Vec::new();
    let mut border_colors = Vec::new();
    
    // Generate random colors for the chart
    let color_palette = [
        "rgba(255, 99, 132, 0.6)",
        "rgba(54, 162, 235, 0.6)",
        "rgba(255, 206, 86, 0.6)",
        "rgba(75, 192, 192, 0.6)",
        "rgba(153, 102, 255, 0.6)",
        "rgba(255, 159, 64, 0.6)",
        "rgba(199, 199, 199, 0.6)",
        "rgba(83, 102, 255, 0.6)",
        "rgba(40, 159, 64, 0.6)",
        "rgba(210, 199, 199, 0.6)",
    ];
    
    let border_palette = [
        "rgba(255, 99, 132, 1)",
        "rgba(54, 162, 235, 1)",
        "rgba(255, 206, 86, 1)",
        "rgba(75, 192, 192, 1)",
        "rgba(153, 102, 255, 1)",
        "rgba(255, 159, 64, 1)",
        "rgba(199, 199, 199, 1)",
        "rgba(83, 102, 255, 1)",
        "rgba(40, 159, 64, 1)",
        "rgba(210, 199, 199, 1)",
    ];
    
    // Prepare data for each operation type
    for (i, (op_type, metrics)) in aggregated.iter().enumerate() {
        labels.push(op_type.to_string());
        let micros = u64::try_from(metrics.avg_duration.as_micros()).unwrap_or(u64::MAX);
        data.push(micros as f64 / 1000.0); // convert to milliseconds
        
        // Use colors from palette or generate new ones
        let color_idx = i % color_palette.len();
        colors.push(color_palette[color_idx].to_string());
        border_colors.push(border_palette[color_idx].to_string());
    }
    
    // Generate summary text
    let summary = profiler.generate_report();
    
    // Create the chart HTML
    let chart_type = match config.chart_type {
        ChartType::Bar => "bar",
        ChartType::Line => "line",
        ChartType::Pie => "pie",
        ChartType::Scatter => "scatter",
    };
    
    HTML_TEMPLATE
        .replace("{title}", &config.title)
        .replace("{width}", &config.width.to_string())
        .replace("{height}", &config.height.to_string())
        .replace("{chart_type}", chart_type)
        .replace("{labels}", &format!("{labels:?}"))
        .replace("{data}", &format!("{data:?}"))
        .replace("{colors}", &format!("{colors:?}"))
        .replace("{border_colors}", &format!("{border_colors:?}"))
        .replace("{show_legend}", &config.show_legend.to_string())
        .replace("{dataset_label}", "Average Duration (ms)")
        .replace("{y_axis_label}", "Duration (ms)")
        .replace("{x_axis_label}", "Operation Type ")
        .replace("{summary}", &summary)
}

/// Save HTML report to a file
///
/// # Arguments
///
/// * `profiler` - The performance profiler containing metrics
/// * `config` - Visualization configuration
/// * `path` - File path
///
/// # Returns
///
/// IO result indicating success or failure
///
/// # Errors
///
/// Will return an error if writing to the file fails.
pub fn save_html_report(
    profiler: &PerformanceProfiler,
    config: &VisualizationConfig,
    path: &str,
) -> std::io::Result<()> {
    let html = generate_html_report(profiler, config);
    let mut file = File::create(path)?;
    file.write_all(html.as_bytes())?;
    Ok(())
}

/// Create a performance dashboard HTML file
///
/// # Arguments
///
/// * `profiler` - The performance profiler containing metrics
/// * `output_path` - Path to write the dashboard HTML
///
/// # Returns
///
/// IO result indicating success or failure
///
/// # Errors
///
/// Will return an error if writing to the file fails.
pub fn create_performance_dashboard(
    profiler: &PerformanceProfiler,
    output_path: &str,
) -> std::io::Result<()> {
    // Create a comprehensive HTML dashboard
    let mut html = String::from(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>Quantum Protocols Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .dashboard { display: flex; flex-wrap: wrap; }
        .chartcontainer { 
            width: 48%; 
            margin: 1%; 
            padding: 10px; 
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
        }
        .fullwidth { width: 98%; }
        h1, h2 { color: #333; }
        .summary { 
            background-color: #f9f9f9; 
            padding: 15px; 
            border-radius: 5px;
            margin-bottom: 20px;
        }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        tr:hover { background-color: #f5f5f5; }
    </style>
</head>
<body>
    <h1>Quantum Protocols Performance Dashboard</h1>
"#
    );

    // Add summary section
    let metrics = profiler.get_metrics();
    let aggregated = profiler.aggregate_by_type();
    
    html.push_str("<div class=summary>");
    html.push_str("<h2>Performance Summary</h2>");
    
    if metrics.is_empty() {
        html.push_str("<p>No performance data collected </p>");
    } else {
        // Calculate total operations and average duration
        let total_operations = metrics.len();
        let total_duration: u128 = metrics.iter().map(|m| m.duration.as_micros()).sum();
        
        // Fix: Use safe conversions to prevent precision loss
        let total_duration_ms = (u64::try_from(total_duration).unwrap_or(u64::MAX)) as f64 / 1000.0;
        let avg_duration = if total_operations > 0 {
            total_duration_ms / (u32::try_from(total_operations).unwrap_or(u32::MAX)) as f64
        } else {
            0.0
        };
        
        let _ = write!(html, "<p>Total Operations: {total_operations}</p>");
        let _ = write!(html, "<p>Total Duration: {:.2} ms</p>", total_duration_ms);
        let _ = write!(html, "<p>Average Duration: {:.2} ms</p>", avg_duration);
        
        // Find slowest and fastest operations
        if let Some(slowest) = metrics.iter().max_by_key(|m| m.duration) {
            // Fix: Use safe conversion for duration
            let slowest_ms = (u64::try_from(slowest.duration.as_micros()).unwrap_or(u64::MAX)) as f64 / 1000.0;
            let _ = write!(
                html,
                "<p>Slowest Operation: {} ({}) - {:.2} ms</p>",
                slowest.name, 
                slowest.op_type,
                slowest_ms
            );
        }
        
        if let Some(fastest) = metrics.iter().min_by_key(|m| m.duration) {
            // Fix: Use safe conversion for duration
            let fastest_ms = (u64::try_from(fastest.duration.as_micros()).unwrap_or(u64::MAX)) as f64 / 1000.0;
            let _ = write!(
                html,
                "<p>Fastest Operation: {} ({}) - {:.2} ms</p>",
                fastest.name, 
                fastest.op_type,
                fastest_ms
            );
        }
    }
    
    html.push_str("</div>");
    
    // Add dashboard charts
    html.push_str("<div class=dashboard>");
    
    // Operation type duration chart
    html.push_str("<div class=chartcontainer>");
    html.push_str("<h2>Average Duration by Operation Type</h2>");
    html.push_str("<canvas id=opTypeChart></canvas>");
    html.push_str("</div>");
    
    // Operation count chart
    html.push_str("<div class=chartcontainer>");
    html.push_str("<h2>Operation Count by Type</h2>");
    html.push_str("<canvas id=opCountChart></canvas>");
    html.push_str("</div>");
    
    // Top 10 slowest operations chart
    html.push_str("<div class=chartcontainer>");
    html.push_str("<h2>Top 10 Slowest Operations</h2>");
    html.push_str("<canvas id=slowestOpsChart></canvas>");
    html.push_str("</div>");
    
    // Performance distribution chart
    html.push_str("<div class=chartcontainer>");
    html.push_str("<h2>Performance Distribution</h2>");
    html.push_str("<canvas id=perfDistChart></canvas>");
    html.push_str("</div>");
    
    // Detailed metrics table (full width)
    html.push_str("<div class=fullwidth>");
    html.push_str("<h2>Detailed Performance Metrics</h2>");
    html.push_str("<table>");
    html.push_str("<tr><th>Operation</th><th>Type</th><th>Duration (ms)</th><th>Memory (if available)</th></tr>");
    
    for metric in &metrics {
        // Get the measured bit value (or estimated from probability)
        let memory_display = if metric.metadata.contains("Memory:") {
            metric.metadata.clone()
        } else {
            "N/A".to_string()
        };
        
        let _ = write!(
            html,
            "<tr><td>{}</td><td>{}</td><td>{:.2}</td><td>{}</td></tr>",
            metric.name,
            metric.op_type,
            metric.duration.as_micros() as f64 / 1000.0,
            memory_display
        );
    }
    
    html.push_str("</table>");
    html.push_str("</div>");
    
    html.push_str("</div>"); // Close dashboard div
    
    // Add JavaScript for charts
    html.push_str("<script>");
    
    // Prepare data for charts
    let mut op_type_labels = Vec::new();
    let mut op_type_data = Vec::new();
    let mut op_count_data = Vec::new();
    
    for (op_type, metrics) in &aggregated {
        op_type_labels.push(op_type.to_string());
        op_type_data.push(metrics.avg_duration.as_micros() as f64 / 1000.0);
        op_count_data.push(metrics.count);
    }
    
    // Generate colors
    let colors: Vec<String> = op_type_labels
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let color_idx = i % 10;
            format!(
                "rgba({}, {}, {}, 0.6)",
                55 + color_idx * 20,
                100 + color_idx * 15,
                200 - color_idx * 10
            )
        })
        .collect();
    
    // Format operation type chart data
    let op_type_labels_str = format!("{op_type_labels:?}");
    let op_type_data_str = format!("{op_type_data:?}");
    let op_count_data_str = format!("{op_count_data:?}");
    let colors_str = format!("{colors:?}");

    // Write operation type chart
    write!(html, r#"
        const opTypeCtx = document.getElementById("opTypeChart").getContext("2d");
        new Chart(opTypeCtx, {{
            type: "bar",
            data: {{
                labels: {op_type_labels_str},
                datasets: [{{
                    label: "Average Duration (ms)",
                    data: {op_type_data_str},
                    backgroundColor: {colors_str},
                    borderColor: {colors_str},
                    borderWidth: 1
                }},
                {{
                    label: "Operation Count",
                    data: {op_count_data_str},
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                    yAxisID: 'count'
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{ y: {{ beginAtZero: true }} }}
            }}
        }});
        "#).unwrap();

    // Add operation count chart
    write!(html, r#"
        const opCountCtx = document.getElementById("opCountChart").getContext("2d");
        new Chart(opCountCtx, {{
            type: "pie",
            data: {{
                labels: {op_type_labels_str},
                datasets: [{{
                    label: "Operation Count",
                    data: {op_count_data_str},
                    backgroundColor: {colors_str},
                    borderColor: {colors_str},
                    borderWidth: 1
                }}]
            }},
            options: {{ responsive: true }}
        }});
        "#).unwrap();
    
    // Top 10 slowest operations chart
    let mut sorted_metrics = metrics.clone();
    sorted_metrics.sort_by(|a, b| b.duration.cmp(&a.duration));
    let slowest_ops = sorted_metrics.iter().take(10).collect::<Vec<_>>();
    
    let slowest_labels: Vec<_> = slowest_ops.iter().map(|m| m.name.clone()).collect();
    let slowest_data: Vec<_> = slowest_ops.iter().map(|m| m.duration.as_micros() as f64 / 1000.0).collect();
    
    let slowest_labels_str = format!("{slowest_labels:?}");
    let slowest_data_str = format!("{slowest_data:?}");
    
    write!(html, r#"
        const slowestOpsCtx = document.getElementById("slowestOpsChart").getContext("2d");
        new Chart(slowestOpsCtx, {{
            type: "bar",
            data: {{
                labels: {slowest_labels_str},
                datasets: [{{
                    label: "Duration (ms)",
                    data: {slowest_data_str},
                    backgroundColor: "rgba(255, 99, 132, 0.6)",
                    borderColor: "rgba(255, 99, 132, 1)",
                    borderWidth: 1
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                plugins: {{ title: {{ display: true, text: 'Top 10 Slowest Operations' }} }}
            }}
        }});
        "#).unwrap();
    
    // Performance distribution chart (histogram-like)
    let mut range_data = HashMap::new();
    let range_size = 0.5; // 0.5ms bins
    
    for metric in &metrics {
        let duration_ms = metric.duration.as_micros() as f64 / 1000.0;
        let range_idx = (duration_ms / range_size).floor() as i32;
        let range_key = format!("{}-{} ms", range_idx as f64 * range_size, (range_idx as f64 + 1.0) * range_size);
        
        *range_data.entry(range_key).or_insert(0) += 1;
    }
    
    let range_labels: Vec<_> = range_data.keys().cloned().collect();
    let range_values: Vec<_> = range_data.values().copied().collect();
    
    let range_labels_str = format!("{range_labels:?}");
    let range_values_str = format!("{range_values:?}");
    
    write!(html, r#"
        const perfDistCtx = document.getElementById("perfDistChart").getContext("2d");
        new Chart(perfDistCtx, {{
            type: "bar",
            data: {{
                labels: {range_labels_str},
                datasets: [{{
                    label: "Number of Operations",
                    data: {range_values_str},
                    backgroundColor: "rgba(75, 192, 192, 0.6)",
                    borderColor: "rgba(75, 192, 192, 1)",
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ title: {{ display: true, text: 'Performance Distribution' }} }}
            }}
        }});
        "#).unwrap();
    
    html.push_str("</script>");
    html.push_str("</body></html>");
    
    // Write to file
    let mut file = File::create(output_path)?;
    file.write_all(html.as_bytes())?;
    
    Ok(())
}

/// Export performance data to CSV format
///
/// # Errors
/// 
/// Returns an error if writing to the file fails
pub fn export_to_csv(profiler: &PerformanceProfiler, path: &str) -> std::io::Result<()> {
    let metrics = profiler.get_metrics();
    if metrics.is_empty() {
        return Ok(());
    }
    
    let mut file = File::create(path)?;
    
    // Write CSV header
    writeln!(file, "name,operation_type,duration_ms,memory_bytes,qubit_count")?;
    
    // Write each metric
    for metric in metrics {
        writeln!(
            file,
            "{},{},{:.3},{}",
            metric.name,
            metric.op_type,
            metric.duration.as_micros() as f64 / 1000.0,
            if metric.metadata.contains("Memory:") { 
                metric.metadata.clone() 
            } else { 
                "-".to_string() 
            }
        )?;
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::*;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_generate_html_report() {
        let profiler = PerformanceProfiler::new();
        
        // Add test operations
        profiler.profile("test_op1", &OperationType::QubitOperation, || {
            thread::sleep(Duration::from_millis(5));
        });
        
        profiler.profile("test_op2", &OperationType::StateOperation, || {
            thread::sleep(Duration::from_millis(10));
        });
        
        let config = VisualizationConfig {
            title: "Test Chart".to_string(),
            ..Default::default()
        };
        
        let html = generate_html_report(&profiler, &config);
        
        // Check that the HTML contains expected elements
        assert!(html.contains("Test Chart"));
        assert!(html.contains("Qubit Operation"));
        assert!(html.contains("State Operation"));
        assert!(html.contains("<canvas id=chart></canvas>"));
    }
    
    #[test]
    fn test_create_performance_dashboard() {
        let profiler = PerformanceProfiler::new();
        
        // Add test operations
        profiler.profile("fast_op", &OperationType::QubitOperation, || {
            thread::sleep(Duration::from_millis(1));
        });
        
        profiler.profile("slow_op", &OperationType::StateOperation, || {
            thread::sleep(Duration::from_millis(15));
        });
        
        // Create a temporary file
        let temp_file = std::env::temp_dir().join("test_dashboard.html");
        let path = temp_file.to_str().unwrap();
        
        // Create dashboard
        let result = create_performance_dashboard(&profiler, path);
        assert!(result.is_ok());
        
        // Check if file exists
        assert!(std::path::Path::new(path).exists());
        
        // Clean up
        std::fs::remove_file(path).ok();
    }
} 