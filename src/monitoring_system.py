#!/usr/bin/env python3
"""
Advanced Monitoring and Logging System for PDF Structure Detection
Provides comprehensive monitoring, metrics collection, and performance analysis
"""

import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import psutil
import os
from pathlib import Path

@dataclass
class ProcessingMetrics:
    """Metrics for a single PDF processing operation"""
    file_name: str
    file_size_mb: float
    page_count: int
    processing_time: float
    memory_peak_mb: float
    cpu_usage_percent: float
    blocks_extracted: int
    headings_detected: int
    confidence_avg: float
    success: bool
    error_message: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class SystemMetrics:
    """System-wide performance metrics"""
    total_files_processed: int = 0
    total_processing_time: float = 0.0
    total_failures: int = 0
    avg_processing_time: float = 0.0
    max_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    files_exceeding_time_limit: int = 0
    memory_peak_mb: float = 0.0
    cpu_avg_percent: float = 0.0
    throughput_files_per_minute: float = 0.0
    success_rate: float = 0.0
    
    def update_with_processing_metrics(self, metrics: ProcessingMetrics):
        """Update system metrics with processing metrics"""
        self.total_files_processed += 1
        self.total_processing_time += metrics.processing_time
        
        if not metrics.success:
            self.total_failures += 1
        
        # Update timing statistics
        if metrics.processing_time > self.max_processing_time:
            self.max_processing_time = metrics.processing_time
        
        if metrics.processing_time < self.min_processing_time:
            self.min_processing_time = metrics.processing_time
        
        if metrics.processing_time > 10.0:  # Hackathon time limit
            self.files_exceeding_time_limit += 1
        
        # Update averages
        self.avg_processing_time = self.total_processing_time / self.total_files_processed
        self.success_rate = (self.total_files_processed - self.total_failures) / self.total_files_processed * 100
        
        # Update resource usage
        if metrics.memory_peak_mb > self.memory_peak_mb:
            self.memory_peak_mb = metrics.memory_peak_mb
        
        # Update throughput (files per minute)
        if self.total_processing_time > 0:
            self.throughput_files_per_minute = (self.total_files_processed / self.total_processing_time) * 60

class PerformanceMonitor:
    """Performance monitoring and metrics collection"""
    
    def __init__(self, enable_system_monitoring: bool = True):
        self.enable_system_monitoring = enable_system_monitoring
        self.processing_metrics: List[ProcessingMetrics] = []
        self.system_metrics = SystemMetrics()
        
        # Real-time monitoring
        self.resource_history = deque(maxlen=1000)  # Keep last 1000 measurements
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Alerting thresholds
        self.alert_thresholds = {
            'memory_mb': 1024,  # Alert if memory usage exceeds 1GB
            'processing_time': 15.0,  # Alert if processing time exceeds 15s
            'cpu_percent': 90.0,  # Alert if CPU usage exceeds 90%
            'failure_rate': 10.0  # Alert if failure rate exceeds 10%
        }
        
        self.alert_callbacks: List[Callable] = []
        
        if self.enable_system_monitoring:
            self.start_system_monitoring()
    
    def start_system_monitoring(self):
        """Start system resource monitoring thread"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_system_resources, daemon=True)
        self.monitoring_thread.start()
        logging.info("System monitoring started")
    
    def stop_system_monitoring(self):
        """Stop system resource monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        logging.info("System monitoring stopped")
    
    def _monitor_system_resources(self):
        """Monitor system resources in background thread"""
        while self.monitoring_active:
            try:
                # Get current resource usage
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=None)
                
                resource_data = {
                    'timestamp': datetime.now().isoformat(),
                    'memory_used_mb': memory_info.used / (1024 * 1024),
                    'memory_percent': memory_info.percent,
                    'cpu_percent': cpu_percent,
                    'available_memory_mb': memory_info.available / (1024 * 1024)
                }
                
                self.resource_history.append(resource_data)
                
                # Check for alerts
                self._check_resource_alerts(resource_data)
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                logging.error(f"Error in system monitoring: {e}")
                time.sleep(5.0)  # Wait longer on error
    
    def _check_resource_alerts(self, resource_data: Dict[str, Any]):
        """Check for resource usage alerts"""
        alerts = []
        
        if resource_data['memory_used_mb'] > self.alert_thresholds['memory_mb']:
            alerts.append({
                'type': 'high_memory',
                'message': f"Memory usage: {resource_data['memory_used_mb']:.1f}MB",
                'severity': 'warning'
            })
        
        if resource_data['cpu_percent'] > self.alert_thresholds['cpu_percent']:
            alerts.append({
                'type': 'high_cpu',
                'message': f"CPU usage: {resource_data['cpu_percent']:.1f}%",
                'severity': 'warning'
            })
        
        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logging.error(f"Error in alert callback: {e}")
    
    def start_processing_measurement(self, file_name: str) -> Dict[str, Any]:
        """Start measuring processing for a file"""
        
        # Get file info
        file_path = Path(file_name)
        file_size_mb = 0
        
        if file_path.exists():
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Get baseline resource usage
        process = psutil.Process()
        memory_info = process.memory_info()
        
        measurement_context = {
            'file_name': file_path.name,
            'file_size_mb': file_size_mb,
            'start_time': time.time(),
            'start_memory_mb': memory_info.rss / (1024 * 1024),
            'start_cpu_time': process.cpu_times().user + process.cpu_times().system,
            'process': process
        }
        
        return measurement_context
    
    def end_processing_measurement(self, measurement_context: Dict[str, Any], 
                                 result: Dict[str, Any], 
                                 success: bool = True, 
                                 error_message: Optional[str] = None) -> ProcessingMetrics:
        """End measurement and calculate metrics"""
        
        end_time = time.time()
        processing_time = end_time - measurement_context['start_time']
        
        # Get final resource usage
        process = measurement_context['process']
        memory_info = process.memory_info()
        end_memory_mb = memory_info.rss / (1024 * 1024)
        end_cpu_time = process.cpu_times().user + process.cpu_times().system
        
        # Calculate metrics
        memory_peak_mb = max(measurement_context['start_memory_mb'], end_memory_mb)
        cpu_time_used = end_cpu_time - measurement_context['start_cpu_time']
        cpu_usage_percent = (cpu_time_used / processing_time) * 100 if processing_time > 0 else 0
        
        # Extract result metrics
        blocks_extracted = result.get('blocks_extracted', 0)
        headings_detected = len(result.get('outline', []))
        confidence_avg = result.get('avg_confidence', 0.0)
        
        # Estimate page count from processing time (rough heuristic)
        page_count = max(1, int(processing_time * 5))  # Assume ~5 pages per second processing
        
        metrics = ProcessingMetrics(
            file_name=measurement_context['file_name'],
            file_size_mb=measurement_context['file_size_mb'],
            page_count=page_count,
            processing_time=processing_time,
            memory_peak_mb=memory_peak_mb,
            cpu_usage_percent=cpu_usage_percent,
            blocks_extracted=blocks_extracted,
            headings_detected=headings_detected,
            confidence_avg=confidence_avg,
            success=success,
            error_message=error_message
        )
        
        # Store metrics
        self.processing_metrics.append(metrics)
        self.system_metrics.update_with_processing_metrics(metrics)
        
        # Check for processing alerts
        self._check_processing_alerts(metrics)
        
        return metrics
    
    def _check_processing_alerts(self, metrics: ProcessingMetrics):
        """Check for processing-related alerts"""
        alerts = []
        
        if metrics.processing_time > self.alert_thresholds['processing_time']:
            alerts.append({
                'type': 'slow_processing',
                'message': f"File {metrics.file_name} took {metrics.processing_time:.2f}s",
                'severity': 'warning'
            })
        
        if not metrics.success:
            alerts.append({
                'type': 'processing_failure',
                'message': f"Failed to process {metrics.file_name}: {metrics.error_message}",
                'severity': 'error'
            })
        
        # Check failure rate
        if len(self.processing_metrics) >= 10:
            recent_failures = sum(1 for m in self.processing_metrics[-10:] if not m.success)
            failure_rate = (recent_failures / 10) * 100
            
            if failure_rate > self.alert_thresholds['failure_rate']:
                alerts.append({
                    'type': 'high_failure_rate',
                    'message': f"Failure rate: {failure_rate:.1f}% (last 10 files)",
                    'severity': 'error'
                })
        
        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logging.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return {
            'system_metrics': asdict(self.system_metrics),
            'recent_resource_usage': list(self.resource_history)[-10:] if self.resource_history else [],
            'processing_count': len(self.processing_metrics),
            'last_updated': datetime.now().isoformat()
        }
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        
        if not self.processing_metrics:
            return {'error': 'No processing metrics available'}
        
        # Calculate additional statistics
        processing_times = [m.processing_time for m in self.processing_metrics]
        memory_peaks = [m.memory_peak_mb for m in self.processing_metrics]
        file_sizes = [m.file_size_mb for m in self.processing_metrics]
        
        # Performance distribution
        time_percentiles = {
            'p50': float(np.percentile(processing_times, 50)),
            'p90': float(np.percentile(processing_times, 90)),
            'p95': float(np.percentile(processing_times, 95)),
            'p99': float(np.percentile(processing_times, 99))
        }
        
        # File size analysis
        size_analysis = {
            'avg_file_size_mb': float(np.mean(file_sizes)),
            'max_file_size_mb': float(max(file_sizes)),
            'min_file_size_mb': float(min(file_sizes))
        }
        
        # Error analysis
        errors = [m for m in self.processing_metrics if not m.success]
        error_analysis = {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(self.processing_metrics) * 100,
            'common_errors': {}
        }
        
        # Group errors by message
        error_counts = defaultdict(int)
        for error in errors:
            error_msg = error.error_message or 'Unknown error'
            error_counts[error_msg] += 1
        
        error_analysis['common_errors'] = dict(error_counts)
        
        # Resource usage analysis
        resource_analysis = {
            'avg_memory_peak_mb': float(np.mean(memory_peaks)),
            'max_memory_peak_mb': float(max(memory_peaks)),
            'memory_efficiency': size_analysis['avg_file_size_mb'] / float(np.mean(memory_peaks)) if np.mean(memory_peaks) > 0 else 0
        }
        
        # Compliance analysis
        compliance_analysis = {
            'hackathon_time_compliance': self.system_metrics.files_exceeding_time_limit / self.system_metrics.total_files_processed * 100,
            'memory_limit_compliance': 100.0 - (sum(1 for m in memory_peaks if m > 512) / len(memory_peaks) * 100),
            'success_rate': self.system_metrics.success_rate
        }
        
        return {
            'system_metrics': asdict(self.system_metrics),
            'performance_distribution': time_percentiles,
            'file_size_analysis': size_analysis,
            'error_analysis': error_analysis,
            'resource_analysis': resource_analysis,
            'compliance_analysis': compliance_analysis,
            'report_generated': datetime.now().isoformat(),
            'total_metrics': len(self.processing_metrics)
        }
    
    def export_metrics(self, file_path: str, format: str = 'json'):
        """Export metrics to file"""
        
        data = {
            'system_metrics': asdict(self.system_metrics),
            'processing_metrics': [asdict(m) for m in self.processing_metrics],
            'resource_history': list(self.resource_history),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if format.lower() == 'json':
                json.dump(data, f, indent=2, ensure_ascii=False)
            elif format.lower() == 'csv':
                # Export processing metrics as CSV
                import pandas as pd
                df = pd.DataFrame([asdict(m) for m in self.processing_metrics])
                df.to_csv(f, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        logging.info(f"Metrics exported to {file_path}")
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.processing_metrics.clear()
        self.system_metrics = SystemMetrics()
        self.resource_history.clear()
        logging.info("Metrics reset")

class EnhancedLogger:
    """Enhanced logging with structured logging and performance tracking"""
    
    def __init__(self, name: str, performance_monitor: Optional[PerformanceMonitor] = None):
        self.logger = logging.getLogger(name)
        self.performance_monitor = performance_monitor
        self.context_stack = []
    
    def push_context(self, **kwargs):
        """Push logging context"""
        self.context_stack.append(kwargs)
    
    def pop_context(self):
        """Pop logging context"""
        if self.context_stack:
            self.context_stack.pop()
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with context"""
        
        # Combine all context
        combined_context = {}
        for context in self.context_stack:
            combined_context.update(context)
        combined_context.update(kwargs)
        
        if combined_context:
            context_str = " | ".join(f"{k}={v}" for k, v in combined_context.items())
            return f"{message} | {context_str}"
        
        return message
    
    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message with context"""
        self.logger.error(self._format_message(message, **kwargs))
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def performance(self, message: str, duration: float, **kwargs):
        """Log performance message"""
        self.info(f"PERF: {message}", duration=f"{duration:.3f}s", **kwargs)
    
    def metric(self, metric_name: str, value: Any, **kwargs):
        """Log metric"""
        self.info(f"METRIC: {metric_name}={value}", **kwargs)

# Global monitoring instance
_global_monitor: Optional[PerformanceMonitor] = None

def get_global_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor

def log_alert_to_console(alert: Dict[str, Any]):
    """Default alert handler that logs to console"""
    severity = alert.get('severity', 'info').upper()
    message = alert.get('message', 'Unknown alert')
    alert_type = alert.get('type', 'unknown')
    
    log_message = f"ALERT [{severity}] {alert_type}: {message}"
    
    if severity == 'ERROR':
        logging.error(log_message)
    elif severity == 'WARNING':
        logging.warning(log_message)
    else:
        logging.info(log_message)

# Import numpy for percentile calculations
try:
    import numpy as np
except ImportError:
    # Fallback implementation
    def percentile(data, p):
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[-1]
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
    
    class np:
        @staticmethod
        def percentile(data, p):
            return percentile(data, p)
        
        @staticmethod
        def mean(data):
            return sum(data) / len(data)
        
        @staticmethod
        def max(data):
            return max(data)
        
        @staticmethod
        def min(data):
            return min(data)

def main():
    """Example usage of monitoring system"""
    
    # Initialize monitoring
    monitor = PerformanceMonitor()
    monitor.add_alert_callback(log_alert_to_console)
    
    logger = EnhancedLogger("pdf_processor", monitor)
    
    # Simulate processing
    logger.push_context(operation="test_processing")
    
    for i in range(5):
        file_name = f"test_{i}.pdf"
        
        # Start measurement
        measurement = monitor.start_processing_measurement(file_name)
        
        logger.info(f"Processing {file_name}")
        
        # Simulate processing
        time.sleep(0.5)
        
        # End measurement
        result = {
            'outline': [{'level': 'H1', 'text': 'Test', 'page': 1}],
            'blocks_extracted': 10,
            'avg_confidence': 0.8
        }
        
        metrics = monitor.end_processing_measurement(measurement, result, success=True)
        
        logger.performance(f"Processed {file_name}", metrics.processing_time)
    
    logger.pop_context()
    
    # Get report
    report = monitor.get_detailed_report()
    print(json.dumps(report, indent=2))
    
    # Clean up
    monitor.stop_system_monitoring()

if __name__ == "__main__":
    main()