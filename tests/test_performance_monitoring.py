"""
Test Performance Monitoring
Simple test script to verify performance monitoring functionality.
"""

import time
import psutil
import os
import numpy as np
import pandas as pd

class PerformanceMonitor:
    """Monitor and track computational performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.process = psutil.Process(os.getpid())
        
    def start_monitoring(self, model_name: str, entity_name: str):
        """Start monitoring for a specific model."""
        key = f"{entity_name}_{model_name}"
        self.metrics[key] = {
            'start_time': time.time(),
            'start_memory': self.process.memory_info().rss / 1024 / 1024,  # MB
            'start_cpu_percent': self.process.cpu_percent(),
            'peak_memory': 0,
            'peak_cpu_percent': 0,
            'total_memory_used': 0,
            'memory_samples': [],
            'cpu_samples': []
        }
        print(f"Started monitoring {model_name} for {entity_name}")
        
    def update_monitoring(self, model_name: str, entity_name: str):
        """Update monitoring metrics during training."""
        key = f"{entity_name}_{model_name}"
        if key in self.metrics:
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            current_cpu = self.process.cpu_percent()
            
            self.metrics[key]['memory_samples'].append(current_memory)
            self.metrics[key]['cpu_samples'].append(current_cpu)
            
            self.metrics[key]['peak_memory'] = max(self.metrics[key]['peak_memory'], current_memory)
            self.metrics[key]['peak_cpu_percent'] = max(self.metrics[key]['peak_cpu_percent'], current_cpu)
    
    def stop_monitoring(self, model_name: str, entity_name: str):
        """Stop monitoring and return performance metrics."""
        key = f"{entity_name}_{model_name}"
        if key in self.metrics:
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            metrics = self.metrics[key]
            metrics['end_time'] = end_time
            metrics['end_memory'] = end_memory
            metrics['execution_time'] = end_time - metrics['start_time']
            metrics['memory_increase'] = end_memory - metrics['start_memory']
            metrics['total_memory_used'] = end_memory - metrics['start_memory']
            
            # Calculate average memory and CPU usage
            if metrics['memory_samples']:
                metrics['avg_memory'] = np.mean(metrics['memory_samples'])
                metrics['avg_cpu_percent'] = np.mean(metrics['cpu_samples'])
            
            # Estimate computational cost (rough approximation)
            # CPU seconds = execution_time * avg_cpu_percent / 100
            metrics['cpu_seconds'] = metrics['execution_time'] * metrics.get('avg_cpu_percent', 0) / 100
            
            # Memory cost (MB-seconds)
            metrics['memory_seconds'] = metrics['avg_memory'] * metrics['execution_time']
            
            print(f"Performance for {model_name} ({entity_name}):")
            print(f"  Execution time: {metrics['execution_time']:.2f}s")
            print(f"  Peak memory: {metrics['peak_memory']:.1f}MB")
            print(f"  Peak CPU: {metrics['peak_cpu_percent']:.1f}%")
            print(f"  Memory increase: {metrics['memory_increase']:.1f}MB")
            
            return metrics
        return {}

def test_performance_monitoring():
    """Test the performance monitoring functionality."""
    print("Testing Performance Monitoring...")
    
    # Initialize monitor
    monitor = PerformanceMonitor()
    
    # Test 1: Simple computation
    print("\nTest 1: Simple computation")
    monitor.start_monitoring('test_model', 'test_entity')
    
    # Simulate some computation
    data = np.random.rand(1000, 1000)
    for i in range(10):
        result = np.dot(data, data.T)
        monitor.update_monitoring('test_model', 'test_entity')
        time.sleep(0.1)  # Simulate processing time
    
    metrics = monitor.stop_monitoring('test_model', 'test_entity')
    
    # Test 2: Memory-intensive operation
    print("\nTest 2: Memory-intensive operation")
    monitor.start_monitoring('memory_test', 'test_entity')
    
    # Create large arrays
    large_data = []
    for i in range(5):
        large_data.append(np.random.rand(1000, 1000))
        monitor.update_monitoring('memory_test', 'test_entity')
        time.sleep(0.2)
    
    metrics2 = monitor.stop_monitoring('memory_test', 'test_entity')
    
    # Test 3: CPU-intensive operation
    print("\nTest 3: CPU-intensive operation")
    monitor.start_monitoring('cpu_test', 'test_entity')
    
    # Simulate CPU-intensive computation
    for i in range(20):
        # Complex computation
        result = sum([j**2 for j in range(10000)])
        monitor.update_monitoring('cpu_test', 'test_entity')
    
    metrics3 = monitor.stop_monitoring('cpu_test', 'test_entity')
    
    # Summary
    print("\n" + "="*50)
    print("PERFORMANCE MONITORING TEST SUMMARY")
    print("="*50)
    
    all_metrics = [metrics, metrics2, metrics3]
    test_names = ['Simple Computation', 'Memory Test', 'CPU Test']
    
    for i, (test_name, metric) in enumerate(zip(test_names, all_metrics)):
        if metric:
            print(f"\n{test_name}:")
            print(f"  Execution Time: {metric['execution_time']:.3f}s")
            print(f"  Peak Memory: {metric['peak_memory']:.1f}MB")
            print(f"  Peak CPU: {metric['peak_cpu_percent']:.1f}%")
            print(f"  Memory Increase: {metric['memory_increase']:.1f}MB")
            print(f"  CPU Seconds: {metric.get('cpu_seconds', 0):.3f}")
            print(f"  Memory Seconds: {metric.get('memory_seconds', 0):.1f}")
    
    print("\nPerformance monitoring test completed successfully!")
    return True

if __name__ == "__main__":
    test_performance_monitoring() 