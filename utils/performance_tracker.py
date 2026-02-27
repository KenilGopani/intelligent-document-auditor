import time
from typing import Dict, Any
from contextlib import contextmanager


class PerformanceTracker:
    """Utility class for tracking performance metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    @contextmanager
    def track_time(self, operation_name: str):
        """
        Context manager to track execution time of operations
        
        Args:
            operation_name: Name of the operation to track
        """
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.metrics[operation_name] = elapsed_time
    
    def get_metric(self, operation_name: str) -> float:
        """
        Get timing metric for an operation
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            float: Elapsed time in seconds
        """
        return self.metrics.get(operation_name, 0.0)
    
    def get_all_metrics(self) -> Dict[str, float]:
        """
        Get all recorded metrics
        
        Returns:
            Dict[str, float]: Dictionary of all metrics
        """
        return self.metrics.copy()
    
    def format_time(self, seconds: float) -> str:
        """
        Format time in human-readable format
        
        Args:
            seconds: Time in seconds
            
        Returns:
            str: Formatted time string
        """
        if seconds < 1:
            return f"{seconds*1000:.1f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"


class QueryResult:
    """Container for query results with metadata"""
    
    def __init__(self, response: str, source_nodes: list, execution_time: float):
        self.response = response
        self.source_nodes = source_nodes
        self.execution_time = execution_time
        self.timestamp = time.time()
    
    def get_source_text(self) -> str:
        """Extract text content from source nodes"""
        if not self.source_nodes:
            return "No sources retrieved"
        
        texts = []
        for node in self.source_nodes:
            if hasattr(node, 'text'):
                texts.append(node.text)
            elif hasattr(node, 'get_content'):
                texts.append(node.get_content())
            else:
                texts.append(str(node))
        
        return "\n---\n".join(texts[:3])  # Limit to first 3 sources for readability