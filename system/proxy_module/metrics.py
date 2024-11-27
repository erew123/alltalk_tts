import logging
import threading
import time
from typing import Dict, Any
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

@dataclass
class ProxyMetrics:
    start_time: float = field(default_factory=time.time)
    total_requests: int = 0
    active_connections: int = 0
    bytes_transferred: int = 0
    request_times: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    requests_per_minute: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    status_codes: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

class MetricsCollector:
    def __init__(self, proxy_manager):
        self.proxy_manager = proxy_manager
        self.logger = logging.getLogger("ProxyMetrics")
        self.metrics = ProxyMetrics()
        self.collecting = False
        self.collector_thread = None
        self.collect_interval = 60  # seconds
        self._lock = threading.Lock()

    def start_collecting(self):
        """Start metrics collection"""
        if self.collecting:
            return
            
        self.collecting = True
        self.metrics = ProxyMetrics()  # Reset metrics
        self.collector_thread = threading.Thread(target=self._collect_loop)
        self.collector_thread.daemon = True
        self.collector_thread.start()
        self.logger.info("Metrics collection started")

    def stop_collecting(self):
        """Stop metrics collection"""
        self.collecting = False
        if self.collector_thread:
            self.collector_thread.join()
        self.logger.info("Metrics collection stopped")

    def _collect_loop(self):
        """Main collection loop"""
        while self.collecting:
            self._update_metrics()
            time.sleep(self.collect_interval)

    def _update_metrics(self):
        """Update periodic metrics"""
        with self._lock:
            # Reset per-minute counters
            self.metrics.requests_per_minute.clear()

    def record_request(self, endpoint: str, status_code: int, response_time: float, bytes_sent: int):
        """Record metrics for a single request"""
        with self._lock:
            self.metrics.total_requests += 1
            self.metrics.bytes_transferred += bytes_sent
            self.metrics.request_times[endpoint] = response_time
            self.metrics.status_codes[str(status_code)] += 1
            self.metrics.requests_per_minute[endpoint] += 1

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        with self._lock:
            uptime = time.time() - self.metrics.start_time
            return {
                "uptime": str(timedelta(seconds=int(uptime))),
                "total_requests": self.metrics.total_requests,
                "active_connections": self.metrics.active_connections,
                "bytes_transferred": f"{self.metrics.bytes_transferred / 1024 / 1024:.2f} MB",
                "average_response_times": {
                    endpoint: f"{time:.2f}ms" 
                    for endpoint, time in self.metrics.request_times.items()
                },
                "requests_per_minute": dict(self.metrics.requests_per_minute),
                "status_codes": dict(self.metrics.status_codes)
            }

    def increment_connections(self):
        """Increment active connection count"""
        with self._lock:
            self.metrics.active_connections += 1

    def decrement_connections(self):
        """Decrement active connection count"""
        with self._lock:
            self.metrics.active_connections = max(0, self.metrics.active_connections - 1)