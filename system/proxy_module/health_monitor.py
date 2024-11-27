import logging
import threading
import time
from typing import Dict, Any
import requests
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class HealthMetrics:
    last_check: float = 0
    uptime_start: float = 0
    gradio_endpoint_up: bool = False
    api_endpoint_up: bool = False
    consecutive_failures: int = 0
    total_restarts: int = 0
    response_times: Dict[str, float] = field(default_factory=dict)

class HealthMonitor:
    def __init__(self, proxy_manager):
        self.proxy_manager = proxy_manager
        self.logger = logging.getLogger("ProxyHealthMonitor")
        self.metrics = HealthMetrics()
        self.monitoring = False
        self.monitor_thread = None
        self.check_interval = 30  # seconds
        
    def start_monitoring(self):
        """Start health monitoring thread"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.metrics.uptime_start = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)  # Make thread daemon
        self.monitor_thread.start()
        self.logger.info("Health monitoring started")

    def stop_monitoring(self):
        """Stop health monitoring thread"""
        if not self.monitoring:
            return
            
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            try:
                # Give the thread a chance to finish naturally
                self.monitor_thread.join(timeout=5)
            except Exception as e:
                self.logger.warning(f"Could not gracefully stop monitoring thread: {e}")
            finally:
                self.monitor_thread = None
        
        self.logger.info("Health monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            self._check_health()
            time.sleep(self.check_interval)

    def _check_health(self):
        """Check health of proxy endpoints"""
        config = self.proxy_manager.config_manager.get_instance()
        proxy_settings = config.proxy_settings
        self.metrics.last_check = time.time()
        
        try:
            # Only check endpoints that are enabled and running
            if proxy_settings.gradio_endpoint.enabled == "Enabled":
                try:
                    start_time = time.time()
                    response = requests.get(
                        f"http://127.0.0.1:{config.gradio_port_number}/",  # Use internal port
                        timeout=5
                    )
                    self.metrics.response_times['gradio'] = time.time() - start_time
                    self.metrics.gradio_endpoint_up = response.status_code == 200
                except Exception as e:
                    self.logger.debug(f"Gradio endpoint check failed: {e}")
                    self.metrics.gradio_endpoint_up = False

            if proxy_settings.api_endpoint.enabled == "Enabled":
                try:
                    start_time = time.time()
                    response = requests.get(
                        f"http://127.0.0.1:{config.api_def.api_port_number}/",  # Use internal port
                        timeout=5
                    )
                    self.metrics.response_times['api'] = time.time() - start_time
                    self.metrics.api_endpoint_up = response.status_code == 200
                except Exception as e:
                    self.logger.debug(f"API endpoint check failed: {e}")
                    self.metrics.api_endpoint_up = False

            if not self.metrics.gradio_endpoint_up and not self.metrics.api_endpoint_up:
                self.metrics.consecutive_failures += 1
            else:
                self.metrics.consecutive_failures = 0
                
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            self.metrics.consecutive_failures += 1
        
        # Only attempt restart if we have multiple complete failures
        if self.metrics.consecutive_failures >= 3:
            self.logger.warning("Too many consecutive failures, attempting restart")
            try:
                if self.proxy_manager.stop_proxy() and self.proxy_manager.start_proxy():
                    self.metrics.total_restarts += 1
                    self.metrics.consecutive_failures = 0
            except Exception as e:
                self.logger.error(f"Failed to restart proxy: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health metrics"""
        uptime = time.time() - self.metrics.uptime_start if self.metrics.uptime_start else 0
        return {
            "status": "healthy" if not self.metrics.consecutive_failures else "degraded",
            "uptime": f"{int(uptime/3600)}h {int((uptime%3600)/60)}m",
            "last_check": datetime.fromtimestamp(self.metrics.last_check).strftime('%Y-%m-%d %H:%M:%S') 
                         if self.metrics.last_check else "Never",
            "gradio_endpoint": "up" if self.metrics.gradio_endpoint_up else "down",
            "api_endpoint": "up" if self.metrics.api_endpoint_up else "down",
            "response_times": self.metrics.response_times,
            "total_restarts": self.metrics.total_restarts
        }