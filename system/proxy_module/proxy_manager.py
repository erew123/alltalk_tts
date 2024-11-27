import os
import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import asdict
import psutil
from logging.handlers import RotatingFileHandler
import logging
import os
from .security import SecurityManager
from .health_monitor import HealthMonitor
from .cert_manager import CertificateManager
from .metrics import MetricsCollector

class ProxyManager:
    def __init__(self, config_manager, base_path: str = None):
        self.config_manager = config_manager
        # Set up base paths first
        self.base_path = Path(base_path) if base_path else Path(__file__).parent
        
        # Create required directories
        self.certs_path = self.base_path / "certificates"
        self.logs_path = self.base_path / "logs"
        self.config_path = self.base_path / "config"
        
        for path in [self.certs_path, self.logs_path, self.config_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging before managers
        self.setup_logging()
        
        # Initialize managers after paths are created
        self.security = SecurityManager(self)
        self.health_monitor = HealthMonitor(self)
        self.cert_manager = CertificateManager(self)
        self.metrics = MetricsCollector(self)
        
        self.proxy_process = None
        self.logger.info("Proxy manager initialized successfully")

    def setup_logging(self):
        """Setup logging configuration with customized format"""
        config = self.config_manager.get_instance()
        log_level = getattr(logging, config.proxy_settings.log_level, logging.INFO)
        
        # Configure rotating file handler
        log_file = self.logs_path / "proxy.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        
        # Configure logging formats
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_format = logging.Formatter(
            '[AllTalk PRX] %(message)s'
        )
        
        file_handler.setFormatter(file_format)
        
        # Setup logger
        self.logger = logging.getLogger("ProxyManager")
        self.logger.setLevel(log_level)
        self.logger.addHandler(file_handler)
        
        # Only add console handler if proxy debugging is enabled
        if config.debugging.debug_proxy:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)
            self.logger.info("Proxy debug logging enabled")

    def _monitor_output(self, pipe, name):
        """Monitor subprocess output and redirect to logger"""
        try:
            for line in iter(pipe.readline, ''):
                line = line.strip()
                if line:
                    if name == "stderr":
                        self.logger.error(line)
                    else:
                        # Pass through with our logging format
                        print(line)  # Direct print for formatted output
        except Exception as e:
            self.logger.error(f"Error monitoring {name}: {e}")
        finally:
            pipe.close()

    def start_proxy(self) -> bool:
        """Start the proxy server with current configuration"""
        if self.proxy_process and self.proxy_process.poll() is None:
            self.logger.warning("Proxy already running")
            return False

        config = self.config_manager.get_instance()
        proxy_settings = config.proxy_settings

        try:

            cmd = [
                "python", "-m", "twisted_server",
                "--external_ip_a", proxy_settings.api_endpoint.external_ip if proxy_settings.api_endpoint.enabled == "Enabled" else "0",
                "--internal_ip_a", "127.0.0.1",
                "--external_ip_b", proxy_settings.gradio_endpoint.external_ip if proxy_settings.gradio_endpoint.enabled == "Enabled" else "0",
                "--internal_ip_b", "127.0.0.1",
                "--cert_file_a", proxy_settings.api_endpoint.cert_name or "0",
                "--cert_file_b", proxy_settings.gradio_endpoint.cert_name or "0"
            ]            
            
            # Start process with pipe for output
            self.proxy_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.base_path),
                text=True,
                bufsize=1,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'},  # Force unbuffered output
                universal_newlines=True
            )
            
            # Start output monitoring threads
            threading.Thread(target=self._monitor_output, 
                           args=(self.proxy_process.stdout, "stdout"), 
                           daemon=True).start()
            threading.Thread(target=self._monitor_output, 
                           args=(self.proxy_process.stderr, "stderr"), 
                           daemon=True).start()
            
            if proxy_settings.proxy_enabled:
                self.health_monitor.start_monitoring()
                self.metrics.start_collecting()
            
            self.logger.info("Proxy server started successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start proxy: {e}")
            return False

    def stop_proxy(self) -> bool:
        """Stop the proxy server and all monitoring"""
        if not self.proxy_process:
            self.logger.warning("No proxy process running")
            return False

        try:
            # Stop all monitoring first
            self.health_monitor.stop_monitoring()
            self.metrics.stop_collecting()
            
            # Stop the main process
            process = psutil.Process(self.proxy_process.pid)
            for proc in process.children(recursive=True):
                proc.terminate()
            process.terminate()
            
            try:
                process.wait(timeout=5)
            except psutil.TimeoutExpired:
                process.kill()
            
            self.proxy_process = None
            self.logger.info("Proxy server stopped successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop proxy: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the proxy system"""
        config = self.config_manager.get_instance()
        proxy_config = config.proxy_settings
        
        status = {
            "running": self.proxy_process is not None and self.proxy_process.poll() is None,
            "config": {
                "gradio_endpoint": asdict(proxy_config.gradio_endpoint),
                "api_endpoint": asdict(proxy_config.api_endpoint),
                "start_on_startup": proxy_config.start_on_startup
            },
            "certificates": self.cert_manager.get_certificates_status(),
            "health": self.health_monitor.get_health_status(),
            "metrics": self.metrics.get_current_metrics(),
            "security": self.security.get_security_status()
        }
        
        return status

    def handle_cert_upload(self, cert_file, key_file, name):
        """Handle certificate upload from the interface"""
        if not cert_file or not key_file or not name:
            return "Please provide both certificate and key files, and a name"
        
        cert_path = Path(cert_file.name)
        key_path = Path(key_file.name)
        
        success = self.cert_manager.install_certificate(cert_path, key_path, name)
        return "Certificate uploaded successfully" if success else "Failed to upload certificate"

    def handle_start(self):
        """Handle proxy start from the interface"""
        success = self.start_proxy()
        return "Proxy Server: Starting..." if success else "Failed to start proxy server"

    def handle_stop(self):
        """Handle proxy stop from the interface"""
        success = self.stop_proxy()
        return "Proxy Server: Stopped" if success else "Failed to stop proxy server"

    def handle_status(self):
        """Handle status request from the interface"""
        status = self.get_status()
        if status['running']:
            return "Proxy Server: Running\n" + \
                f"API Endpoint: {'Active' if status['config']['api_endpoint']['enabled'] == 'Enabled' else 'Disabled'}\n" + \
                f"Gradio Endpoint: {'Active' if status['config']['gradio_endpoint']['enabled'] == 'Enabled' else 'Disabled'}\n" + \
                f"Certificates: {len(status['certificates'])} configured"
        else:
            return "Proxy Server: Stopped"

    def handle_security_event(self, event_type: str, details: Dict[str, Any]):
        """Handle security events from the SecurityManager"""
        self.logger.warning(f"Security event: {event_type} - {details}")
        # Implement security response logic here

    def cleanup(self):
        """Cleanup resources before shutdown"""
        self.stop_proxy()
        self.logger.info("Proxy manager cleanup completed")