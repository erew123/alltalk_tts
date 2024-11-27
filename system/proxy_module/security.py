import logging
from typing import Dict, Any, Set
import time
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock

@dataclass
class SecurityMetrics:
    total_requests: int = 0
    blocked_requests: int = 0
    last_incident: float = 0
    blacklisted_ips: Set[str] = None
    
    def __post_init__(self):
        if self.blacklisted_ips is None:
            self.blacklisted_ips = set()

class SecurityManager:
    def __init__(self, proxy_manager):
        self.proxy_manager = proxy_manager
        self.logger = logging.getLogger("ProxySecurityManager")
        self.metrics = SecurityMetrics()
        self.rate_limits = defaultdict(lambda: {"count": 0, "reset_time": 0})
        self.rate_limit_lock = Lock()
        
    def check_rate_limit(self, ip: str, limit: int = 100, window: int = 60) -> bool:
        """Check if IP has exceeded rate limit"""
        with self.rate_limit_lock:
            current_time = time.time()
            if self.rate_limits[ip]["reset_time"] < current_time:
                self.rate_limits[ip] = {"count": 0, "reset_time": current_time + window}
            
            self.rate_limits[ip]["count"] += 1
            return self.rate_limits[ip]["count"] <= limit

    def is_blacklisted(self, ip: str) -> bool:
        """Check if IP is blacklisted"""
        return ip in self.metrics.blacklisted_ips

    def add_to_blacklist(self, ip: str, reason: str):
        """Add IP to blacklist"""
        self.metrics.blacklisted_ips.add(ip)
        self.logger.warning(f"IP {ip} blacklisted: {reason}")
        self.metrics.last_incident = time.time()

    def remove_from_blacklist(self, ip: str):
        """Remove IP from blacklist"""
        self.metrics.blacklisted_ips.discard(ip)
        self.logger.info(f"IP {ip} removed from blacklist")

    def get_security_status(self) -> Dict[str, Any]:
        """Get current security metrics"""
        return {
            "total_requests": self.metrics.total_requests,
            "blocked_requests": self.metrics.blocked_requests,
            "blacklisted_ips": len(self.metrics.blacklisted_ips),
            "last_incident": time.strftime('%Y-%m-%d %H:%M:%S', 
                                         time.localtime(self.metrics.last_incident)) 
                            if self.metrics.last_incident else "None"
        }