import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import ssl
import OpenSSL
from dataclasses import dataclass

@dataclass
class CertificateInfo:
    subject: str
    issuer: str
    valid_from: datetime
    valid_until: datetime
    serial_number: str
    is_valid: bool

class CertificateManager:
    def __init__(self, proxy_manager):
        self.proxy_manager = proxy_manager
        self.logger = logging.getLogger("ProxyCertManager")
        self.certs_path = proxy_manager.certs_path

    def validate_certificate(self, cert_path: Path) -> Optional[CertificateInfo]:
        """Validate and get information about a certificate"""
        try:
            with open(cert_path, 'rb') as cert_file:
                cert_data = cert_file.read()
                x509 = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, cert_data)
                
                valid_from = datetime.strptime(x509.get_notBefore().decode(), '%Y%m%d%H%M%SZ')
                valid_until = datetime.strptime(x509.get_notAfter().decode(), '%Y%m%d%H%M%SZ')
                now = datetime.now()
                
                return CertificateInfo(
                    subject=x509.get_subject().CN,
                    issuer=x509.get_issuer().CN,
                    valid_from=valid_from,
                    valid_until=valid_until,
                    serial_number=str(x509.get_serial_number()),
                    is_valid=valid_from <= now <= valid_until
                )
        except Exception as e:
            self.logger.error(f"Certificate validation error: {e}")
            return None

    def install_certificate(self, cert_path: Path, key_path: Path, name: str) -> bool:
        try:
            import shutil
            # Validate certificate first
            cert_info = self.validate_certificate(cert_path)
            if not cert_info or not cert_info.is_valid:
                self.logger.error("Invalid or corrupt certificate")
                return False

            cert_dest = self.certs_path / f"{name}_cert.pem"
            key_dest = self.certs_path / f"{name}_key.pem"
            
            shutil.copy2(cert_path, cert_dest)
            shutil.copy2(key_path, key_dest)
            
            # Update config
            config = self.proxy_manager.config_manager.get_instance()
            config.proxy_settings.gradio_endpoint.cert_name = name
            config.proxy_settings.api_endpoint.cert_name = name
            config.save()
            
            self.logger.info(f"Certificate '{name}' installed successfully")
            return True
                
        except Exception as e:
            self.logger.error(f"Certificate installation error: {e}")
            return False

    def remove_certificate(self, name: str) -> bool:
        """Remove a certificate"""
        try:
            cert_path = self.certs_path / f"{name}_cert.pem"
            key_path = self.certs_path / f"{name}_key.pem"
            
            if cert_path.exists():
                cert_path.unlink()
            if key_path.exists():
                key_path.unlink()
                
            self.logger.info(f"Certificate '{name}' removed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Certificate removal error: {e}")
            return False

    def get_certificates_status(self) -> Dict[str, Any]:
        """Get status of all installed certificates"""
        certificates = {}
        for cert_file in self.certs_path.glob("*_cert.pem"):
            name = cert_file.name.replace("_cert.pem", "")
            cert_info = self.validate_certificate(cert_file)
            if cert_info:
                certificates[name] = {
                    "subject": cert_info.subject,
                    "issuer": cert_info.issuer,
                    "valid_from": cert_info.valid_from.strftime('%Y-%m-%d %H:%M:%S'),
                    "valid_until": cert_info.valid_until.strftime('%Y-%m-%d %H:%M:%S'),
                    "is_valid": cert_info.is_valid,
                    "days_until_expiry": (cert_info.valid_until - datetime.now()).days
                }
        return certificates

    def check_expiring_certificates(self, days_warning: int = 30) -> Dict[str, int]:
        """Check for certificates nearing expiration"""
        expiring = {}
        for cert_file in self.certs_path.glob("*_cert.pem"):
            name = cert_file.name.replace("_cert.pem", "")
            cert_info = self.validate_certificate(cert_file)
            if cert_info:
                days_left = (cert_info.valid_until - datetime.now()).days
                if days_left < days_warning:
                    expiring[name] = days_left
        return expiring