import subprocess
import sys
from pathlib import Path
import logging

# Auto-install requirements if missing
def install_requirements():
    """Install missing requirements only if they're not already in sys.modules"""
    required_packages = [
        "twisted",
        "cryptography", 
        "service_identity",
        "requests",
        "psutil"
    ]
    
    for package in required_packages:
        if package not in sys.modules:
            try:
                __import__(package)
            except ImportError:
                print(f"Package '{package}' is not installed. Installing now...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_requirements()

from twisted.internet import reactor, ssl
from twisted.web import proxy, server
import argparse
import inspect

# Handle config import for both package and standalone usage
try:
    from ...config import AlltalkConfig  # Package import
except ImportError:
    # Get the path to the root directory (3 levels up from this file)
    root_dir = Path(__file__).parent.parent.parent
    sys.path.append(str(root_dir))
    from config import AlltalkConfig  # Standalone import

def initialize_configs():
    """Initialize all configuration instances"""
    config_initalize = AlltalkConfig.get_instance()
    return config_initalize

# Load in configs
central_config = initialize_configs()

##########################
# Central print function #
##########################
# ANSI color codes
BLUE = "\033[94m"
# MAGENTA = "\033[95m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

def print_message(message, message_type="standard", component="PRX"):
    """Centralized print function for AllTalk messages
    Args:
        message (str): The message to print
        message_type (str): Type of message (standard/warning/error/debug_*/debug)
        component (str): Component identifier (TTS/ENG/GEN/API/etc.)
    """
    prefix = f"[{central_config.branding}{component}] "

    if message_type.startswith("debug_"):
        debug_flag = getattr(central_config.debugging, message_type, False)
        if not debug_flag:
            return

        if message_type == "debug_func" and "Function entry:" in message:
            message_parts = message.split("Function entry:", 1)
            print(
                f"{prefix}{BLUE}Debug{RESET} {YELLOW}{message_type}{RESET} Function entry:{GREEN}{message_parts[1]}{RESET} twisted_server.py"
            )
        else:
            print(f"{prefix}{BLUE}Debug{RESET} {YELLOW}{message_type}{RESET} {message}")

    elif message_type == "debug":
        print(f"{prefix}{BLUE}Debug{RESET} {message}")

    elif message_type == "warning":
        print(f"{prefix}{YELLOW}Warning{RESET} {message}")

    elif message_type == "error":
        print(f"{prefix}{RED}Error{RESET} {message}")

    else:
        print(f"{prefix}{message}")

def debug_func_entry():
    """Print debug message for function entry if debug_func is enabled"""
    if central_config.debugging.debug_func:
        current_func = inspect.currentframe().f_back.f_code.co_name
        print_message(f"Function entry: {current_func}", "debug_func")

class ProxyToInternalPort(proxy.ReverseProxyResource):
    def __init__(self, target_host, target_port):
        super().__init__(target_host, target_port, b"")
        self.logger = logging.getLogger("ProxyManager")

    def proxyClientConnectionFailed(self, connector, reason):
        print_message(f"Connection failed: {reason}", message_type="debug_proxy")
        return proxy.ReverseProxyResource.proxyClientConnectionFailed(self, connector, reason)

    def proxyClientConnectionMade(self, connector):
        print_message(f"Connection established to {self.host}:{self.port}", message_type="debug_proxy")
        return proxy.ReverseProxyResource.proxyClientConnectionMade(self, connector)

    def getChild(self, path, request):
        try:
            decoded_path = path.decode('utf-8', 'ignore')
            client_addr = request.getClientAddress()
            ip = client_addr.host if hasattr(client_addr, 'host') else 'unknown'
            method = request.method.decode('utf-8', 'ignore')
            
            print_message(f"{GREEN}Request: {RED}{method} {decoded_path}{GREEN} from {RESET}{ip} {GREEN}Forwarding to:{RESET} {self.host}:{self.port} {RED}{decoded_path}{RESET}", message_type="debug_proxy")
            #print_message(f"Forwarding to: {self.host}:{self.port}{decoded_path}", message_type="debug_proxy")
        except Exception as e:
            print_message(f"Error in getChild: {str(e)}", message_type="error")
        
        child = proxy.ReverseProxyResource(self.host, self.port, self.path + b'/' + path)
        return child

def check_certificates(cert_file_base):
    """Check if both key and cert files exist for the given certificate base name."""
    debug_func_entry()
    if cert_file_base != "0":
        cert_path = Path("certificates")  # Add base path
        key_file = cert_path / f"{cert_file_base}_key.pem"
        cert_file = cert_path / f"{cert_file_base}_cert.pem"
        
        if not key_file.exists() or not cert_file.exists():
            return False
        return True
    return False

def run_proxy(external_ip_a, internal_ip_a, external_ip_b, internal_ip_b, cert_file_a, cert_file_b):
    debug_func_entry()
    proxies_started = False
    api_port = central_config.api_def.api_port_number      
    gradio_port = central_config.gradio_port_number
    api_port_external = central_config.proxy_settings.api_endpoint.external_port    
    gradio_port_external = central_config.proxy_settings.gradio_endpoint.external_port 

    print_message("Starting proxy setup with the following parameters:", message_type="debug_proxy")
    print_message(f" |- External IP API    : {'[DISABLED]' if external_ip_a == '0' else f'{external_ip_a}:{api_port_external}'} -> {internal_ip_a}:{api_port}", message_type="debug_proxy")
    print_message(f" |- External IP Gradio : {'[DISABLED]' if external_ip_b == '0' else f'{external_ip_b}:{gradio_port_external}'} -> {internal_ip_b}:{gradio_port}", message_type="debug_proxy") 
    print_message(f" |- Certificate File A : {cert_file_a} {'[No Certificate So Disabled]' if cert_file_a == '0' else ''}", message_type="debug_proxy")
    print_message(f" |- Certificate File B : {cert_file_b} {'[No Certificate So Disabled]' if cert_file_b == '0' else ''}", message_type="debug_proxy")

    # Gradio Proxy
    if external_ip_b != "0" and central_config.proxy_settings.gradio_endpoint.enabled == "Enabled":        
        if cert_file_b != "0" and check_certificates(cert_file_b):
            sslContextB = ssl.DefaultOpenSSLContextFactory(
                f"certificates/{cert_file_b}_key.pem", 
                f"certificates/{cert_file_b}_cert.pem" 
            )
            proxy_gradio = ProxyToInternalPort(internal_ip_b, gradio_port)
            proxy_site_gradio = server.Site(proxy_gradio)
            reactor.listenSSL(gradio_port_external, proxy_site_gradio, sslContextB, interface=external_ip_b)
            print_message(f"Gradio SSL Proxy : {GREEN}From{RESET} {external_ip_b}:{gradio_port_external} {GREEN}->{RESET} {internal_ip_b}:{gradio_port} {GREEN}is now active.{RESET}")
        else:
            proxy_gradio = ProxyToInternalPort(internal_ip_b, gradio_port)
            proxy_site_gradio = server.Site(proxy_gradio)
            reactor.listenTCP(gradio_port_external, proxy_site_gradio, interface=external_ip_b)
            print_message(f"Gradio HTTP Proxy : {GREEN}From{RESET} {external_ip_b}:{gradio_port_external} {GREEN}->{RESET} {internal_ip_b}:{gradio_port} {GREEN}is now active.{RESET}")
        proxies_started = True
    else:
        print_message(f"Gradio Proxy is {RED}DISABLED{RESET}")

    # API Proxy
    if external_ip_a != "0" and central_config.proxy_settings.api_endpoint.enabled == "Enabled":        
        if cert_file_a != "0" and check_certificates(cert_file_a):
            sslContextA = ssl.DefaultOpenSSLContextFactory(
                f"certificates/{cert_file_a}_key.pem",    
                f"certificates/{cert_file_a}_cert.pem"    
            )
            proxy_api = ProxyToInternalPort(internal_ip_a, api_port)
            proxy_site_api = server.Site(proxy_api)
            reactor.listenSSL(api_port_external, proxy_site_api, sslContextA, interface=external_ip_a)
            print_message(f"API SSL Proxy    : {GREEN}From{RESET} {external_ip_a}:{api_port_external} {GREEN}->{RESET} {internal_ip_a}:{api_port} {GREEN}is now active.{RESET}")
        else:
            proxy_api = ProxyToInternalPort(internal_ip_a, api_port)
            proxy_site_api = server.Site(proxy_api)
            reactor.listenTCP(api_port_external, proxy_site_api, interface=external_ip_a)
            print_message(f"API HTTP Proxy    : {GREEN}From{RESET} {external_ip_a}:{api_port_external} {GREEN}->{RESET} {internal_ip_a}:{api_port} {GREEN}is now active.{RESET}")
        proxies_started = True
    else:
        print_message(f"API Proxy is {RED}DISABLED{RESET}")

    if not proxies_started:
        print_message(f"No proxies were started. {RED}Both{RESET} API and Gradio proxies are {RED}DISABLED{RESET}.", message_type="warning")
        return

    reactor.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--external_ip_a", required=True, help="External IP Address A ('0' to disable)")
    parser.add_argument("--internal_ip_a", required=True, help="Internal IP Address A")
    parser.add_argument("--external_ip_b", required=True, help="External IP Address B ('0' to disable)")
    parser.add_argument("--internal_ip_b", required=True, help="Internal IP Address B")
    parser.add_argument("--cert_file_a", required=True, help="Certificate file base name for External IP A (without '_key' or '_cert', or '0' for HTTP only)")
    parser.add_argument("--cert_file_b", required=True, help="Certificate file base name for External IP B (without '_key' or '_cert', or '0' for HTTP only)")
    args = parser.parse_args()

    run_proxy(
        external_ip_a=args.external_ip_a,
        internal_ip_a=args.internal_ip_a,
        external_ip_b=args.external_ip_b,
        internal_ip_b=args.internal_ip_b,
        cert_file_a=args.cert_file_a,
        cert_file_b=args.cert_file_b
    )
