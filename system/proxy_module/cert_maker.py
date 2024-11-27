from pathlib import Path
import argparse
from OpenSSL import crypto
import sys

def generate_certificate(cert_name: str, output_dir: Path, cn: str = "localhost", days: int = 365):
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 2048)
    
    cert = crypto.X509()
    cert.get_subject().CN = cn
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(days*24*60*60)
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha256')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f"{cert_name}_cert.pem", "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    with open(output_dir / f"{cert_name}_key.pem", "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))

def main():
    print("Certificate Generator")
    print("-" * 20)
    
    # Get default paths
    default_path = Path(__file__).parent / "certificates"
    
    # Get user input with defaults
    cert_name = input(f"Certificate name (default: test_cert): ") or "test_cert"
    cn = input(f"Common Name (default: localhost): ") or "localhost"
    days = input(f"Valid days (default: 365): ") or "365"
    output_dir = input(f"Output directory (default: {default_path}): ") or str(default_path)
    
    try:
        days = int(days)
        output_path = Path(output_dir)
        generate_certificate(cert_name, output_path, cn, days)
        print(f"\nSuccess! Certificate files created:")
        print(f"- {output_path / f'{cert_name}_cert.pem'}")
        print(f"- {output_path / f'{cert_name}_key.pem'}")
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()