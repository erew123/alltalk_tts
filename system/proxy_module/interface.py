import gradio as gr
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .proxy_manager import ProxyManager

def create_proxy_interface(proxy_manager: 'ProxyManager'):
    """Create the proxy management interface"""
    config = proxy_manager.config_manager.get_instance()
    
    def validate_ports(gradio_port, api_port):
        """Validate port configurations"""
        if gradio_port == api_port and gradio_port != 0 and api_port != 0:
            return False, "Gradio and API ports cannot be the same!"
        if not (0 <= gradio_port <= 65535) or not (0 <= api_port <= 65535):
            return False, "Ports must be between 0 and 65535"
        return True, "Valid port configuration"

    def handle_save_config(proxy_enabled, gradio_enabled, api_enabled, 
                         gradio_port, api_port, startup, external_ip):
        # Validate ports first
        valid, message = validate_ports(gradio_port, api_port)
        if not valid:
            return message

        config.proxy_settings.proxy_enabled = proxy_enabled
        config.proxy_settings.start_on_startup = startup
        
        # Update Gradio endpoint
        config.proxy_settings.gradio_endpoint.enabled = gradio_enabled
        config.proxy_settings.gradio_endpoint.external_port = int(gradio_port) if gradio_enabled else 0
        config.proxy_settings.gradio_endpoint.external_ip = external_ip
        
        # Update API endpoint
        config.proxy_settings.api_endpoint.enabled = api_enabled
        config.proxy_settings.api_endpoint.external_port = int(api_port) if api_enabled else 0
        config.proxy_settings.api_endpoint.external_ip = external_ip
        
        config.save()
        return "Configuration saved successfully"

    def update_control_states(proxy_enabled):
        """Update enabled/disabled state of controls based on master switch"""
        return {
            start_button: gr.update(interactive=proxy_enabled),
            stop_button: gr.update(interactive=proxy_enabled),
            gradio_enabled: gr.update(interactive=proxy_enabled),
            api_enabled: gr.update(interactive=proxy_enabled),
            start_on_startup: gr.update(interactive=proxy_enabled)
        }

    def delete_all_certificates():
        """Delete all certificates and reset config"""
        try:
            # Delete certificate files
            for cert_file in proxy_manager.certs_path.glob("*.pem"):
                cert_file.unlink()
            
            # Reset config
            config = proxy_manager.config_manager.get_instance()
            config.proxy_settings.gradio_endpoint.cert_name = ""
            config.proxy_settings.api_endpoint.cert_name = ""
            config.save()
            
            return "All certificates deleted and config reset"
        except Exception as e:
            return f"Error deleting certificates: {str(e)}"
    
    with gr.Tab("Proxy Settings"):
        with gr.Column():
            with gr.Row():
                with gr.Group():
                    proxy_enabled = gr.Checkbox(
                        label="Enable Proxy System", 
                        value=config.proxy_settings.proxy_enabled,
                        info="Master switch/safety lockout for proxy functionality"
                    )
                    start_on_startup = gr.Checkbox(
                        label="Start Proxy Automatically on Startup", 
                        value=config.proxy_settings.start_on_startup,
                        info="Automatically start proxy when app launches"
                    )
                    external_ip = gr.Textbox(
                        label="External IP",
                        value=config.proxy_settings.gradio_endpoint.external_ip,
                        info="Use 0.0.0.0 to bind to all interfaces"
                    )
                with gr.Group():
                    gr.Markdown("API interface Proxy Management")
                    api_enabled = gr.Dropdown(
                        choices=["Enabled", "Disabled"],
                        value="Enabled" if config.proxy_settings.api_endpoint.enabled == "Enabled" else "Disabled",
                        label="API (TTS Generation) Proxy",
                        info="Enable or Disable Proxying the API (TTS Generation)"
                    )
                    api_port = gr.Number(
                        value=config.proxy_settings.api_endpoint.external_port,
                        label="API Port",
                        info="External port you want to use for the API (TTS Generation)"
                    )                                      
                with gr.Group():
                    gr.Markdown("Gradio interface Proxy Management")
                    gradio_enabled = gr.Dropdown(
                        choices=["Enabled", "Disabled"],
                        value="Enabled" if config.proxy_settings.gradio_endpoint.enabled == "Enabled" else "Disabled",
                        label="Gradio Interface Proxy",
                        info="Enable or Disable Proxying the Gradio interface"
                    )
                    gradio_port = gr.Number(
                        value=config.proxy_settings.gradio_endpoint.external_port,
                        label="Gradio Interface Port",
                        info="External port you want to use for the Gradio interface"
                    )

            # Status and control section
            with gr.Row():
                with gr.Column(scale=2):
                    status_output = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=4
                    )
                with gr.Column(scale=1):
                    save_button = gr.Button("Save Configuration", variant="primary")                    
                    with gr.Row():
                        start_button = gr.Button("Start Service")
                        stop_button = gr.Button("Stop Service")
                        status_button = gr.Button("Check Status")

            # Certificate section
            with gr.Row():
                gr.Markdown("### Certificate Management")
            with gr.Row():
                with gr.Column(scale=1):
                    cert_file = gr.File(label="Certificate (.crt/.pem)")
                with gr.Column(scale=1):
                    key_file = gr.File(label="Key (.key)")
                with gr.Column(scale=1):
                    cert_name = gr.Textbox(label="Certificate Name", placeholder="e.g., gradio_cert")
                    upload_button = gr.Button("Upload Certificate", variant="primary")
                    delete_button = gr.Button("Delete Certificates", variant="secondary")

            # Connect everything
            save_button.click(
                fn=handle_save_config,
                inputs=[
                    proxy_enabled,
                    gradio_enabled,
                    api_enabled,
                    gradio_port,
                    api_port,
                    start_on_startup,
                    external_ip
                ],
                outputs=status_output
            )
            proxy_enabled.change(
                fn=update_control_states,
                inputs=[proxy_enabled],
                outputs=[start_button, stop_button, gradio_enabled, api_enabled, start_on_startup]
            )
            start_button.click(fn=proxy_manager.start_proxy, outputs=status_output)
            stop_button.click(fn=proxy_manager.stop_proxy, outputs=status_output)
            status_button.click(fn=proxy_manager.get_status, outputs=status_output)
            upload_button.click(
                fn=proxy_manager.handle_cert_upload,
                inputs=[cert_file, key_file, cert_name],
                outputs=status_output
            )
            delete_button.click(fn=delete_all_certificates, outputs=status_output)

    return proxy_enabled, start_on_startup, gradio_port, api_port