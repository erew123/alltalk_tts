import sys
import time
import signal
import pyperclip
import requests
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon

class ClipboardMonitor(QThread):
    text_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.last_text = pyperclip.paste()
        self.running = True

    def run(self):
        while self.running:
            current_text = pyperclip.paste()
            if current_text != self.last_text:
                self.text_changed.emit(current_text)
                self.last_text = current_text
            time.sleep(0.5)

    def stop(self):
        self.running = False

class TTSWorker(QThread):
    def __init__(self, api_url, params):
        super().__init__()
        self.api_url = api_url
        self.params = params
        self.text_queue = []

    def add_text(self, text):
        self.text_queue.append(text)

    def run(self):
        while True:
            if self.text_queue:
                text = self.text_queue.pop(0)
                self.send_to_tts(text)
            else:
                time.sleep(0.1)

    def send_to_tts(self, text):
        params = self.params.copy()
        params["text_input"] = text
        try:
            response = requests.post(self.api_url, data=params)
            if response.status_code != 200:
                print(f"TTS generation failed with status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error sending request to TTS API: {e}")

class SettingsWindow(QWidget):
    def __init__(self, tts_worker):
        super().__init__()
        self.tts_worker = tts_worker
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(QLabel("TTS API URL:"))
        self.api_url_input = QLineEdit(self.tts_worker.api_url)
        layout.addWidget(self.api_url_input)

        layout.addWidget(QLabel("Voice:"))
        self.voice_input = QLineEdit(self.tts_worker.params["character_voice_gen"])
        layout.addWidget(self.voice_input)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)

        self.setWindowTitle("TTS Settings")
        self.setGeometry(300, 300, 300, 150)

    def save_settings(self):
        self.tts_worker.api_url = self.api_url_input.text()
        self.tts_worker.params["character_voice_gen"] = self.voice_input.text()
        self.close()

class TTSApp(QSystemTrayIcon):
    def __init__(self, app):
        super().__init__(parent=app)
        self.app = app
        self.init_ui()

    def init_ui(self):
        self.tts_worker = TTSWorker(
            api_url="http://127.0.0.1:7851/api/tts-generate",
            params={
                "text_filtering": "standard",
                "character_voice_gen": "female_01.wav",
                "language": "en",
                "output_file_name": "clipboard_tts",
                "output_file_timestamp": "true",
                "autoplay": "true",
                "autoplay_volume": "0.8"
            }
        )
        self.tts_worker.start()

        self.clipboard_monitor = ClipboardMonitor()
        self.clipboard_monitor.text_changed.connect(self.on_text_changed)

        self.setIcon(QIcon("icon.png"))
        self.setVisible(True)

        menu = QMenu()
        start_action = menu.addAction("Start Monitoring")
        start_action.triggered.connect(self.start_monitoring)
        stop_action = menu.addAction("Stop Monitoring")
        stop_action.triggered.connect(self.stop_monitoring)
        settings_action = menu.addAction("Settings")
        settings_action.triggered.connect(self.show_settings)
        exit_action = menu.addAction("Exit")
        exit_action.triggered.connect(self.exit_app)
        self.setContextMenu(menu)

        self.settings_window = SettingsWindow(self.tts_worker)

    def on_text_changed(self, text):
        self.tts_worker.add_text(text)

    def start_monitoring(self):
        self.clipboard_monitor.start()
        self.showMessage("TTS App", "Clipboard monitoring started")

    def stop_monitoring(self):
        self.clipboard_monitor.stop()
        self.showMessage("TTS App", "Clipboard monitoring stopped")

    def show_settings(self):
        self.settings_window.show()

    def exit_app(self):
        self.clipboard_monitor.stop()
        self.tts_worker.quit()
        QApplication.instance().quit()

def signal_handler(signal, frame):
    QApplication.instance().quit()

if __name__ == "__main__":
    print("[AllTalk TTS] AllTalk Clipboard to TTS is starting...")
    print("[AllTalk TTS] Press Ctrl+C to exit")
    signal.signal(signal.SIGINT, signal_handler)

    app = QApplication(sys.argv)
    
    if not QSystemTrayIcon.isSystemTrayAvailable():
        print("System tray is not available on this system.")
        sys.exit(1)

    tts_app = TTSApp(app)
    
    # This allows for Ctrl+C to work
    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    sys.exit(app.exec_())
