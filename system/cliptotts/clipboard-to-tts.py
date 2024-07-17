import sys
import time
import signal
import json
import warnings
import pyperclip
import requests
from PyQt5.QtWidgets import (QApplication, QSystemTrayIcon, QMenu, QWidget, QVBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QMessageBox, QComboBox, QSlider)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon

# Suppress the specific warning
warnings.filterwarnings("ignore", category=UserWarning, module="charset_normalizer")

SETTINGS_FILE = "tts_settings.json"
TTS_SERVER_URL = "http://127.0.0.1:7851"

LANGUAGES = {
    "ar": "Arabic", "zh-cn": "Chinese (Simplified)", "cs": "Czech", "nl": "Dutch",
    "en": "English", "fr": "French", "de": "German", "hi": "Hindi (limited support)",
    "hu": "Hungarian", "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "es": "Spanish", "tr": "Turkish"
}

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
    error_occurred = pyqtSignal(str)

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
        params["language"] = params["language"][:2]  # Ensure we only send the 2-letter code
        try:
            response = requests.post(f"{self.api_url}/api/tts-generate", data=params)
            if response.status_code != 200:
                error_message = f"TTS generation failed with status code: {response.status_code}. Response: {response.text}"
                print(error_message)
                self.error_occurred.emit(error_message)
        except requests.RequestException as e:
            error_message = f"Error sending request to TTS API: {e}"
            print(error_message)
            self.error_occurred.emit(error_message)

class SettingsWindow(QWidget):
    settings_updated = pyqtSignal(dict)

    def __init__(self, tts_worker):
        super().__init__()
        self.tts_worker = tts_worker
        self.voices = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(QLabel("AllTalk API URL:"))
        self.api_url_input = QLineEdit(self.tts_worker.api_url)
        layout.addWidget(self.api_url_input)

        layout.addWidget(QLabel("Voice:"))
        self.voice_combo = QComboBox()
        layout.addWidget(self.voice_combo)

        layout.addWidget(QLabel("Language:"))
        self.language_combo = QComboBox()
        for code, name in LANGUAGES.items():
            self.language_combo.addItem(f"{name} ({code})", code)
        current_lang = self.tts_worker.params.get("language", "en")
        index = self.language_combo.findData(current_lang)
        if index >= 0:
            self.language_combo.setCurrentIndex(index)
        layout.addWidget(self.language_combo)

        layout.addWidget(QLabel("Text Filtering:"))
        self.text_filtering_input = QLineEdit(self.tts_worker.params["text_filtering"])
        layout.addWidget(self.text_filtering_input)

        layout.addWidget(QLabel("Playback Volume:"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(1, 10)
        self.volume_slider.setValue(int(float(self.tts_worker.params["autoplay_volume"]) * 10))
        layout.addWidget(self.volume_slider)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)

        refresh_button = QPushButton("Refresh Voices")
        refresh_button.clicked.connect(self.refresh_voices)
        layout.addWidget(refresh_button)

        self.setWindowTitle("AllTalk Clipboard TTS")
        self.setGeometry(300, 300, 300, 400)

        self.refresh_voices()

    def refresh_voices(self):
        try:
            response = requests.get(f"{self.tts_worker.api_url}/api/voices")
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "success" and "voices" in data:
                    self.voices = data["voices"]
                    self.voice_combo.clear()
                    for voice in self.voices:
                        self.voice_combo.addItem(voice)
                    if self.voices:
                        current_voice = self.tts_worker.params["character_voice_gen"]
                        if current_voice in self.voices:
                            self.voice_combo.setCurrentText(current_voice)
                        else:
                            self.voice_combo.setCurrentIndex(0)
                else:
                    QMessageBox.warning(self, "Error", "Failed to fetch voices: Unexpected response format")
            else:
                QMessageBox.warning(self, "Error", f"Failed to fetch voices. Status code: {response.status_code}")
        except requests.RequestException as e:
            QMessageBox.warning(self, "Error", f"Failed to fetch voices: {str(e)}")

    def save_settings(self):
            new_settings = {
                "api_url": self.api_url_input.text(),
                "params": {
                    "character_voice_gen": self.voice_combo.currentText(),
                    "language": self.language_combo.currentData(),
                    "text_filtering": self.text_filtering_input.text(),
                    "output_file_name": "clipboard_tts",
                    "output_file_timestamp": "true",
                    "autoplay": "true",
                    "autoplay_volume": str(self.volume_slider.value() / 10.0)
                }
            }
            self.settings_updated.emit(new_settings)
            self.hide()

class TTSApp(QSystemTrayIcon):
    def __init__(self, app):
        super().__init__(parent=app)
        self.app = app
        self.load_settings()
        self.init_ui()

    def load_settings(self):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                self.settings = json.load(f)
        except FileNotFoundError:
            self.settings = {
                "api_url": TTS_SERVER_URL,
                "params": {
                    "text_filtering": "standard",
                    "character_voice_gen": "",
                    "language": "en",
                    "output_file_name": "clipboard_tts",
                    "output_file_timestamp": "true",
                    "autoplay": "true",
                    "autoplay_volume": "0.8"
                }
            }

    def save_settings(self):
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(self.settings, f)

    def init_ui(self):
        self.tts_worker = TTSWorker(self.settings["api_url"], self.settings["params"])
        self.tts_worker.error_occurred.connect(self.show_error_message)
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
        self.settings_window.settings_updated.connect(self.update_settings)

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

    def update_settings(self, new_settings):
        self.settings = new_settings
        self.tts_worker.api_url = new_settings["api_url"]
        self.tts_worker.params = new_settings["params"]
        self.save_settings()
        self.showMessage("TTS App", "Settings updated successfully")

    def show_error_message(self, message):
        QMessageBox.critical(None, "Error", message)

    def exit_app(self):
        self.clipboard_monitor.stop()
        self.tts_worker.quit()
        QApplication.instance().quit()

def check_server_ready(url, timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/api/ready")
            if response.status_code == 200 and response.text == "Ready":
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False

def signal_handler(signal, frame):
    print("[Clipboard to TTS] Exiting...")
    QApplication.instance().quit()

if __name__ == "__main__":
    print("[Clipboard to TTS] AllTalk Clipboard to TTS is starting...")

    if not check_server_ready(TTS_SERVER_URL):
        print("[Clipboard to TTS] Error: AllTalk TTS server is not ready. Please start the AllTalk server and try again.")
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)

    app = QApplication(sys.argv)
    
    if not QSystemTrayIcon.isSystemTrayAvailable():
        print("[Clipboard to TTS] System tray is not available on this system.")
        sys.exit(1)

    tts_app = TTSApp(app)
    
    # This allows for Ctrl+C to work
    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    print("[Clipboard to TTS] AllTalk Clipboard to TTS is running... Check your system tray for the icon.")
    print("[Clipboard to TTS] Press Ctrl+C to exit")
    sys.exit(app.exec_())
