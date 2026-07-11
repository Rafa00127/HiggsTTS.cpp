"""Simple TTS GUI — Higgs TTS text-to-speech with voice cloning."""
import ctypes, os, sys, json, struct, socket, time, threading, wave
import numpy as np
import sounddevice as sd
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_FILE_DIR)

# ======================== Constants ========================
CONFIG = os.path.join(_FILE_DIR, "tts_gui_config.json")
SAMPLE_RATE = 24000
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 9988

QSS = """
* { font-family: "Segoe UI", sans-serif; }
QMainWindow, QWidget { background-color: #111; color: #ddd; }
QPlainTextEdit, QTextEdit {
    background-color: #1a1a18; color: #c8b878; border: 1px solid #333;
    border-radius: 8px; padding: 16px; font-size: 16px;
    selection-background-color: #3a3520;
}
QLineEdit {
    background: #252525; color: #ddd; border: 1px solid #444;
    border-radius: 6px; padding: 6px 10px; font-size: 13px;
}
QLineEdit:focus { border-color: #666; }
QPushButton {
    background: #2a2a2a; color: #ddd; border: 1px solid #444;
    border-radius: 6px; padding: 8px 16px; font-size: 13px;
}
QPushButton:hover { background: #3a3a3a; }
QPushButton#primaryBtn { background: #2d5a2d; color: #fff; font-size: 14px; font-weight: bold; padding: 10px 24px; border: none; }
QPushButton#primaryBtn:hover { background: #3a6a3a; }
QComboBox {
    background: #252525; color: #ddd; border: 1px solid #444;
    border-radius: 6px; padding: 6px 10px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background: #252525; color: #ddd; selection-background-color: #333; }
QLabel { background: transparent; }
"""

# ======================== Helpers ========================

def load_cfg():
    if os.path.exists(CONFIG):
        with open(CONFIG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"server": {}}

def save_cfg(c):
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(c, f, ensure_ascii=False, indent=2)


# ======================== TTS Engine ========================

class TtsEngine:
    @staticmethod
    def _recvn(sock, n):
        data = bytearray()
        while len(data) < n:
            try:
                chunk = sock.recv(n - len(data))
            except ConnectionResetError:
                chunk = b""
            if not chunk:
                break
            data.extend(chunk)
        return data

    @classmethod
    def synth(cls, text, port=9988, temperature=0.9):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(120)
        sock.connect((SERVER_HOST, port))
        payload = text.encode("utf-8")
        sock.sendall(struct.pack(">i", len(payload)))
        sock.sendall(struct.pack(">f", temperature))
        sock.sendall(payload)
        hdr = cls._recvn(sock, 4)
        if len(hdr) < 4:
            sock.close()
            raise RuntimeError("Server returned empty header")
        n_samples = struct.unpack(">i", hdr)[0]
        if n_samples <= 0:
            sock.close()
            raise RuntimeError(f"Server error: n_samples={n_samples}")
        data = cls._recvn(sock, n_samples * 4)
        try:
            sock.shutdown(socket.SHUT_RD)
        except OSError:
            pass
        sock.close()
        arr = np.frombuffer(data, dtype=np.float32).copy()
        if len(arr) != n_samples:
            raise RuntimeError(f"Truncated PCM: got {len(arr)}, expected {n_samples}")
        return arr


# ======================== Model Config Dialog ========================

class ModelConfigDialog(QDialog):
    def __init__(self, parent, cfg):
        super().__init__(parent); self._c = cfg
        self.setWindowTitle("Model Config"); self.setFixedSize(620, 420)
        self.setStyleSheet(QSS)
        lay = QVBoxLayout(self); lay.setSpacing(6)
        sv = cfg.get("server", {})

        def _row(label, *widgets):
            r = QHBoxLayout(); r.addWidget(QLabel(label))
            for w in widgets: r.addWidget(w)
            lay.addLayout(r)

        hl = QVBoxLayout(); hl.setContentsMargins(0, 0, 0, 0)

        self._exe = QLineEdit(sv.get("exe", "higgs_server.exe"))
        r = QHBoxLayout(); r.addWidget(QLabel("Server Path")); r.addWidget(self._exe)
        b = QPushButton("Browse"); b.clicked.connect(lambda: self._br_exe(self._exe)); r.addWidget(b); hl.addLayout(r)

        self._model = QLineEdit(sv.get("model", ""))
        r = QHBoxLayout(); r.addWidget(QLabel("Model GGUF")); r.addWidget(self._model)
        b = QPushButton("Browse"); b.clicked.connect(lambda: self._br(self._model)); r.addWidget(b); hl.addLayout(r)

        self._ref_wav = QLineEdit(sv.get("ref_wav", ""))
        r = QHBoxLayout(); r.addWidget(QLabel("Reference Audio")); r.addWidget(self._ref_wav)
        b = QPushButton("Browse"); b.clicked.connect(lambda: self._br(self._ref_wav)); r.addWidget(b); hl.addLayout(r)

        self._ref_text = QLineEdit(sv.get("ref_text", ""))
        r = QHBoxLayout(); r.addWidget(QLabel("Reference Text")); r.addWidget(self._ref_text)
        hl.addLayout(r)

        self._tokenizer = QLineEdit(sv.get("tokenizer", ""))
        r = QHBoxLayout(); r.addWidget(QLabel("Tokenizer (json)")); r.addWidget(self._tokenizer)
        b = QPushButton("Browse"); b.clicked.connect(lambda: self._br_json(self._tokenizer)); r.addWidget(b); hl.addLayout(r)

        self._temp = QLineEdit(sv.get("temp", "0.9"))
        self._temp.setPlaceholderText("temperature")
        r = QHBoxLayout(); r.addWidget(QLabel("Temperature")); r.addWidget(self._temp)
        hl.addLayout(r)

        self._seed = QLineEdit(sv.get("seed", "42"))
        self._seed.setPlaceholderText("seed")
        r = QHBoxLayout(); r.addWidget(QLabel("Seed")); r.addWidget(self._seed)
        hl.addLayout(r)

        lay.addLayout(hl)

        self._port = QLineEdit(sv.get("port", "9988"))
        _row("Port", self._port)

        btn = QPushButton("Launch Server")
        btn.setObjectName("primaryBtn"); btn.clicked.connect(self._launch)
        lay.addWidget(btn)
        self._status = QLabel(""); lay.addWidget(self._status)

    def _br(self, w):
        p = QFileDialog.getOpenFileName(self, "Select", w.text(), "GGUF (*.gguf);;Audio (*.wav *.mp3);;All (*.*)")
        if p[0]: w.setText(p[0])
    def _br_exe(self, w):
        p = QFileDialog.getOpenFileName(self, "Select", w.text(), "exe (*.exe);;All (*.*)")
        if p[0]: w.setText(p[0])
    def _br_json(self, w):
        p = QFileDialog.getOpenFileName(self, "Select", w.text(), "JSON (*.json);;All (*.*)")
        if p[0]: w.setText(p[0])

    def _launch(self):
        sv = {
            "model_type": "Higgs TTS",
            "port": self._port.text(),
            "exe": self._exe.text(), "model": self._model.text(),
            "ref_wav": self._ref_wav.text(), "ref_text": self._ref_text.text(),
            "temp": self._temp.text(), "seed": self._seed.text(),
            "tokenizer": self._tokenizer.text(),
        }
        self._c["server"] = sv
        save_cfg(self._c)

        exe = (sv.get("exe") or "").strip() or "higgs_server.exe"
        args = [exe, "--model", sv["model"], "--ref-wav", sv["ref_wav"],
                "--port", sv.get("port", "9989"),
                "--temperature", sv.get("temp", "0.9"),
                "--seed", sv.get("seed", "42")]
        if sv.get("ref_text"): args += ["--ref-text", sv["ref_text"]]
        if sv.get("tokenizer"): args += ["--tokenizer", sv["tokenizer"]]

        cmd_line = " ".join(f'"{a}"' if " " in a else a for a in args)
        bat = os.path.join(_FILE_DIR, "tts_gui_launch.bat")
        with open(bat, "w", encoding="utf-8") as f:
            f.write(f"@echo off\r\nchcp 65001 >nul\r\n{cmd_line}\r\npause\r\n")
        os.startfile(bat)

        self._status.setText("Server launching...")
        self._status.setStyleSheet("color:#fa0;")
        port = int(sv.get("port", SERVER_PORT))

        def _poll():
            for _ in range(20):
                time.sleep(0.5)
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(1)
                    s.connect((SERVER_HOST, port))
                    s.close()
                    QMetaObject.invokeMethod(self._status, "setText", Qt.ConnectionType.QueuedConnection,
                        Q_ARG(str, f"Server: {SERVER_HOST}:{port}"))
                    QMetaObject.invokeMethod(self._status, "setStyleSheet", Qt.ConnectionType.QueuedConnection,
                        Q_ARG(str, "color:#0a0;"))
                    return
                except (ConnectionRefusedError, socket.timeout, OSError):
                    continue
            QMetaObject.invokeMethod(self._status, "setText", Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, "Server timed out"))
            QMetaObject.invokeMethod(self._status, "setStyleSheet", Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, "color:#f00;"))

        threading.Thread(target=_poll, daemon=True).start()


# ======================== Main Window ========================

class SimpleTtsGui(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Higgs TTS — Simple GUI")
        self.resize(700, 600); self.setStyleSheet(QSS)
        self._c = load_cfg()
        self._pcm = None
        self._playing = False

        sv = self._c.get("server", {})
        if sv.get("port"):
            global SERVER_PORT
            SERVER_PORT = int(sv["port"])

        self._build()

    def _build(self):
        cw = QWidget(); self.setCentralWidget(cw)
        lay = QVBoxLayout(cw); lay.setContentsMargins(12, 10, 12, 10); lay.setSpacing(8)

        # ── Toolbar ──
        tb = QHBoxLayout()
        cfg_btn = QPushButton("Model Config")
        cfg_btn.clicked.connect(lambda: ModelConfigDialog(self, self._c).exec())
        tb.addWidget(cfg_btn)
        launch_btn = QPushButton("Launch Server")
        launch_btn.clicked.connect(self._launch_server)
        tb.addWidget(launch_btn)
        self._status_lbl = QLabel("Not connected"); self._status_lbl.setStyleSheet("color:#888;")
        tb.addWidget(self._status_lbl)
        tb.addStretch()
        lay.addLayout(tb)

        # ── Text input ──
        lay.addWidget(QLabel("Input Text"))
        self._text_edit = QPlainTextEdit()
        self._text_edit.setPlaceholderText("Enter text to synthesize...")
        self._text_edit.setMinimumHeight(200)
        lay.addWidget(self._text_edit)

        # ── Controls ──
        btn_row = QHBoxLayout()
        self._synth_btn = QPushButton("Synthesize")
        self._synth_btn.setObjectName("primaryBtn")
        self._synth_btn.clicked.connect(self._synthesize)
        btn_row.addWidget(self._synth_btn)

        btn_row.addWidget(QLabel("  Temp"))
        self._temp_spin = QDoubleSpinBox()
        self._temp_spin.setRange(0.1, 2.0); self._temp_spin.setSingleStep(0.05)
        self._temp_spin.setValue(float(self._c.get("temperature", 0.9)))
        self._temp_spin.setDecimals(2); self._temp_spin.setFixedWidth(70)
        self._temp_spin.valueChanged.connect(lambda v: self._c.update({"temperature": v}) or save_cfg(self._c))
        btn_row.addWidget(self._temp_spin)

        btn_row.addWidget(QLabel("  Tag"))
        self._tag = QComboBox()
        self._tag.setFixedWidth(220)
        self._tag.addItem("(none)", "")
        self._tag.addItem("── Emotion ──", "")
        for t, d in [("<|emotion:elation|>","Elation"),("<|emotion:amusement|>","Amusement"),("<|emotion:enthusiasm|>","Enthusiasm"),("<|emotion:determination|>","Determination"),("<|emotion:pride|>","Pride"),("<|emotion:contentment|>","Contentment"),("<|emotion:affection|>","Affection"),("<|emotion:relief|>","Relief"),("<|emotion:contemplation|>","Contemplation"),("<|emotion:confusion|>","Confusion"),("<|emotion:surprise|>","Surprise"),("<|emotion:awe|>","Awe"),("<|emotion:longing|>","Longing"),("<|emotion:arousal|>","Arousal"),("<|emotion:anger|>","Anger"),("<|emotion:fear|>","Fear"),("<|emotion:disgust|>","Disgust"),("<|emotion:bitterness|>","Bitterness"),("<|emotion:sadness|>","Sadness"),("<|emotion:shame|>","Shame"),("<|emotion:helplessness|>","Helplessness")]:
            self._tag.addItem(f"  {d}", t)
        self._tag.addItem("── Style ──", "")
        for t, d in [("<|style:singing|>","Singing"),("<|style:shouting|>","Shouting"),("<|style:whispering|>","Whispering")]:
            self._tag.addItem(f"  {d}", t)
        self._tag.addItem("── SFX ──", "")
        for t, d in [("<|sfx:cough|>","Cough"),("<|sfx:laughter|>","Laughter"),("<|sfx:crying|>","Crying"),("<|sfx:screaming|>","Screaming"),("<|sfx:burping|>","Burping"),("<|sfx:humming|>","Humming"),("<|sfx:sigh|>","Sigh"),("<|sfx:sniff|>","Sniff"),("<|sfx:sneeze|>","Sneeze")]:
            self._tag.addItem(f"  {d}", t)
        self._tag.addItem("── Prosody ──", "")
        for t, d in [("<|prosody:speed_very_slow|>","Speed Very Slow"),("<|prosody:speed_slow|>","Speed Slow"),("<|prosody:speed_fast|>","Speed Fast"),("<|prosody:speed_very_fast|>","Speed Very Fast"),("<|prosody:pitch_low|>","Pitch Low"),("<|prosody:pitch_high|>","Pitch High"),("<|prosody:pause|>","Pause"),("<|prosody:long_pause|>","Long Pause"),("<|prosody:expressive_high|>","Expressive High"),("<|prosody:expressive_low|>","Expressive Low")]:
            self._tag.addItem(f"  {d}", t)
        cur = self._c.get("tag", "")
        for i in range(self._tag.count()):
            if self._tag.itemData(i) == cur:
                self._tag.setCurrentIndex(i); break
        self._tag.currentIndexChanged.connect(lambda: self._c.update({"tag": self._tag.currentData()}) or save_cfg(self._c))
        btn_row.addWidget(self._tag)

        self._play_btn = QPushButton("Play")
        self._play_btn.setEnabled(False)
        self._play_btn.clicked.connect(self._toggle_play)
        btn_row.addWidget(self._play_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save)
        btn_row.addWidget(self._save_btn)

        btn_row.addStretch()
        self._info_lbl = QLabel(""); btn_row.addWidget(self._info_lbl)
        lay.addLayout(btn_row)

    # ── Server launch ──
    def _launch_server(self):
        bat = os.path.join(_FILE_DIR, "tts_gui_launch.bat")
        if not os.path.exists(bat):
            QMessageBox.warning(self, "Notice", "Please configure and launch the server first in Model Config.")
            return
        os.startfile(bat)
        self._status_lbl.setText("Server launching..."); self._status_lbl.setStyleSheet("color:#fa0;")
        sv = self._c.get("server", {})
        port = int(sv.get("port", SERVER_PORT))

        def _poll():
            for _ in range(20):
                time.sleep(0.5)
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(1)
                    s.connect((SERVER_HOST, port))
                    s.close()
                    QMetaObject.invokeMethod(self._status_lbl, "setText", Qt.ConnectionType.QueuedConnection,
                        Q_ARG(str, f"{SERVER_HOST}:{port}"))
                    QMetaObject.invokeMethod(self._status_lbl, "setStyleSheet", Qt.ConnectionType.QueuedConnection,
                        Q_ARG(str, "color:#0a0;"))
                    return
                except (ConnectionRefusedError, socket.timeout, OSError):
                    continue
            QMetaObject.invokeMethod(self._status_lbl, "setText", Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, "Server timed out"))
            QMetaObject.invokeMethod(self._status_lbl, "setStyleSheet", Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, "color:#f00;"))

        threading.Thread(target=_poll, daemon=True).start()

    # ── Synthesize ──
    def _synthesize(self):
        text = self._text_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Notice", "Please enter some text.")
            return

        self._pcm = None
        self._play_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._synth_btn.setEnabled(False)
        self._synth_btn.setText("Synthesizing...")
        self._info_lbl.setText("")
        QApplication.processEvents()

        sv = self._c.get("server", {})
        port = int(sv.get("port", SERVER_PORT))
        temperature = self._temp_spin.value()

        tag = self._c.get("tag", "")
        synth_text = text
        if tag:
            synth_text = tag + text

        # for a, b in [('「', '"'), ('」', '"'), ('『', "'"), ('』', "'")]:
        #     synth_text = synth_text.replace(a, b)

        def _run():
            try:
                pcm = TtsEngine.synth(synth_text, port, temperature)
                self._pcm = pcm
                duration = len(pcm) / SAMPLE_RATE
                self._synth_btn.setText("Resynthesize")
                self._synth_btn.setEnabled(True)
                self._play_btn.setEnabled(True)
                self._save_btn.setEnabled(True)
                self._info_lbl.setText(f"Duration {duration:.1f}s | {len(pcm)} samples")
                self._info_lbl.setStyleSheet("color:#0a0;")
            except Exception as e:
                self._synth_btn.setText("Synthesize")
                self._synth_btn.setEnabled(True)
                self._info_lbl.setText(f"Error: {e}")
                self._info_lbl.setStyleSheet("color:#f00;")

        threading.Thread(target=_run, daemon=True).start()

    # ── Playback ──
    def _toggle_play(self):
        if self._pcm is None:
            return
        if self._playing:
            sd.stop()
            self._playing = False
            self._play_btn.setText("Play")
        else:
            self._playing = True
            self._play_btn.setText("Stop")
            sd.play(self._pcm.astype(np.float32), SAMPLE_RATE)

            def _wait():
                try:
                    if sd.get_stream() is not None and sd.get_stream().active:
                        QTimer.singleShot(200, _wait)
                    else:
                        self._playing = False
                        self._play_btn.setText("Play")
                except Exception:
                    self._playing = False
                    self._play_btn.setText("Play")

            QTimer.singleShot(200, _wait)

    # ── Save ──
    def _save(self):
        if self._pcm is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Audio", "output.wav",
                                               "WAV (*.wav);;All (*.*)")
        if not path:
            return
        import wave
        pcm16 = np.clip(self._pcm, -1, 1) * 32767
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm16.astype(np.int16).tobytes())
        self._info_lbl.setText(f"Saved: {os.path.basename(path)}")
        self._info_lbl.setStyleSheet("color:#0a0;")

    def closeEvent(self, e):
        sd.stop(); save_cfg(self._c); e.accept()


if __name__ == "__main__":
    if sys.platform == "win32":
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
    app = QApplication(sys.argv); app.setStyle("Fusion")
    win = SimpleTtsGui(); win.show()
    sys.exit(app.exec())
