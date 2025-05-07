import sys
import os
import numpy as np
import parselmouth
import sounddevice as sd
import tempfile
import scipy.io.wavfile as wavfile
import time
import threading
import signal
import cv2
from moviepy.editor import AudioFileClip, VideoFileClip
from scipy.interpolate import interp1d
from scipy.signal import medfilt, savgol_filter
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QCheckBox, QLineEdit,
    QFrame, QSizePolicy, QFileDialog, QMessageBox, QSlider
)
from PyQt6.QtCore import Qt, QTimer, QSize, QEvent, QUrl
from PyQt6.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent, QPainter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import json
from PIL import Image
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
import vlc

class VideoWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: black;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame = None
        self._aspect_ratio = None

    def set_frame(self, frame):
        self._frame = frame
        if frame is not None:
            h, w = frame.shape[:2]
            self._aspect_ratio = w / h
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._frame is not None:
            h, w = self._frame.shape[:2]
            label_w = self.width()
            label_h = self.height()
            # Calculate target size
            frame_ratio = w / h
            label_ratio = label_w / label_h
            if frame_ratio > label_ratio:
                # Fit to width
                new_w = label_w
                new_h = int(label_w / frame_ratio)
            else:
                # Fit to height
                new_h = label_h
                new_w = int(label_h * frame_ratio)
            # Use PIL for resizing for best quality
            pil_img = Image.fromarray(self._frame)
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
            rgb_img = pil_img.convert('RGB')
            img_data = rgb_img.tobytes('raw', 'RGB')
            image = QImage(img_data, new_w, new_h, 3 * new_w, QImage.Format.Format_RGB888)
            # Center the image
            x = (label_w - new_w) // 2
            y = (label_h - new_h) // 2
            painter = QPainter(self)
            painter.drawImage(x, y, image)
            painter.end()

class PitchAccentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize state variables
        self.is_playing_thread_active = False
        self.native_audio_path = None
        self.user_audio_path = os.path.join(tempfile.gettempdir(), "user_recording.wav")
        self.playing = False
        self.recording = False
        self.last_native_loop_time = None
        self.overlay_patch = None
        self.record_overlay = None
        self.selection_patch = None
        self._loop_start = 0.0
        self._loop_end = None
        self.user_playing = False
        self.show_video = True
        self.max_recording_time = 10  # seconds
        self.smoothing = 0
        self.current_rotation = 0  # Initialize rotation to 0 degrees
        self.original_frame = None  # Initialize original_frame to None
        
        # Get audio devices
        self.input_devices = [d for d in sd.query_devices() if d['max_input_channels'] > 0]
        self.output_devices = [d for d in sd.query_devices() if d['max_output_channels'] > 0]
        
        # Setup UI
        self.setup_ui()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Setup locks
        self.selection_lock = threading.Lock()
        self.playback_lock = threading.Lock()
        self.recording_lock = threading.Lock()

    def setup_ui(self):
        """Initialize the main UI components"""
        self.setWindowTitle("Pitch Accent Trainer")
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create top control bar
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        
        # Add device selectors
        input_label = QLabel("Input Device:")
        self.input_selector = QComboBox()
        self.input_selector.addItems([d['name'] for d in self.input_devices])
        
        output_label = QLabel("Output Device:")
        self.output_selector = QComboBox()
        self.output_selector.addItems([d['name'] for d in self.output_devices])
        
        # Add loop info label
        self.loop_info_label = QLabel("Loop: Full clip")
        
        # Add widgets to top layout
        top_layout.addWidget(input_label)
        top_layout.addWidget(self.input_selector)
        top_layout.addWidget(output_label)
        top_layout.addWidget(self.output_selector)
        top_layout.addStretch()
        top_layout.addWidget(self.loop_info_label)
        
        # Add top bar to main layout
        main_layout.addWidget(top_bar)
        
        # Create video and controls section
        video_controls = QWidget()
        video_controls_layout = QHBoxLayout(video_controls)
        
        # Create video display container
        video_container = QWidget()
        video_container_layout = QVBoxLayout(video_container)
        
        # Create video display
        self.vlc_instance = vlc.Instance()
        self.vlc_player = self.vlc_instance.media_player_new()
        self.video_widget = QLabel()  # Changed from QWidget to QLabel
        self.video_widget.setMinimumSize(400, 300)
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center the image
        self.vlc_player.set_hwnd(int(self.video_widget.winId()))  # Windows only
        
        # Add video controls
        video_buttons = QHBoxLayout()
        self.rotate_left_btn = QPushButton("↺")
        self.rotate_left_btn.clicked.connect(lambda: self.rotate_video(-90))
        self.rotate_right_btn = QPushButton("↻")
        self.rotate_right_btn.clicked.connect(lambda: self.rotate_video(90))
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.is_paused = False
        
        video_buttons.addWidget(self.rotate_left_btn)
        video_buttons.addWidget(self.rotate_right_btn)
        video_buttons.addWidget(self.pause_btn)
        video_buttons.addStretch()
        
        video_container_layout.addWidget(self.video_widget)
        video_container_layout.addLayout(video_buttons)
        
        # Create controls section
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        
        # Native audio controls
        native_group = QFrame()
        native_group.setFrameStyle(QFrame.Shape.StyledPanel)
        native_layout = QVBoxLayout(native_group)
        
        native_label = QLabel("Native Audio")
        native_label.setStyleSheet("font-weight: bold;")
        self.clear_loop_btn = QPushButton("Clear Loop Selection")
        self.clear_loop_btn.clicked.connect(self.clear_selection)
        self.play_native_btn = QPushButton("Play Native")
        self.play_native_btn.setEnabled(False)
        self.loop_native_btn = QPushButton("Loop Native")
        self.loop_native_btn.setEnabled(False)
        self.stop_native_btn = QPushButton("Stop Native")
        self.stop_native_btn.setEnabled(False)
        
        native_layout.addWidget(native_label)
        native_layout.addWidget(self.clear_loop_btn)
        native_layout.addWidget(self.play_native_btn)
        native_layout.addWidget(self.loop_native_btn)
        native_layout.addWidget(self.stop_native_btn)
        
        # Recording indicator
        self.recording_indicator = QLabel("")
        self.recording_indicator.setStyleSheet("color: red; font-weight: bold; font-size: 16px;")
        self.recording_indicator.setVisible(False)
        controls_layout.addWidget(self.recording_indicator)
        
        # User audio controls
        user_group = QFrame()
        user_group.setFrameStyle(QFrame.Shape.StyledPanel)
        user_layout = QVBoxLayout(user_group)
        
        user_label = QLabel("User Audio")
        user_label.setStyleSheet("font-weight: bold;")
        # Recording indicator next to label
        user_label_row = QHBoxLayout()
        user_label_row.addWidget(user_label)
        self.recording_indicator = QLabel("")
        self.recording_indicator.setStyleSheet("color: red; font-weight: bold; font-size: 16px;")
        self.recording_indicator.setVisible(False)
        user_label_row.addWidget(self.recording_indicator)
        user_label_row.addStretch()
        user_layout.addLayout(user_label_row)
        self.record_btn = QPushButton("Record")
        self.record_btn.setEnabled(True)
        self.play_user_btn = QPushButton("Play User")
        self.play_user_btn.setEnabled(False)
        self.loop_user_btn = QPushButton("Loop User")
        self.loop_user_btn.setEnabled(False)
        self.stop_user_btn = QPushButton("Stop User")
        self.stop_user_btn.setEnabled(False)
        
        user_layout.addWidget(self.record_btn)
        user_layout.addWidget(self.play_user_btn)
        user_layout.addWidget(self.loop_user_btn)
        user_layout.addWidget(self.stop_user_btn)
        
        # Add groups to controls layout
        controls_layout.addWidget(native_group)
        controls_layout.addWidget(user_group)
        
        # Add file selection button
        self.select_file_btn = QPushButton("Select Video File")
        self.select_file_btn.clicked.connect(self.select_file)
        controls_layout.addWidget(self.select_file_btn)
        
        # Add video and controls to layout
        video_controls_layout.addWidget(video_container, stretch=2)
        video_controls_layout.addWidget(controls, stretch=1)
        
        # Add video controls section to main layout
        main_layout.addWidget(video_controls)
        
        # Create waveform display section
        waveform_section = QWidget()
        waveform_layout = QVBoxLayout(waveform_section)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Create subplots
        self.ax_native = self.figure.add_subplot(211)  # Native pitch
        self.ax_user = self.figure.add_subplot(212)    # User pitch
        
        # Configure subplots
        self.ax_native.set_ylabel('Hz')
        self.ax_native.set_title('Native Speaker (Raw Pitch)')
        self.ax_user.set_xlabel('Time (s)')
        self.ax_user.set_ylabel('Hz')
        self.ax_user.set_title('Your Recording (Raw Pitch)')
        
        # Add span selector for loop selection (on native pitch)
        self.span = SpanSelector(
            self.ax_native,
            self.on_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='blue'),
            interactive=True
        )
        
        # Add waveform section to main layout
        waveform_layout.addWidget(self.canvas)
        main_layout.addWidget(waveform_section)
        
        # Set window size based on screen resolution
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.75)  # 75% of screen width
        height = int(width * 0.6)  # Maintain aspect ratio
        self.resize(width, height)
        
        # Store dimensions for later use
        self.base_height = height
        self.landscape_height = int(height * 0.3)
        
        # Scale video dimensions proportionally
        scale = width / 1800
        self.portrait_video_width = int(400 * scale)
        self.landscape_video_height = int(300 * scale)
        self.max_video_width = int(800 * scale)
        self.max_video_height = int(800 * scale)

        # Connect button signals
        self.play_native_btn.clicked.connect(self.play_native)
        self.loop_native_btn.clicked.connect(self.loop_native)
        self.stop_native_btn.clicked.connect(self.stop_native)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.play_user_btn.clicked.connect(self.play_user)
        self.loop_user_btn.clicked.connect(self.loop_user)
        self.stop_user_btn.clicked.connect(self.stop_user)

        # Enable drag & drop
        self.setAcceptDrops(True)

    def signal_handler(self, sig, frame):
        """Handle Ctrl+C signal"""
        print("\nCtrl+C detected. Cleaning up...")
        self.close()

    def closeEvent(self, event):
        """Handle window close event"""
        print("Cleaning up...")
        try:
            # Stop any ongoing playback
            self.playing = False
            sd.stop()
            
            # Stop any ongoing recording
            self.recording = False
            
            # Clear video window if exists
            if hasattr(self, 'video_window'):
                self.video_window.close()
            
            # Destroy all matplotlib figures
            plt.close('all')
            
            event.accept()
        except Exception as e:
            print(f"Error during cleanup: {e}")
            event.accept()

    def on_select(self, xmin, xmax):
        """Handle span selection for loop points"""
        with self.selection_lock:
            self._loop_start = max(0.0, xmin)
            self._loop_end = xmax
            self.update_loop_info()
            self.redraw_waveform()

    def update_loop_info(self):
        """Update the loop information label"""
        if self._loop_end is None:
            self.loop_info_label.setText("Loop: Full clip")
        else:
            self.loop_info_label.setText(f"Loop: {self._loop_start:.2f}s - {self._loop_end:.2f}s")

    def redraw_waveform(self):
        """Redraw the native and user pitch curves with current loop selection"""
        # Safely stop playback timers and remove playback lines before redrawing
        self._cleanup_playback_lines()
        # Native pitch
        self.ax_native.clear()
        if hasattr(self, 'native_times') and hasattr(self, 'native_pitch') and hasattr(self, 'native_voiced'):
            x = self.native_times
            y = self.native_pitch
            voiced = self.native_voiced
            start = None
            for i in range(len(voiced)):
                if voiced[i] and start is None:
                    start = i
                elif (not voiced[i] or i == len(voiced) - 1) and start is not None:
                    end = i if not voiced[i] else i + 1
                    seg_len = end - start
                    if seg_len > 1:
                        seg_x = x[start:end]
                        seg_y = y[start:end]
                        if len(seg_x) > 3:
                            dense_x = np.linspace(seg_x[0], seg_x[-1], int((seg_x[-1] - seg_x[0]) / 0.003) + 1)
                            f = interp1d(seg_x, seg_y, kind='linear')
                            dense_y = f(dense_x)
                            self.ax_native.plot(
                                dense_x, dense_y, color='blue', linewidth=6, solid_capstyle='round', label='Native' if start == 0 else "")
                        else:
                            self.ax_native.plot(
                                seg_x, seg_y, color='blue', linewidth=6, solid_capstyle='round', label='Native' if start == 0 else "")
                    start = None
            if self._loop_end is not None:
                self.ax_native.axvspan(self._loop_start, self._loop_end, color='blue', alpha=0.2)
            self.ax_native.set_ylabel('Hz')
            self.ax_native.set_title('Native Speaker (Raw Pitch)')
            if hasattr(self, 'native_pitch') and np.any(self.native_voiced):
                max_pitch = np.max(self.native_pitch[self.native_voiced])
                self.ax_native.set_ylim(0, max(500, max_pitch + 20))
            else:
                self.ax_native.set_ylim(0, 500)
            self.ax_native.legend()
            self.ax_native.grid(True)
        # User pitch
        self.ax_user.clear()
        if hasattr(self, 'user_times') and hasattr(self, 'user_pitch') and hasattr(self, 'user_voiced'):
            x = self.user_times
            y = self.user_pitch
            voiced = self.user_voiced
            start = None
            for i in range(len(voiced)):
                if voiced[i] and start is None:
                    start = i
                elif (not voiced[i] or i == len(voiced) - 1) and start is not None:
                    end = i if not voiced[i] else i + 1
                    seg_len = end - start
                    if seg_len > 1:
                        seg_x = x[start:end]
                        seg_y = y[start:end]
                        if len(seg_x) > 3:
                            dense_x = np.linspace(seg_x[0], seg_x[-1], int((seg_x[-1] - seg_x[0]) / 0.003) + 1)
                            f = interp1d(seg_x, seg_y, kind='linear')
                            dense_y = f(dense_x)
                            self.ax_user.plot(
                                dense_x, dense_y, color='orange', linewidth=6, solid_capstyle='round', label='User' if start == 0 else "")
                        else:
                            self.ax_user.plot(
                                seg_x, seg_y, color='orange', linewidth=6, solid_capstyle='round', label='User' if start == 0 else "")
                    start = None
            # Set y-limits: if user pitch goes above 500 Hz, set ylim to max(500, max_user_pitch + 20)
            if hasattr(self, 'user_pitch') and np.any(self.user_voiced):
                max_user_pitch = np.max(self.user_pitch[self.user_voiced])
                self.ax_user.set_ylim(0, max(500, max_user_pitch + 20))
            else:
                self.ax_user.set_ylim(0, 500)
            self.ax_user.legend()
        self.ax_user.set_xlabel('Time (s)')
        self.ax_user.set_ylabel('Hz')
        self.ax_user.set_title('Your Recording (Raw Pitch)')
        self.ax_user.grid(True)
        self.figure.tight_layout()
        self.canvas.draw()

    def select_file(self):
        """Handle file selection"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*.*)"
        )
        
        if file_path:
            try:
                self.load_file(file_path)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

    def load_file(self, file_path):
        """Load and process the selected file"""
        self._cleanup_playback_lines()
        ext = os.path.splitext(file_path)[1].lower()
        audio_path = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
        if ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
            video = VideoFileClip(file_path)
            video.audio.write_audiofile(audio_path)
            # Set video file for VLC
            media = self.vlc_instance.media_new(file_path)
            self.vlc_player.set_media(media)
            self.video_widget.show()  # Always show video widget
            # Store the first frame for rotation
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            if ret:
                self.original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_rotation = 0
                self.resize_video_display()  # Show first frame immediately
            cap.release()
        elif ext in [".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a"]:
            audio = AudioFileClip(file_path)
            audio.write_audiofile(audio_path)
            self.vlc_player.set_media(None)
            self.video_widget.hide()
        else:
            raise ValueError("Unsupported file type.")
        
        # Store paths
        self.native_audio_path = audio_path
        self.video_path = file_path
        
        # Process audio
        self.process_audio()
        
        # Enable controls
        self.play_native_btn.setEnabled(True)
        self.loop_native_btn.setEnabled(True)
        self.stop_native_btn.setEnabled(True)
        self.record_btn.setEnabled(True)

    def process_audio(self):
        """Process the audio file to extract waveform and pitch"""
        self._cleanup_playback_lines()
        sound = parselmouth.Sound(self.native_audio_path)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_times = pitch.xs()
        # Use only raw voiced points
        voiced = pitch_values > 0
        self.native_times = pitch_times
        self.native_pitch = pitch_values
        self.native_voiced = voiced
        self.redraw_waveform()

    def play_native(self):
        """Play native audio and video (VLC)"""
        try:
            if self.playing:
                return
            self.playing = True
            self.is_paused = False
            self.play_native_btn.setEnabled(False)
            self.loop_native_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.pause_btn.setText("Pause")
            # Start VLC video playback
            if self.video_path and hasattr(self, 'original_frame'):
                print("Starting VLC playback...")
                self.vlc_player.audio_set_mute(True)  # Mute VLC audio to prevent double audio
                self.vlc_player.play()
                # Display the first frame immediately
                self.resize_video_display()
            self.start_native_playback_with_timer()
        except Exception as e:
            print(f"Error in play_native: {e}")
            import traceback
            traceback.print_exc()
            self.playing = False
            self.play_native_btn.setEnabled(True)
            self.loop_native_btn.setEnabled(True)

    def start_native_playback_with_timer(self):
        import time
        from PyQt6.QtCore import QTimer
        self._cleanup_playback_lines()
        if hasattr(self, 'native_playback_line') and self.native_playback_line:
            self.native_playback_line.remove()
        self.native_playback_line = self.ax_native.axvline(0, color='red', linestyle='--', linewidth=2)
        self.canvas.draw_idle()
        self.native_playback_start_time = time.time()
        self._paused_elapsed = 0
        # Determine playback range
        try:
            import numpy as np
            import scipy.io.wavfile as wavfile
            sample_rate, audio_data = wavfile.read(self.native_audio_path)
            if self._loop_end is not None and self._loop_end > self._loop_start:
                start_sample = int(self._loop_start * sample_rate)
                end_sample = int(self._loop_end * sample_rate)
                audio_data = audio_data[start_sample:end_sample]
                duration = (end_sample - start_sample) / sample_rate
                offset = self._loop_start
            else:
                duration = len(audio_data) / sample_rate
                offset = 0
        except Exception:
            duration = 0
            offset = 0
        self.native_playback_timer = QTimer()
        self.native_playback_timer.setInterval(20)
        def update_playback_line_and_video():
            if self.is_paused:
                self._paused_elapsed = time.time() - self.native_playback_start_time
                return
            elapsed = time.time() - self.native_playback_start_time
            pos = offset + elapsed
            try:
                if self.native_playback_line and self.native_playback_line in self.ax_native.lines:
                    self.native_playback_line.set_xdata([pos, pos])
                    self.canvas.draw_idle()
            except Exception:
                pass
            # --- Video sync logic ---
            if hasattr(self, 'cap') and hasattr(self, 'video_path') and self.cap.isOpened():
                import cv2
                video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                video_duration = frame_count / video_fps if video_fps > 0 else 0
                if video_duration > 0:
                    video_time = min(elapsed, video_duration)
                    frame_idx = int(video_time * video_fps)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = self.cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if self.current_rotation != 0:
                            if self.current_rotation == 90:
                                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                            elif self.current_rotation == 180:
                                frame = cv2.rotate(frame, cv2.ROTATE_180)
                            elif self.current_rotation == 270:
                                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        self.video_widget.setPixmap(QPixmap.fromImage(QImage(frame.tobytes(), frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888)))
            # --- End video sync logic ---
            if elapsed >= duration or not self.playing:
                try:
                    self.native_playback_timer.stop()
                except Exception:
                    pass
                try:
                    if self.native_playback_line and self.native_playback_line in self.ax_native.lines:
                        self.native_playback_line.remove()
                except Exception:
                    pass
                self.native_playback_line = None
                try:
                    self.canvas.draw_idle()
                except Exception:
                    pass
                self.pause_btn.setEnabled(False)
        self.native_playback_timer.timeout.connect(update_playback_line_and_video)
        self.native_playback_timer.start()
        import threading
        threading.Thread(target=self._play_native_thread, daemon=True).start()

    def _play_native_thread(self):
        try:
            print("Starting native audio playback thread...")
            import numpy as np
            import scipy.io.wavfile as wavfile
            sample_rate, audio_data = wavfile.read(self.native_audio_path)
            if self._loop_end is not None and self._loop_end > self._loop_start:
                start_sample = int(self._loop_start * sample_rate)
                end_sample = int(self._loop_end * sample_rate)
                audio_data = audio_data[start_sample:end_sample]
            duration = len(audio_data) / sample_rate
            # Stop any existing playback before starting new one
            sd.stop()
            sd.wait()  # Wait for stop to complete
            print("Playing audio...")
            sd.play(audio_data, sample_rate)
            sd.wait()
            print("Audio playback completed")
            self.playing = False
        except Exception as e:
            print(f"Error during playback: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.playing = False
            self.play_native_btn.setEnabled(True)
            self.loop_native_btn.setEnabled(True)

    def loop_native(self):
        """Loop native audio"""
        if self.playing:
            return
            
        self.playing = True
        self.play_native_btn.setEnabled(False)
        self.loop_native_btn.setEnabled(False)
        
        # Start loop playback in a separate thread
        threading.Thread(target=self._loop_native_thread, daemon=True).start()

    def _loop_native_thread(self):
        """Thread function for native audio loop playback"""
        try:
            # Load audio data
            sample_rate, audio_data = wavfile.read(self.native_audio_path)
            
            # Apply loop if set
            if self._loop_end is not None:
                start_sample = int(self._loop_start * sample_rate)
                end_sample = int(self._loop_end * sample_rate)
                audio_data = audio_data[start_sample:end_sample]
            
            # Loop playback
            while self.playing:
                sd.play(audio_data, sample_rate)
                sd.wait()
                
        except Exception as e:
            print(f"Error during loop playback: {e}")
        finally:
            self.playing = False
            self.play_native_btn.setEnabled(True)
            self.loop_native_btn.setEnabled(True)

    def stop_native(self):
        """Stop native audio and video playback"""
        self.playing = False
        sd.stop()
        self._cleanup_playback_lines()
        # Stop VLC video playback but keep the frame visible
        self.vlc_player.stop()
        if hasattr(self, 'original_frame'):
            self.resize_video_display()

    def toggle_recording(self):
        """Toggle recording state"""
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """Start recording user audio"""
        if self.recording:
            return
        self.recording = True
        self.record_btn.setText("Stop Recording")
        self.play_user_btn.setEnabled(False)
        self.loop_user_btn.setEnabled(False)
        self.recording_indicator.setText("● Recording...")
        self.recording_indicator.setVisible(True)
        try:
            print("[DEBUG] Starting _record_thread...")
            threading.Thread(target=self._record_thread, daemon=True).start()
        except Exception as e:
            print(f"[DEBUG] Failed to start _record_thread: {e}")

    def _record_thread(self):
        """Thread function for recording"""
        print("[DEBUG] _record_thread started")
        try:
            try:
                # Get selected input device
                device_id = self.input_selector.currentIndex()
                print(f"[DEBUG] Using input device index: {device_id}")
                # Start recording
                recording = sd.rec(
                    int(self.max_recording_time * 44100),
                    samplerate=44100,
                    channels=1,
                    device=device_id
                )
                print("[DEBUG] sd.rec called, entering while loop...")
                # Wait for recording to complete or stop
                while self.recording:
                    print("[DEBUG] ...recording in progress...")
                    time.sleep(0.1)
                print("[DEBUG] Exited while loop, calling sd.stop()")
                # Stop recording
                sd.stop()
                sd.wait()
                print("[DEBUG] sd.stop() called, about to write file...")
                # Always process and save after recording stops
                print(f"[DEBUG] recording.shape: {recording.shape}, dtype: {recording.dtype}")
                print(f"[DEBUG] recording min: {np.min(recording)}, max: {np.max(recording)}")
                try:
                    # Trim trailing zeros (silence)
                    abs_rec = np.abs(recording.squeeze())
                    nonzero = np.where(abs_rec > 1e-4)[0]
                    if len(nonzero) > 0:
                        last = nonzero[-1] + 1
                        trimmed = recording[:last]
                    else:
                        trimmed = recording
                    # Convert float32 [-1, 1] to int16 for wavfile.write
                    recording_int16 = np.int16(np.clip(trimmed, -1, 1) * 32767)
                    print(f"[DEBUG] recording_int16.shape: {recording_int16.shape}, dtype: {recording_int16.dtype}")
                    wavfile.write(self.user_audio_path, 44100, recording_int16)
                    print(f"[DEBUG] Saved user recording to: {self.user_audio_path}")
                    if os.path.exists(self.user_audio_path):
                        print(f"[DEBUG] User recording file size: {os.path.getsize(self.user_audio_path)} bytes")
                    else:
                        print("[DEBUG] User recording file not found!")
                except Exception as e:
                    print(f"[DEBUG] Exception during wavfile.write: {e}")
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.critical(self, "Error", f"Exception during saving recording: {e}")
                self.process_user_audio()
                self.play_user_btn.setEnabled(True)
                self.loop_user_btn.setEnabled(True)
            except Exception as thread_inner_e:
                print(f"[DEBUG] Exception in _record_thread inner block: {thread_inner_e}")
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", f"Exception in recording thread: {thread_inner_e}")
        except Exception as thread_outer_e:
            print(f"[DEBUG] Exception in _record_thread outer block: {thread_outer_e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Exception in recording thread (outer): {thread_outer_e}")
        finally:
            self.recording = False
            self.record_btn.setText("Record")
            self.recording_indicator.setVisible(False)

    def stop_recording(self):
        """Stop recording user audio"""
        self.recording = False
        self.recording_indicator.setVisible(False)

    def play_user(self):
        """Play user recording"""
        if self.user_playing:
            return
        self.user_playing = True
        self.play_user_btn.setEnabled(False)
        self.loop_user_btn.setEnabled(False)
        self.stop_user_btn.setEnabled(True)
        # Start playback with timer for moving line
        self.start_user_playback_with_timer()

    def start_user_playback_with_timer(self):
        import time
        from PyQt6.QtCore import QTimer
        # Prevent overlapping playbacks/timers
        self._cleanup_playback_lines()
        if hasattr(self, 'user_playback_line') and self.user_playback_line:
            self.user_playback_line.remove()
        self.user_playback_line = self.ax_user.axvline(0, color='red', linestyle='--', linewidth=2)
        self.canvas.draw_idle()
        self.user_playback_start_time = time.time()
        try:
            import numpy as np
            import scipy.io.wavfile as wavfile
            sample_rate, audio_data = wavfile.read(self.user_audio_path)
            duration = len(audio_data) / sample_rate
        except Exception:
            duration = 0
        self.user_playback_timer = QTimer()
        self.user_playback_timer.setInterval(20)
        def update_playback_line():
            elapsed = time.time() - self.user_playback_start_time
            pos = elapsed
            try:
                if self.user_playback_line and self.user_playback_line in self.ax_user.lines:
                    self.user_playback_line.set_xdata([pos, pos])
                    self.canvas.draw_idle()
            except Exception:
                pass
            if elapsed >= duration or not self.user_playing:
                try:
                    self.user_playback_timer.stop()
                except Exception:
                    pass
                try:
                    if self.user_playback_line and self.user_playback_line in self.ax_user.lines:
                        self.user_playback_line.remove()
                except Exception:
                    pass
                self.user_playback_line = None
                try:
                    self.canvas.draw_idle()
                except Exception:
                    pass
        self.user_playback_timer.timeout.connect(update_playback_line)
        self.user_playback_timer.start()
        # Start playback in a background thread
        import threading
        threading.Thread(target=self._play_user_thread, daemon=True).start()

    def _play_user_thread(self):
        try:
            sample_rate, audio_data = wavfile.read(self.user_audio_path)
            # Trim trailing zeros (silence) for playback
            abs_rec = np.abs(audio_data.squeeze())
            nonzero = np.where(abs_rec > 10)[0]  # int16 threshold
            if len(nonzero) > 0:
                last = nonzero[-1] + 1
                audio_data = audio_data[:last]
            sd.play(audio_data, sample_rate)
            sd.wait()
            self.user_playing = False
        except Exception as e:
            print(f"Error during playback: {e}")
        finally:
            self.user_playing = False
            self.play_user_btn.setEnabled(True)
            self.loop_user_btn.setEnabled(True)
            self.stop_user_btn.setEnabled(False)

    def loop_user(self):
        """Loop user recording"""
        if self.user_playing:
            return
        self.user_playing = True
        self.play_user_btn.setEnabled(False)
        self.loop_user_btn.setEnabled(False)
        self.stop_user_btn.setEnabled(True)
        # Start loop playback in a separate thread
        self.start_user_loop_playback_with_timer()

    def start_user_loop_playback_with_timer(self):
        import time
        from PyQt6.QtCore import QTimer
        self._cleanup_playback_lines()
        if hasattr(self, 'user_playback_line') and self.user_playback_line:
            self.user_playback_line.remove()
        self.user_playback_line = self.ax_user.axvline(0, color='red', linestyle='--', linewidth=2)
        self.canvas.draw_idle()
        self.user_playback_start_time = time.time()
        try:
            import numpy as np
            import scipy.io.wavfile as wavfile
            sample_rate, audio_data = wavfile.read(self.user_audio_path)
            abs_rec = np.abs(audio_data.squeeze())
            nonzero = np.where(abs_rec > 10)[0]
            if len(nonzero) > 0:
                last = nonzero[-1] + 1
                audio_data = audio_data[:last]
            duration = len(audio_data) / sample_rate
        except Exception:
            duration = 0
        self.user_playback_timer = QTimer()
        self.user_playback_timer.setInterval(20)
        def update_playback_line():
            elapsed = (time.time() - self.user_playback_start_time) % duration if duration > 0 else 0
            pos = elapsed
            try:
                if self.user_playback_line and self.user_playback_line in self.ax_user.lines:
                    self.user_playback_line.set_xdata([pos, pos])
                    self.canvas.draw_idle()
            except Exception:
                pass
            if not self.user_playing:
                try:
                    self.user_playback_timer.stop()
                except Exception:
                    pass
                try:
                    if self.user_playback_line and self.user_playback_line in self.ax_user.lines:
                        self.user_playback_line.remove()
                except Exception:
                    pass
                self.user_playback_line = None
                try:
                    self.canvas.draw_idle()
                except Exception:
                    pass
        self.user_playback_timer.timeout.connect(update_playback_line)
        self.user_playback_timer.start()
        # Start loop playback in a background thread
        import threading
        threading.Thread(target=self._loop_user_thread, daemon=True).start()

    def _loop_user_thread(self):
        try:
            sample_rate, audio_data = wavfile.read(self.user_audio_path)
            # Trim trailing zeros (silence) for playback
            abs_rec = np.abs(audio_data.squeeze())
            nonzero = np.where(abs_rec > 10)[0]  # int16 threshold
            if len(nonzero) > 0:
                last = nonzero[-1] + 1
                audio_data = audio_data[:last]
            while self.user_playing:
                sd.play(audio_data, sample_rate)
                sd.wait()
        except Exception as e:
            print(f"Error during loop playback: {e}")
        finally:
            self.user_playing = False
            self.play_user_btn.setEnabled(True)
            self.loop_user_btn.setEnabled(True)
            self.stop_user_btn.setEnabled(False)

    def stop_user(self):
        """Stop user audio playback"""
        self.user_playing = False
        sd.stop()
        self.stop_user_btn.setEnabled(False)
        self._cleanup_playback_lines()

    def process_user_audio(self):
        """Process the user recording to extract and plot pitch curve"""
        self._cleanup_playback_lines()
        try:
            print(f"[DEBUG] Processing user audio: {self.user_audio_path}")
            if not os.path.exists(self.user_audio_path):
                print("[DEBUG] User audio file does not exist!")
                return
            sound = parselmouth.Sound(self.user_audio_path)
            pitch = sound.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            pitch_times = pitch.xs()
            voiced = pitch_values > 0
            self.user_times = pitch_times
            self.user_pitch = pitch_values
            self.user_voiced = voiced
            self.redraw_waveform()
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Error processing user audio: {e}")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a"]:
                self.load_file(file_path)
                break

    def clear_selection(self):
        with self.selection_lock:
            self._loop_start = 0.0
            self._loop_end = None
            self.update_loop_info()
            # Remove selection patch if present
            if hasattr(self, 'selection_patch') and self.selection_patch is not None:
                try:
                    self.selection_patch.remove()
                except Exception:
                    pass
                self.selection_patch = None
            # Clear the span selector (removes selection rectangle)
            if hasattr(self, 'span') and self.span is not None:
                try:
                    self.span.clear()
                except Exception:
                    pass
            self.redraw_waveform()
            self.canvas.draw_idle()

    def _cleanup_playback_lines(self):
        # Stop native playback timer and remove line
        try:
            if hasattr(self, 'native_playback_timer') and self.native_playback_timer is not None:
                self.native_playback_timer.stop()
                self.native_playback_timer = None
        except Exception:
            pass
        try:
            if hasattr(self, 'native_playback_line') and self.native_playback_line is not None:
                if self.native_playback_line in self.ax_native.lines:
                    self.native_playback_line.remove()
                self.native_playback_line = None
        except Exception:
            pass
        # Stop user playback timer and remove line
        try:
            if hasattr(self, 'user_playback_timer') and self.user_playback_timer is not None:
                self.user_playback_timer.stop()
                self.user_playback_timer = None
        except Exception:
            pass
        try:
            if hasattr(self, 'user_playback_line') and self.user_playback_line is not None:
                if self.user_playback_line in self.ax_user.lines:
                    self.user_playback_line.remove()
                self.user_playback_line = None
        except Exception:
            pass

    def toggle_pause(self):
        """Pause or resume audio and video playback"""
        if not self.playing:
            return
        self.is_paused = not self.is_paused
        if self.is_paused:
            sd.stop()
            self.pause_btn.setText("Resume")
            self.vlc_player.pause()
        else:
            self.native_playback_start_time = time.time() - self._paused_elapsed
            self.pause_btn.setText("Pause")
            self.vlc_player.set_pause(0)
            # Resume audio playback from where it was paused
            import threading
            threading.Thread(target=self._play_native_thread_resume, daemon=True).start()

    def _play_native_thread_resume(self):
        try:
            import numpy as np
            import scipy.io.wavfile as wavfile
            sample_rate, audio_data = wavfile.read(self.native_audio_path)
            if self._loop_end is not None and self._loop_end > self._loop_start:
                start_sample = int(self._loop_start * sample_rate)
                end_sample = int(self._loop_end * sample_rate)
                audio_data = audio_data[start_sample:end_sample]
            duration = len(audio_data) / sample_rate
            # Resume from paused position
            start_sample = int(self._paused_elapsed * sample_rate)
            audio_data = audio_data[start_sample:]
            sd.play(audio_data, sample_rate)
            sd.wait()
            self.playing = False
        except Exception as e:
            print(f"Error during resume playback: {e}")

    def rotate_video(self, angle):
        """Rotate video display"""
        if not hasattr(self, 'original_frame'):
            return
            
        self.current_rotation = (self.current_rotation + angle) % 360
        self.resize_video_display()

    def resize_video_display(self):
        """Display the last frame at widget size, let Qt scale"""
        try:
            if not hasattr(self, 'original_frame') or self.original_frame is None:
                print("No original frame available")
                return
            print("Resizing video display...")
            frame = self.original_frame.copy()
            if self.current_rotation != 0:
                if self.current_rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif self.current_rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif self.current_rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # Convert frame to QImage
            height, width = frame.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            # Convert to QPixmap and scale to fit widget
            pixmap = QPixmap.fromImage(q_image)
            widget_size = self.video_widget.size()
            scaled_pixmap = pixmap.scaled(widget_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.video_widget.setPixmap(scaled_pixmap)
            print("Video display updated successfully")
        except Exception as e:
            print(f"Error in resize_video_display: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PitchAccentApp()
    window.show()
    sys.exit(app.exec()) 