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
from moviepy.editor import AudioFileClip
from scipy.interpolate import interp1d
from scipy.signal import medfilt, savgol_filter
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QCheckBox, QLineEdit,
    QFrame, QSizePolicy, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

class PitchAccentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize state variables
        self.is_playing_thread_active = False
        self.native_audio_path = None
        self.user_audio_path = os.path.join(tempfile.gettempdir(), "user_recording.wav")
        self.playing = False
        self.recording = False
        self.pending_recording = False
        self.last_native_loop_time = None
        self.overlay_patch = None
        self.record_overlay = None
        self.selection_patch = None
        self._loop_start = 0.0
        self._loop_end = None
        self.user_playing = False
        self.show_video = True
        self.max_recording_time = 10  # seconds
        
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
        
        # Create video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(400, 300)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("background-color: black;")
        
        # Create controls section
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        
        # Native audio controls
        native_group = QFrame()
        native_group.setFrameStyle(QFrame.Shape.StyledPanel)
        native_layout = QVBoxLayout(native_group)
        
        native_label = QLabel("Native Audio")
        native_label.setStyleSheet("font-weight: bold;")
        self.play_native_btn = QPushButton("Play Native")
        self.play_native_btn.setEnabled(False)
        self.loop_native_btn = QPushButton("Loop Native")
        self.loop_native_btn.setEnabled(False)
        self.stop_native_btn = QPushButton("Stop Native")
        self.stop_native_btn.setEnabled(False)
        
        native_layout.addWidget(native_label)
        native_layout.addWidget(self.play_native_btn)
        native_layout.addWidget(self.loop_native_btn)
        native_layout.addWidget(self.stop_native_btn)
        
        # User audio controls
        user_group = QFrame()
        user_group.setFrameStyle(QFrame.Shape.StyledPanel)
        user_layout = QVBoxLayout(user_group)
        
        user_label = QLabel("User Audio")
        user_label.setStyleSheet("font-weight: bold;")
        self.record_btn = QPushButton("Record")
        self.record_btn.setEnabled(False)
        self.play_user_btn = QPushButton("Play User")
        self.play_user_btn.setEnabled(False)
        self.loop_user_btn = QPushButton("Loop User")
        self.loop_user_btn.setEnabled(False)
        self.stop_user_btn = QPushButton("Stop User")
        self.stop_user_btn.setEnabled(False)
        
        user_layout.addWidget(user_label)
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
        video_controls_layout.addWidget(self.video_label, stretch=2)
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
        self.ax1 = self.figure.add_subplot(211)  # Waveform
        self.ax2 = self.figure.add_subplot(212)  # Pitch contour
        
        # Configure subplots
        self.ax1.set_ylabel('Amplitude')
        self.ax1.set_title('Waveform')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Pitch (Hz)')
        self.ax2.set_title('Pitch Contour')
        
        # Add span selector for loop selection
        self.span = SpanSelector(
            self.ax1,
            self.on_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.5, facecolor='tab:blue'),
            interactive=True,
            drag_from_anywhere=True
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
            self.pending_recording = False
            
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
        """Redraw the waveform with current loop selection"""
        if not hasattr(self, 'times') or not hasattr(self, 'waveform'):
            return
            
        self.ax1.clear()
        self.ax1.plot(self.times, self.waveform, 'b-', alpha=0.5)
        self.ax1.set_ylabel('Amplitude')
        self.ax1.set_title('Waveform')
        
        if self._loop_end is not None:
            self.ax1.axvspan(self._loop_start, self._loop_end, color='blue', alpha=0.2)
        
        self.ax2.clear()
        self.ax2.plot(self.times, self.pitch_contour, 'r-', alpha=0.5)
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Pitch (Hz)')
        self.ax2.set_title('Pitch Contour')
        
        if self._loop_end is not None:
            self.ax2.axvspan(self._loop_start, self._loop_end, color='blue', alpha=0.2)
        
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
        # Extract audio from video
        video = AudioFileClip(file_path)
        audio_path = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
        video.audio.write_audiofile(audio_path)
        
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
        
        # Start video display
        self.start_video_display()

    def process_audio(self):
        """Process the audio file to extract waveform and pitch"""
        # Load audio using parselmouth
        sound = parselmouth.Sound(self.native_audio_path)
        
        # Get waveform data
        self.times = np.arange(len(sound.values[0])) / sound.sampling_frequency
        self.waveform = sound.values[0]
        
        # Extract pitch
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        
        # Interpolate pitch values to match waveform length
        pitch_times = np.arange(len(pitch_values)) * pitch.time_step
        f = interp1d(pitch_times, pitch_values, bounds_error=False, fill_value=0)
        self.pitch_contour = f(self.times)
        
        # Apply smoothing
        self.pitch_contour = savgol_filter(self.pitch_contour, 51, 3)
        
        # Redraw waveform
        self.redraw_waveform()

    def start_video_display(self):
        """Start video display in a separate window"""
        if hasattr(self, 'video_window'):
            self.video_window.close()
        
        self.video_window = QWidget()
        self.video_window.setWindowTitle("Video Display")
        self.video_window.setWindowFlags(Qt.WindowType.Window)
        
        layout = QVBoxLayout(self.video_window)
        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_display)
        
        # Set video window size
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.3)  # 30% of screen width
        height = int(width * 0.75)  # 4:3 aspect ratio
        self.video_window.resize(width, height)
        
        # Start video capture
        self.cap = cv2.VideoCapture(self.video_path)
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_video_frame)
        self.video_timer.start(33)  # ~30 FPS
        
        self.video_window.show()

    def update_video_frame(self):
        """Update the video frame"""
        if not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return
        
        # Convert frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame to fit display
        display_size = self.video_display.size()
        frame = cv2.resize(frame, (display_size.width(), display_size.height()))
        
        # Convert to QImage and display
        image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format.Format_RGB888)
        self.video_display.setPixmap(QPixmap.fromImage(image))

    def play_native(self):
        """Play native audio"""
        if self.playing:
            return
            
        self.playing = True
        self.play_native_btn.setEnabled(False)
        self.loop_native_btn.setEnabled(False)
        
        # Start playback in a separate thread
        threading.Thread(target=self._play_native_thread, daemon=True).start()

    def _play_native_thread(self):
        """Thread function for native audio playback"""
        try:
            # Load audio data
            sample_rate, audio_data = wavfile.read(self.native_audio_path)
            
            # Apply loop if set
            if self._loop_end is not None:
                start_sample = int(self._loop_start * sample_rate)
                end_sample = int(self._loop_end * sample_rate)
                audio_data = audio_data[start_sample:end_sample]
            
            # Play audio
            sd.play(audio_data, sample_rate)
            sd.wait()
            
        except Exception as e:
            print(f"Error during playback: {e}")
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
        """Stop native audio playback"""
        self.playing = False
        sd.stop()

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
        
        # Start recording in a separate thread
        threading.Thread(target=self._record_thread, daemon=True).start()

    def _record_thread(self):
        """Thread function for recording"""
        try:
            # Get selected input device
            device_id = self.input_selector.currentIndex()
            
            # Start recording
            recording = sd.rec(
                int(self.max_recording_time * 44100),
                samplerate=44100,
                channels=1,
                device=device_id
            )
            
            # Wait for recording to complete or stop
            while self.recording and not self.pending_recording:
                time.sleep(0.1)
            
            # Stop recording
            sd.stop()
            
            # Save recording if not pending
            if not self.pending_recording:
                wavfile.write(self.user_audio_path, 44100, recording)
                self.play_user_btn.setEnabled(True)
                self.loop_user_btn.setEnabled(True)
            
        except Exception as e:
            print(f"Error during recording: {e}")
        finally:
            self.recording = False
            self.record_btn.setText("Record")
            self.pending_recording = False

    def stop_recording(self):
        """Stop recording user audio"""
        self.recording = False
        self.pending_recording = True

    def play_user(self):
        """Play user recording"""
        if self.user_playing:
            return
            
        self.user_playing = True
        self.play_user_btn.setEnabled(False)
        self.loop_user_btn.setEnabled(False)
        
        # Start playback in a separate thread
        threading.Thread(target=self._play_user_thread, daemon=True).start()

    def _play_user_thread(self):
        """Thread function for user audio playback"""
        try:
            # Load audio data
            sample_rate, audio_data = wavfile.read(self.user_audio_path)
            
            # Play audio
            sd.play(audio_data, sample_rate)
            sd.wait()
            
        except Exception as e:
            print(f"Error during playback: {e}")
        finally:
            self.user_playing = False
            self.play_user_btn.setEnabled(True)
            self.loop_user_btn.setEnabled(True)

    def loop_user(self):
        """Loop user recording"""
        if self.user_playing:
            return
            
        self.user_playing = True
        self.play_user_btn.setEnabled(False)
        self.loop_user_btn.setEnabled(False)
        
        # Start loop playback in a separate thread
        threading.Thread(target=self._loop_user_thread, daemon=True).start()

    def _loop_user_thread(self):
        """Thread function for user audio loop playback"""
        try:
            # Load audio data
            sample_rate, audio_data = wavfile.read(self.user_audio_path)
            
            # Loop playback
            while self.user_playing:
                sd.play(audio_data, sample_rate)
                sd.wait()
                
        except Exception as e:
            print(f"Error during loop playback: {e}")
        finally:
            self.user_playing = False
            self.play_user_btn.setEnabled(True)
            self.loop_user_btn.setEnabled(True)

    def stop_user(self):
        """Stop user audio playback"""
        self.user_playing = False
        sd.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PitchAccentApp()
    window.show()
    sys.exit(app.exec()) 