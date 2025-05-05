import sys
import os
import tempfile
import time
import threading
import signal
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import parselmouth
from scipy.interpolate import interp1d
from scipy.signal import medfilt, savgol_filter
from moviepy.editor import AudioFileClip
import cv2
from PIL import Image, ImageQt

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QComboBox, QLabel, 
                            QFileDialog, QMessageBox, QFrame, QCheckBox,
                            QSpinBox, QDoubleSpinBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import matplotlib.pyplot as plt

class RecordingThread(QThread):
    update_signal = pyqtSignal(list)  # For sending audio frames
    finished_signal = pyqtSignal()    # For signaling completion
    error_signal = pyqtSignal(str)    # For error handling

    def __init__(self, input_device, duration, fs=22050):
        super().__init__()
        self.input_device = input_device
        self.duration = duration
        self.fs = fs
        self.running = False
        self.frames = []

    def run(self):
        try:
            self.running = True
            self.frames = []
            
            def callback(indata, frames, time, status):
                if status:
                    print(f'Recording error: {status}')
                if self.running:
                    self.frames.append(indata.copy())
                    self.update_signal.emit(self.frames)

            with sd.InputStream(samplerate=self.fs, 
                              device=self.input_device,
                              channels=1, 
                              callback=callback):
                while self.running and len(self.frames) * self.fs < self.duration * self.fs:
                    sd.sleep(100)

            if self.frames:
                self.finished_signal.emit()
            
        except Exception as e:
            self.error_signal.emit(str(e))
            
    def stop(self):
        self.running = False


class PitchAccentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pitch Accent Trainer")
        self.setAcceptDrops(True)  # Enable drag & drop
        
        # Initialize state variables
        self.native_audio_path = None
        self.user_audio_path = os.path.join(tempfile.gettempdir(), "user_recording.wav")
        self.playing = False
        self.recording = False
        self.pending_recording = False
        self.user_playing = False
        self.max_recording_time = 10  # seconds
        self.video_window = None
        self.current_rotation = 0
        self.original_frame = None
        
        # Get available audio devices
        self.input_devices = [d for d in sd.query_devices() if d['max_input_channels'] > 0]
        self.output_devices = [d for d in sd.query_devices() if d['max_output_channels'] > 0]
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
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
        
        top_layout.addWidget(input_label)
        top_layout.addWidget(self.input_selector)
        top_layout.addWidget(output_label)
        top_layout.addWidget(self.output_selector)
        top_layout.addStretch()
        top_layout.addWidget(self.loop_info_label)
        
        layout.addWidget(top_bar)
        
        # Create main content area
        content = QWidget()
        content_layout = QHBoxLayout(content)
        
        # Create plot frame
        plot_frame = QWidget()
        plot_layout = QVBoxLayout(plot_frame)
        
        # Setup matplotlib figures
        self.fig, (self.ax_native, self.ax_user) = plt.subplots(2, 1, figsize=(12, 4))
        self.canvas = FigureCanvasQTAgg(self.fig)
        plot_layout.addWidget(self.canvas)
        
        # Initialize plots
        self.setup_plots()
        
        content_layout.addWidget(plot_frame)
        
        # Create control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # Native controls section
        native_label = QLabel("Native Controls")
        self.clear_loop_button = QPushButton("Clear Loop Selection")
        self.load_native_button = QPushButton("Load Native Audio")
        self.show_video_checkbox = QCheckBox("Show Video Screenshot")
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        
        # Delay control
        delay_frame = QWidget()
        delay_layout = QHBoxLayout(delay_frame)
        delay_layout.addWidget(QLabel("Loop Delay:"))
        self.delay_spinbox = QSpinBox()
        self.delay_spinbox.setRange(0, 5000)
        self.delay_spinbox.setSuffix(" ms")
        delay_layout.addWidget(self.delay_spinbox)
        
        # Speed control
        speed_frame = QWidget()
        speed_layout = QVBoxLayout(speed_frame)
        speed_layout.addWidget(QLabel("Playback Speed:"))
        self.speed_label = QLabel("100%")
        self.speed_slider = QDoubleSpinBox()
        self.speed_slider.setRange(50, 100)
        self.speed_slider.setValue(100)
        self.speed_slider.setSingleStep(5)
        self.speed_slider.setSuffix("%")
        speed_layout.addWidget(self.speed_label)
        speed_layout.addWidget(self.speed_slider)
        
        # User controls section
        user_label = QLabel("User Controls")
        self.record_button = QPushButton("Record")
        self.play_user_button = QPushButton("Play Your Recording")
        self.play_user_button.setEnabled(False)
        
        # Add all controls to panel
        for w in [native_label, self.clear_loop_button, self.load_native_button,
                 self.show_video_checkbox, self.play_button, delay_frame,
                 speed_frame, user_label, self.record_button, self.play_user_button]:
            control_layout.addWidget(w)
        
        control_layout.addStretch()
        content_layout.addWidget(control_panel)
        
        layout.addWidget(content)
        
        # Connect signals
        self.clear_loop_button.clicked.connect(self.clear_selection)
        self.load_native_button.clicked.connect(self.load_native)
        self.show_video_checkbox.toggled.connect(self.toggle_video_visibility)
        self.play_button.clicked.connect(self.toggle_native_playback)
        self.speed_slider.valueChanged.connect(self.update_speed_label)
        self.record_button.clicked.connect(self.toggle_recording)
        self.play_user_button.clicked.connect(self.play_user_audio)
        
        # Initialize recording thread
        self.recording_thread = None
        
        # Set window size
        self.resize(1200, 800)
        
    def setup_plots(self):
        #Initialize the matplotlib plots
        self.ax_native.set_title("Native Speaker (Smoothed Pitch)")
        self.ax_user.set_title("Your Recording (Smoothed Pitch)")
        self.ax_user.set_xlabel("Time (s)")
        
        for ax in (self.ax_native, self.ax_user):
            ax.set_ylabel("Hz")
            ax.set_ylim(0, 500)
            ax.grid(True)
        
        self.canvas.draw()

    def extract_smoothed_pitch(self, path, f0_min=75, f0_max=500):
        try:
            snd = parselmouth.Sound(path)
            duration = snd.get_total_duration()
            
            # Calculate minimum possible pitch for the given sampling rate
            min_allowed_pitch = snd.sampling_frequency / 4 / 16  # Praat's internal calculation
            f0_min = max(f0_min, min_allowed_pitch)  # Use the higher of our minimum or Praat's minimum
            
            # Create pitch object with standard parameters
            pitch = snd.to_pitch(
                time_step=0.01,  # Increased time step for more stable analysis
                pitch_floor=f0_min,
                pitch_ceiling=f0_max
            )
            
            pitch_values = pitch.selected_array['frequency']
            confidence = pitch.selected_array['strength']
            times = pitch.xs()

            # More stringent confidence threshold
            voiced_mask = (confidence > 0.25) & (pitch_values > 0)
            voiced_indices = np.where(voiced_mask)[0]
            
            if len(voiced_indices) < 4:
                # Return empty arrays if not enough voiced segments
                return np.array([]), np.array([]), np.array([], dtype=bool)

            n_points = max(4, int(duration * 20))
            step = max(1, len(voiced_indices) // n_points)
            selected = voiced_indices[::step]

            x_sparse = times[selected]
            y_sparse = pitch_values[selected]

            if len(y_sparse) >= 3:
                y_sparse = medfilt(y_sparse, kernel_size=3)

            if len(x_sparse) >= 4:
                f_interp = interp1d(x_sparse, y_sparse, kind='cubic', fill_value="extrapolate")
                x_dense = np.linspace(x_sparse[0], x_sparse[-1], 300)
                y_dense = f_interp(x_dense)

                if len(y_dense) >= 13:
                    y_dense = savgol_filter(y_dense, window_length=13, polyorder=2)

                voiced_dense = np.zeros_like(x_dense, dtype=bool)
                for i, x in enumerate(x_dense):
                    nearest_idx = np.abs(times - x).argmin()
                    voiced_dense[i] = voiced_mask[nearest_idx]

                return x_dense, y_dense, voiced_dense
            else:
                return x_sparse, y_sparse, np.ones_like(x_sparse, dtype=bool)
        except Exception as e:
            print(f"Error extracting smoothed pitch: {e}")
            return None, None, None

    def update_native_plot(self):
        """Update the native speaker's pitch plot"""
        x, y, voiced = self.extract_smoothed_pitch(self.native_audio_path)
        self.ax_native.clear()
        
        # Plot the continuous curve with base thickness and transparency
        self.ax_native.plot(x, y, color='blue', linewidth=1.5, alpha=0.2, label="Native")
        
        # Find the boundaries of voiced segments
        voiced_ranges = []
        start_idx = None
        for i in range(len(voiced)):
            if voiced[i] and start_idx is None:
                start_idx = i
            elif not voiced[i] and start_idx is not None:
                voiced_ranges.append((start_idx, i))
                start_idx = None
        if start_idx is not None:
            voiced_ranges.append((start_idx, len(voiced)))
            
        # Plot each voiced segment separately with increased thickness
        for start, end in voiced_ranges:
            self.ax_native.plot(x[start:end], y[start:end], color='blue', linewidth=9, solid_capstyle='round')
        
        self.ax_native.set_title("Native Speaker (Smoothed Pitch)")
        self.ax_native.set_ylabel("Hz")
        self.ax_native.set_xlim(0, x[-1] if len(x) > 0 else 0)
        self.ax_native.set_ylim(0, 500)
        self.ax_native.legend()
        self.ax_native.grid(True)
        self.canvas.draw()
        self.native_duration = x[-1] if len(x) > 0 else 0

    def toggle_recording(self):
        """Start or stop recording"""
        if self.recording or self.pending_recording:
            # Stop current recording
            self.recording = False
            self.pending_recording = False
            if self.recording_thread:
                self.recording_thread.stop()
                self.recording_thread.wait()  # Wait for thread to finish
            QTimer.singleShot(100, self.finish_recording)
        else:
            # Start new recording
            self.record_audio()

    def record_audio(self):
        """Start recording audio"""
        if self.recording or self.pending_recording:
            print("Recording or timer already active.")
            return

        self.recording = True
        duration = self.max_recording_time
        
        # Clear and setup user plot
        self.ax_user.clear()
        self.ax_user.grid(True)
        self.ax_user.set_ylim(0, 500)
        self.ax_user.set_xlim(0, duration)
        self.ax_user.set_title("Your Recording (Real-time Pitch)")
        self.ax_user.set_ylabel("Hz")
        self.canvas.draw()
        
        # Update UI state
        self.record_button.setText("Stop Recording")
        self.play_user_button.setEnabled(False)

        # Create recording indicator overlay
        if not hasattr(self, 'overlay_frame'):
            self.overlay_frame = QFrame(self.canvas.parent())
            overlay_layout = QHBoxLayout(self.overlay_frame)
            self.recording_dot_label = QLabel('●')
            self.recording_dot_label.setStyleSheet('color: red; font-size: 35px;')
            self.countdown_label = QLabel()
            self.countdown_label.setStyleSheet('color: red; font-size: 16px;')
            overlay_layout.addWidget(self.recording_dot_label)
            overlay_layout.addWidget(self.countdown_label)
            overlay_layout.setContentsMargins(5, 0, 0, 6)

        # Position overlay
        bbox = self.ax_user.get_position()
        canvas_width = self.canvas.width()
        canvas_height = self.canvas.height()
        x_pos = int(bbox.x0 * canvas_width + 15)
        y_pos = int(bbox.y1 * canvas_height + 85)
        self.overlay_frame.move(x_pos, y_pos)
        self.overlay_frame.show()
        self.overlay_frame.raise_()

        # Start recording thread
        try:
            if not self.input_devices:
                raise Exception("No input devices available")
            
            input_index = self.input_devices[self.input_selector.currentIndex()]['index']
            self.recording_thread = RecordingThread(input_index, duration)
            self.recording_thread.update_signal.connect(self.update_recording_display)
            self.recording_thread.finished_signal.connect(self.save_recording)
            self.recording_thread.error_signal.connect(self.handle_recording_error)
            
            # Start countdown timer
            self.start_time = time.time()
            self.update_countdown()
            
            # Start recording
            self.recording_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Recording Error", f"Error starting recording: {str(e)}")
            self.recording = False
            self.finish_recording()

    def update_countdown(self):
        """Update the countdown display"""
        if self.recording:
            remaining = max(0, int(self.max_recording_time - (time.time() - self.start_time)))
            self.countdown_label.setText(f'{remaining}s')
            if remaining > 0 and self.recording:
                QTimer.singleShot(100, self.update_countdown)
            elif remaining <= 0:
                self.recording = False
                self.recording_thread.stop()

    def update_recording_display(self, frames):
        """Update the plot with current recording data"""
        if not frames:
            return

        try:
            # Convert frames to audio data
            all_audio = np.concatenate(frames, axis=0).squeeze()
            temp_file = os.path.join(tempfile.gettempdir(), "temp_chunk.wav")
            wavfile.write(temp_file, self.recording_thread.fs, all_audio)

            # Extract pitch
            x, y, voiced = self.extract_smoothed_pitch(temp_file)
            
            # Update plot
            self.ax_user.clear()
            self.ax_user.grid(True)
            self.ax_user.set_title("Your Recording (Real-time Pitch)")
            self.ax_user.set_ylabel("Hz")
            self.ax_user.set_ylim(0, 500)
            self.ax_user.set_xlim(0, self.max_recording_time)

            if len(x) > 0 and len(y) > 0:  # Only plot if we have valid data
                # Plot the continuous curve
                self.ax_user.plot(x, y, color='orange', linewidth=1.5, alpha=0.2)
                
                # Plot voiced segments
                voiced_ranges = []
                start_idx = None
                for i in range(len(voiced)):
                    if voiced[i] and start_idx is None:
                        start_idx = i
                    elif not voiced[i] and start_idx is not None:
                        voiced_ranges.append((start_idx, i))
                        start_idx = None
                if start_idx is not None:
                    voiced_ranges.append((start_idx, len(voiced)))
                
                for start, end in voiced_ranges:
                    self.ax_user.plot(x[start:end], y[start:end], color='orange', 
                                    linewidth=9, solid_capstyle='round')

            self.canvas.draw_idle()

        except Exception as e:
            print(f"Error updating recording display: {e}")

    def save_recording(self):
        """Save the recorded audio"""
        if self.recording_thread and self.recording_thread.frames:
            try:
                audio = np.concatenate(self.recording_thread.frames, axis=0).squeeze()
                wavfile.write(self.user_audio_path, self.recording_thread.fs, audio)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save recording: {str(e)}")

    def handle_recording_error(self, error_msg):
        """Handle recording errors"""
        QMessageBox.critical(self, "Recording Error", error_msg)
        self.recording = False
        self.finish_recording()

    def finish_recording(self):
        """Clean up after recording ends"""
        try:
            self.record_button.setText("Record")
            self.record_button.setEnabled(True)

            # Hide the overlay frame
            if hasattr(self, 'overlay_frame'):
                self.overlay_frame.hide()

            # Update the plot
            if os.path.exists(self.user_audio_path):
                self.play_user_button.setEnabled(True)
                self.update_user_plot()
            else:
                self.ax_user.clear()
                self.ax_user.grid(True)
                self.ax_user.set_title("Your Recording (Smoothed Pitch)")
                self.ax_user.set_ylabel("Hz")
                self.ax_user.set_ylim(0, 500)
                self.canvas.draw_idle()

        except Exception as e:
            print(f"Error in finish_recording cleanup: {e}")

    def toggle_native_playback(self):
        """Start or stop native audio playback"""
        if not self.native_audio_path:
            return

        if self.playing:
            self.playing = False
            if hasattr(self, 'playback_thread'):
                self.playback_thread.stop()
            self.play_button.setText("Play")
            self.speed_slider.setEnabled(True)
        else:
            self.playing = True
            self.play_button.setText("Stop")
            self.speed_slider.setEnabled(False)
            self.start_native_playback()

    def start_native_playback(self):
        """Start playing native audio"""
        try:
            output_index = self.output_devices[self.output_selector.currentIndex()]['index']
            speed = self.speed_slider.value() / 100.0
            
            # Create and start playback thread
            self.playback_thread = PlaybackThread(
                self.native_audio_path,
                output_index,
                start_time=self._loop_start if hasattr(self, '_loop_start') else 0,
                end_time=self._loop_end if hasattr(self, '_loop_end') else None,
                speed=speed
            )
            
            self.playback_thread.position_signal.connect(self.update_native_playback_overlay)
            self.playback_thread.finished_signal.connect(self.on_native_playback_finished)
            self.playback_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Playback Error", str(e))
            self.playing = False
            self.play_button.setText("Play")
            self.speed_slider.setEnabled(True)

    def update_native_playback_overlay(self, current_time):
        """Update the playback position overlay on the native plot"""
        try:
            # Remove previous overlay if it exists
            if hasattr(self, 'overlay_patch') and self.overlay_patch:
                self.overlay_patch.remove()
            
            # Create new overlay
            start = self._loop_start if hasattr(self, '_loop_start') else 0
            self.overlay_patch = self.ax_native.axvspan(start, start + current_time, 
                                                       color='gray', alpha=0.2)
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"Overlay update error: {e}")

    def on_native_playback_finished(self):
        """Handle completion of native audio playback"""
        if self.playing:
            # Get loop delay
            delay = self.delay_spinbox.value() / 1000.0  # Convert to seconds
            
            if delay > 0:
                QTimer.singleShot(int(delay * 1000), self.start_native_playback)
            else:
                self.start_native_playback()

    def play_user_audio(self):
        """Play the user's recording"""
        if not os.path.exists(self.user_audio_path):
            QMessageBox.warning(self, "Error", "No user recording found.")
            return

        if self.user_playing:
            self.user_playing = False
            if hasattr(self, 'user_playback_thread'):
                self.user_playback_thread.stop()
        else:
            self.user_playing = True
            try:
                output_index = self.output_devices[self.output_selector.currentIndex()]['index']
                
                # Create and start playback thread
                self.user_playback_thread = PlaybackThread(
                    self.user_audio_path,
                    output_index
                )
                
                self.user_playback_thread.position_signal.connect(self.update_user_playback_overlay)
                self.user_playback_thread.finished_signal.connect(self.on_user_playback_finished)
                self.user_playback_thread.start()

            except Exception as e:
                QMessageBox.critical(self, "Playback Error", str(e))
                self.user_playing = False

    def update_user_playback_overlay(self, current_time):
        """Update the playback position overlay on the user plot"""
        try:
            if hasattr(self, 'user_overlay_patch') and self.user_overlay_patch:
                self.user_overlay_patch.remove()
            
            self.user_overlay_patch = self.ax_user.axvspan(0, current_time, 
                                                          color='gray', alpha=0.2)
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"User overlay update error: {e}")

    def on_user_playback_finished(self):
        """Handle completion of user audio playback"""
        self.user_playing = False
        if hasattr(self, 'user_overlay_patch') and self.user_overlay_patch:
            self.user_overlay_patch.remove()
            self.canvas.draw_idle()

    def update_speed_label(self):
        """Update the label showing current playback speed"""
        speed = int(self.speed_slider.value())
        self.speed_label.setText(f"{speed}%")

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events for files"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(url.toLocalFile().lower().endswith(('.wav', '.mp3', '.mp4', '.mov', '.avi')) 
                   for url in urls):
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Handle file drop events"""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.load_file(file_path)

    def load_native(self):
        """Open file dialog to load native audio"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Native Audio/Video",
            "",
            "Audio/Video Files (*.wav *.mp3 *.mp4 *.mov *.avi)"
        )
        if file_path:
            self.load_file(file_path)

    def load_file(self, file_path):
        """Common file loading logic for both drag & drop and button"""
        if not file_path:
            return

        # Clear any existing video window first
        self.clear_video_frame()

        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".mp4", ".mov", ".avi"]:
            # Handle video file
            try:
                audio = AudioFileClip(file_path)
                tmp_path = os.path.join(tempfile.gettempdir(), "native_audio.wav")
                audio.write_audiofile(tmp_path, codec='pcm_s16le')
                self.native_audio_path = tmp_path
                
                # Extract first frame
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.original_frame = frame_rgb
                    height, width = frame_rgb.shape[:2]
                    self.aspect_ratio = width / height
                    
                    # Only show video frame if checkbox is checked
                    if self.show_video_checkbox.isChecked():
                        self.display_video_frame(frame_rgb, width, height)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load video file: {str(e)}")
                return
        else:
            # Handle audio file
            self.native_audio_path = file_path
            # Clear any existing video frame data
            if hasattr(self, 'original_frame'):
                delattr(self, 'original_frame')

        self.clear_selection()
        self.update_native_plot()
        self.play_button.setEnabled(True)

    def display_video_frame(self, frame_rgb, width, height):
        """Display the video frame in a separate window"""
        try:
            # Create new window if it doesn't exist
            if not self.video_window:
                self.video_window = VideoWindow(self)
                
                # Position the window
                main_pos = self.pos()
                self.video_window.move(main_pos.x() + self.width() + 10, main_pos.y())
            
            # Store original image and update display
            self.video_window.original_image = frame_rgb
            self.video_window.update_image()
            
            # Show the window
            self.video_window.show()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display video frame: {str(e)}")

    def clear_video_frame(self):
        """Remove the video window if it exists"""
        if self.video_window:
            self.video_window.close()
            self.video_window = None

    def toggle_video_visibility(self, checked):
        """Toggle video window visibility based on checkbox state"""
        if not hasattr(self, 'original_frame'):
            return
            
        if checked:
            # Show video if we have a frame stored
            height, width = self.original_frame.shape[:2]
            self.display_video_frame(self.original_frame, width, height)
        else:
            # Hide video window
            self.clear_video_frame()

    def setup_selection(self):
        """Setup matplotlib span selector for loop selection"""
        from matplotlib.widgets import SpanSelector
        self.span = SpanSelector(
            self.ax_native,
            self.on_select_region,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='blue'),
            interactive=True
        )
        self.span_active = False
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_mouse_down)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_up)

    def on_select_region(self, xmin, xmax):
        """Handle selection of a region in the native plot"""
        if self.playing:
            return  # Block interaction during playback

        self._loop_start = max(0.0, xmin)
        self._loop_end = xmax
        
        try:
            # Update selection patch
            if hasattr(self, 'selection_patch') and self.selection_patch:
                self.selection_patch.remove()
            self.selection_patch = self.ax_native.axvspan(
                self._loop_start, 
                self._loop_end, 
                color='blue', 
                alpha=0.3
            )
            self.loop_info_label.setText(f"Loop: {self._loop_start:.2f}s - {self._loop_end:.2f}s")
            self.canvas.draw_idle()
        except Exception as e:
            print(f"Error updating selection: {e}")

    def on_mouse_down(self, event):
        """Handle mouse button press"""
        if self.playing:
            self.was_playing = True
            self.playing = False
            if hasattr(self, 'playback_thread'):
                self.playback_thread.stop()
            self.play_button.setText("Play")
            self.span_active = True
        else:
            self.was_playing = False
            self.span_active = True

    def on_mouse_up(self, event):
        """Handle mouse button release"""
        if self.span_active:
            self.span_active = False
            if event.inaxes == self.ax_native:
                if hasattr(self, '_loop_end') and (event.xdata < self._loop_start or event.xdata > self._loop_end):
                    self.clear_selection()

    def clear_selection(self):
        """Clear the current loop selection"""
        # Stop any ongoing playback
        was_playing = self.playing
        if was_playing:
            self.playing = False
            if hasattr(self, 'playback_thread'):
                self.playback_thread.stop()
            self.play_button.setText("Play")
            QThread.msleep(100)

        # Clear selection
        self._loop_start = 0.0
        self._loop_end = None
        
        try:
            # Clear selection patch
            if hasattr(self, 'selection_patch') and self.selection_patch:
                self.selection_patch.remove()
                self.selection_patch = None
            
            if hasattr(self, 'span'):
                self.span.clear()
            
            self.loop_info_label.setText("Loop: Full clip")
            self.canvas.draw_idle()
        except Exception as e:
            print(f"Error clearing selection: {e}")

        # Restart playback if it was playing
        if was_playing:
            QTimer.singleShot(200, self.toggle_native_playback)

    def closeEvent(self, event):
        """Handle application closure"""
        try:
            # Stop any ongoing playback
            self.playing = False
            self.user_playing = False
            sd.stop()
            
            # Stop any ongoing recording
            self.recording = False
            self.pending_recording = False
            if hasattr(self, 'recording_thread') and self.recording_thread:
                self.recording_thread.stop()
                self.recording_thread.wait()
            
            # Clear video window
            self.clear_video_frame()
            
            # Clean up matplotlib
            plt.close('all')
            
            # Accept the close event
            event.accept()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            event.accept()

    def get_loop_delay(self):
        """Get the current loop delay in seconds"""
        return self.delay_spinbox.value() / 1000.0

    def update_user_plot(self):
        """Update the user's pitch plot"""
        x, y, voiced = self.extract_smoothed_pitch(self.user_audio_path)
        self.ax_user.clear()
        
        # Plot the continuous curve with base thickness and transparency
        self.ax_user.plot(x, y, color='orange', linewidth=1.5, alpha=0.2, label="User")
        
        # Find and plot voiced segments
        voiced_ranges = []
        start_idx = None
        for i in range(len(voiced)):
            if voiced[i] and start_idx is None:
                start_idx = i
            elif not voiced[i] and start_idx is not None:
                voiced_ranges.append((start_idx, i))
                start_idx = None
        if start_idx is not None:
            voiced_ranges.append((start_idx, len(voiced)))
            
        for start, end in voiced_ranges:
            self.ax_user.plot(x[start:end], y[start:end], color='orange', 
                            linewidth=9, solid_capstyle='round')
        
        self.ax_user.set_title("Your Recording (Smoothed Pitch)")
        self.ax_user.set_ylabel("Hz")
        self.ax_user.set_xlabel("Time (s)")
        self.ax_user.set_ylim(0, 500)
        self.ax_user.legend()
        self.ax_user.grid(True)
        self.canvas.draw()

class VideoWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Screenshot")
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create image label
        self.image_label = QLabel()
        layout.addWidget(self.image_label)
        
        # Create dimension label
        self.dim_label = QLabel()
        self.dim_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.dim_label)
        
        # Store rotation state
        self.current_rotation = 0
        
        # Setup key event handling
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, event):
        """Handle keyboard events for rotation"""
        if event.key() == Qt.Key.Key_Left:
            self.rotate(-90)
        elif event.key() == Qt.Key.Key_Right:
            self.rotate(90)
        else:
            super().keyPressEvent(event)

    def rotate(self, angle):
        """Rotate the image by the specified angle"""
        self.current_rotation = (self.current_rotation + angle) % 360
        self.update_image()

    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        self.update_image()

    def update_image(self):
        """Update the displayed image based on current rotation and size"""
        if not hasattr(self, 'original_image'):
            return

        # Get available size
        available_width = self.image_label.width()
        available_height = self.image_label.height()

        # Rotate image if needed
        if self.current_rotation == 90:
            rotated = cv2.rotate(self.original_image, cv2.ROTATE_90_CLOCKWISE)
        elif self.current_rotation == 180:
            rotated = cv2.rotate(self.original_image, cv2.ROTATE_180)
        elif self.current_rotation == 270:
            rotated = cv2.rotate(self.original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotated = self.original_image.copy()

        # Calculate new dimensions maintaining aspect ratio
        height, width = rotated.shape[:2]
        aspect = width / height

        if available_width / available_height > aspect:
            new_height = available_height
            new_width = int(new_height * aspect)
        else:
            new_width = available_width
            new_height = int(new_width / aspect)

        # Resize image
        resized = cv2.resize(rotated, (new_width, new_height))
        
        # Convert to QImage and display
        height, width = resized.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(resized.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

        # Update dimension label
        orig_height, orig_width = self.original_image.shape[:2]
        self.dim_label.setText(f"Original dimensions: {orig_width}x{orig_height}")

if __name__ == '__main__':
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    app = QApplication(sys.argv)
    
    # Set process DPI awareness (Windows)
    if sys.platform == 'win32':
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(2)
    
    # Create and show the main window
    window = PitchAccentApp()
    window.show()
    
    sys.exit(app.exec())