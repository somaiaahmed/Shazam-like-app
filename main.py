import sys
import os
import librosa
import numpy as np
import sounddevice as sd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QSlider,
    QFileDialog, QTableWidget, QTableWidgetItem, QWidget, QHBoxLayout, QLineEdit, QSizePolicy, QHeaderView
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
import soundfile as sf
import json
from scipy.spatial.distance import cosine
from audioProcessor import extract_features, hash_features, search_similar_songs
from PyQt5.QtCore import QFile, QTextStream, QSize


def load_stylesheet():
    # Open the stylesheet file
    file = QFile("style.qss")
    if not file.open(QFile.ReadOnly | QFile.Text):
        print("Cannot open the stylesheet file!")
        return ""

    # Read the file content
    stream = QTextStream(file)
    stylesheet = stream.readAll()
    return stylesheet


class AudioSimilarityApp(QMainWindow):
    def __init__(self):
        super().__init__()
        stylesheet = load_stylesheet()

        # Set the stylesheet to the application
        self.setStyleSheet(stylesheet)
        self.setWindowTitle("Audio Similarity and Mixer")
        # Made taller to accommodate new controls
        self.setGeometry(100, 100, 1000, 800)

        # Audio state variables
        self.file1_audio = None
        self.file2_audio = None
        self.sample_rate = None
        self.is_playing = False
        self.audio_output = None

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        
        
        self.top_layout = QVBoxLayout()
        self.top_layout.setObjectName("top_layout")
        
        
        self.central_widget.setLayout(self.layout)
        self.layout.setAlignment(Qt.AlignCenter)
        # File selection section
        self.setup_file_selection()

        # Mixing controls section
        self.setup_mixing_controls()

        # Playback controls section
        self.setup_playback_controls()

        # Search and results section
        self.setup_search_section()

        # Initialize audio stream
        self.audio_timer = QTimer()
        self.audio_timer.timeout.connect(self.play_audio_chunk)
        self.current_position = 0

        # File paths
        self.file1_path = None
        self.file2_path = None

    def setup_file_selection(self):
        # File 1 selection
        file1_layout = QHBoxLayout()
        self.file1_label = QLabel("Track 1: Not Selected")
        self.select_file1_btn = QPushButton("Select Track 1")
        self.select_file1_btn.clicked.connect(self.select_file1)
        self.play_file1_btn = QPushButton("")
        self.play_file1_btn.setObjectName("play_btn")

        file1_layout.addWidget(self.file1_label)
        file1_layout.addWidget(self.select_file1_btn)
        file1_layout.addWidget(self.play_file1_btn)
        self.top_layout.addLayout(file1_layout)

        # File 2 selection
        file2_layout = QHBoxLayout()
        self.file2_label = QLabel("Track 2: Not Selected")
        self.select_file2_btn = QPushButton("Select Track 2")
        self.select_file2_btn.clicked.connect(self.select_file2)

        self.play_file2_btn = QPushButton("")
        self.play_file2_btn.setObjectName("play_btn")

        file2_layout.addWidget(self.file2_label)
        file2_layout.addWidget(self.select_file2_btn)
        file2_layout.addWidget(self.play_file2_btn)
        self.top_layout.addLayout(file2_layout)

    def setup_mixing_controls(self):
        # File 1 slider
        self.slider1_label = QLabel("Track 1 Mix: 50%")
        slider_layout1 = QHBoxLayout()
        slider_layout1.addWidget(self.slider1_label)

        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(100)
        self.slider1.setValue(50)
        self.slider1.valueChanged.connect(self.update_slider1)

        slider_layout1.addWidget(self.slider1)
        self.top_layout.addLayout(slider_layout1)

        # File 2 slider
        self.slider2_label = QLabel("Track 2 Mix: 50%")
        slider_layout2 = QHBoxLayout()
        slider_layout2.addWidget(self.slider2_label)
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setMinimum(0)
        self.slider2.setMaximum(100)
        self.slider2.setValue(50)
        self.slider2.valueChanged.connect(self.update_slider2)
        slider_layout2.addWidget(self.slider2)
        self.top_layout.addLayout(slider_layout2)

    def setup_playback_controls(self):
        self.control_layout = QHBoxLayout()

        self.play_btn = QPushButton("Play Mix")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.control_layout.addWidget(self.play_btn)

        

    def setup_search_section(self):
        # Search button
        self.search_btn = QPushButton("Search Similar Songs")
        self.search_btn.clicked.connect(self.search_similar_songs)
        self.control_layout.addWidget(self.search_btn)
        self.top_layout.addLayout(self.control_layout)
        
        
        self.top_widget = QWidget()
        self.top_widget.setLayout(self.top_layout)
        self.top_widget.setObjectName("top_widget")
        self.layout.addWidget(self.top_widget,0, Qt.AlignCenter)

        # Table for results
        self.results_table = QTableWidget()
        
        self.results_table.resizeColumnsToContents()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(
            ["Song Name", "Similarity Index", ""])
        self.results_table.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.results_table.horizontalHeader().setStretchLastSection(False)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.resizeRowsToContents()
        self.results_table.resizeColumnsToContents()
        self.layout.addWidget(self.results_table )
        self.layout.setAlignment(Qt.AlignCenter)
    def select_file1(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select File 1", "", "Audio Files (*.wav *.mp3)")
        if file_path:
            self.file1_path = file_path
            self.file1_label.setText(f"Track 1: {os.path.basename(file_path)}")
            self.file1_audio, self.sample_rate = librosa.load(
                file_path, sr=None)
            self.mix_audio()

    def select_file2(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select File 2", "", "Audio Files (*.wav *.mp3)")
        if file_path:
            self.file2_path = file_path
            self.file2_label.setText(f"Track 2: {os.path.basename(file_path)}")
            self.file2_audio, _ = librosa.load(
                file_path, sr=self.sample_rate if self.sample_rate else None)
            self.mix_audio()

    def update_slider1(self):
        value = self.slider1.value()
        self.slider1_label.setText(f"File 1 Mix: {value}%")
        self.slider2.setValue(100 - value)
        self.mix_audio()

    def update_slider2(self):
        value = self.slider2.value()
        self.slider2_label.setText(f"File 2 Mix: {value}%")
        self.slider1.setValue(100 - value)
        self.mix_audio()

    def mix_audio(self):
        if self.file1_audio is not None and self.file2_audio is not None:
            # Ensure both audio files are the same length
            min_length = min(len(self.file1_audio), len(self.file2_audio))
            audio1 = self.file1_audio[:min_length]
            audio2 = self.file2_audio[:min_length]

            # Calculate mix ratios
            ratio1 = self.slider1.value() / 100
            ratio2 = self.slider2.value() / 100

            # Mix the audio
            self.audio_output = (audio1 * ratio1) + (audio2 * ratio2)

            # Save the mixed audio for similarity search
            sf.write("mixed_output.wav", self.audio_output, self.sample_rate)

    def toggle_playback(self):
        if not self.is_playing and self.audio_output is not None:
            self.play_btn.setText("Stop")
            self.is_playing = True
            self.current_position = 0
            self.start_playback()
        else:
            self.play_btn.setText("Play Mix")
            self.is_playing = False
            self.stop_playback()

    def start_playback(self):
        chunk_size = int(self.sample_rate * 0.1)  # 100ms chunks
        sd.play(self.audio_output, self.sample_rate)
        self.audio_timer.start(100)  # Update every 100ms

    def stop_playback(self):
        self.audio_timer.stop()
        sd.stop()
        self.is_playing = False
        self.play_btn.setText("Play Mix")
        self.current_position = 0

    def play_audio_chunk(self):
        if not sd.get_stream().active:
            self.stop_playback()

    def search_similar_songs(self):
        def calculate_hash_similarity(hash1, hash2):
            """
            Calculate similarity between two hashes using byte-level comparison
            Returns a value between 0 and 1
            """
            try:
                # Convert hashes to byte arrays for comparison
                hash1_bytes = bytes.fromhex(hash1)
                hash2_bytes = bytes.fromhex(hash2)
                
                # Count matching bytes
                matches = sum(1 for a, b in zip(hash1_bytes, hash2_bytes) if a == b)
                
                # Calculate similarity ratio
                similarity = matches / len(hash1_bytes)
                return similarity
            except Exception as e:
                print(f"Error calculating hash similarity: {str(e)}")
                return 0

        if not self.file1_path or not os.path.exists("output/feature_hashes.json"):
            self.results_table.setRowCount(0)
            return

        try:
            # Load hash database
            with open("output/feature_hashes.json", "r") as f:
                hash_database = json.load(f)
            
            # Process the current audio mix if both files are selected
            if self.file2_path and self.audio_output is not None:
                temp_mix_path = "temp_mix.wav"
                sf.write(temp_mix_path, self.audio_output, self.sample_rate)
                query_path = temp_mix_path
            else:
                query_path = self.file1_path

            # Extract features and generate hash for query
            query_features = extract_features(query_path)
            query_hash = hash_features(query_features)

            # Identify whether the files are vocals or instruments
            file1_is_vocals = "vocals" in self.file1_path.lower() or "vocal" in self.file1_path.lower()
            file2_is_vocals = "vocals" in self.file2_path.lower() or "vocal" in self.file2_path.lower() if self.file2_path else False

            file_type = "vocals" if file1_is_vocals or file2_is_vocals else "instruments"
            
            # Filter hash database based on file type
            filtered_hashes = []
            for entry in hash_database:
                song_name = entry["song_name"]
                if file_type == "vocals" and ("vocals" in song_name.lower() or "vocal" in song_name.lower()):
                    filtered_hashes.append(entry)
                elif file_type == "instruments" and ("music" in song_name.lower() or "instrumental" in song_name.lower()):
                    filtered_hashes.append(entry)

            # Calculate similarity for each filtered entry
            similarities = []
            for entry in filtered_hashes:
                similarity = calculate_hash_similarity(query_hash, entry["hash"])
                similarities.append({
                    "song_name": entry["song_name"],
                    "similarity": similarity
                })

            # Sort results by similarity
            similarities.sort(key=lambda x: x["similarity"], reverse=True)

            # Update table with sorted results
            self.results_table.setRowCount(len(similarities))
            self.results_table.setColumnCount(3)
            self.results_table.setHorizontalHeaderLabels(["Song Name", "Similarity",""])

            for i, result in enumerate(similarities):
                song_name_item = QTableWidgetItem(result["song_name"])
                similarity_item = QTableWidgetItem(f"{result['similarity']:.2%}")
                self.results_table.setItem(i, 0, song_name_item)
                self.results_table.setItem(i, 1, similarity_item)
                play_button = QPushButton()
                play_button.setIcon(QIcon("ico/play.png"))
                play_button.setObjectName("res_play_btn")
                self.results_table.setCellWidget(i, 2, play_button)

            # Enable sorting and resize columns
            self.results_table.setSortingEnabled(True)
            # self.results_table.resizeRowsToContents()
            

            # self.results_table.resizeColumnsToContents()

            # Clean up temporary file
            if self.file2_path and os.path.exists("temp_mix.wav"):
                os.remove("temp_mix.wav")

        except Exception as e:
            print(f"Error during search: {str(e)}")
            self.results_table.setRowCount(0)
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioSimilarityApp()
    window.show()
    sys.exit(app.exec_())
