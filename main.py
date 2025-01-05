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
        self.setWindowTitle("Fingerprint")
        self.setWindowIcon(QIcon("ico/logo.png"))

        # Made taller to accommodate new controls
        self.setGeometry(100, 100, 1000, 800)

        # Audio state variables
        self.file1_audio = None
        self.file2_audio = None
        self.sample_rate = None
        self.is_playing = False
        self.audio_output = None
        self.audio_output_mixed = None

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
        self.file1_label = QLabel("First Track: Not Selected")
        self.select_file1_btn = QPushButton("Select First Track")
        self.select_file1_btn.clicked.connect(self.select_file1)
        self.play_file1_btn = QPushButton("")
        self.play_file1_btn.setObjectName("play_btn")
        self.play_file1_btn.clicked.connect(
            lambda: self.toggle_playback(self.play_file1_btn, 'track1'))
        self.play_file1_btn.setProperty("is_playing", False)

        file1_layout.addWidget(self.file1_label)
        file1_layout.addWidget(self.select_file1_btn)
        file1_layout.addWidget(self.play_file1_btn)
        self.top_layout.addLayout(file1_layout)

        # File 2 selection
        file2_layout = QHBoxLayout()
        self.file2_label = QLabel("Second Track: Not Selected")
        self.select_file2_btn = QPushButton("Select Second Track")
        self.select_file2_btn.clicked.connect(self.select_file2)

        self.play_file2_btn = QPushButton("")
        self.play_file2_btn.setObjectName("play_btn")
        self.play_file2_btn.clicked.connect(
            lambda: self.toggle_playback(self.play_file2_btn, 'track2'))
        self.play_file2_btn.setProperty("is_playing", False)

        file2_layout.addWidget(self.file2_label)
        file2_layout.addWidget(self.select_file2_btn)
        file2_layout.addWidget(self.play_file2_btn)
        self.top_layout.addLayout(file2_layout)

    def setup_mixing_controls(self):
        # File 1 slider
        self.slider1_label = QLabel("First Track Weight: 50%")
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
        self.slider2_label = QLabel("Second Track Weight: 50%")
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

        self.play_btn = QPushButton(" Mix")

        # self.play_btn.clicked.connect(self.toggle_playback(self.play_btn))
        self.play_btn.clicked.connect(
            lambda: self.toggle_playback(self.play_btn, 'mixed'))
        self.play_btn.setProperty("is_playing", False)
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
        self.layout.addWidget(self.top_widget, 0, Qt.AlignCenter)

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
        self.layout.addWidget(self.results_table)
        self.layout.setAlignment(Qt.AlignCenter)

    def select_file1(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select First File", "", "Audio Files (*.wav *.mp3)")
        if file_path:
            self.file1_path = file_path
            self.file1_label.setText(f"First Track: {os.path.basename(file_path)}")
            self.file1_audio, self.sample_rate = librosa.load(
                file_path, sr=None)
            self.mix_audio()

    def select_file2(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Second File", "", "Audio Files (*.wav *.mp3)")
        if file_path:
            self.file2_path = file_path
            self.file2_label.setText(f"Second Track: {os.path.basename(file_path)}")
            self.file2_audio, _ = librosa.load(
                file_path, sr=self.sample_rate if self.sample_rate else None)
            self.mix_audio()

    def update_slider1(self):
        value = self.slider1.value()
        self.slider1_label.setText(f"First Track Weight: {value}%")
        self.slider2.setValue(100 - value)
        self.mix_audio()

    def update_slider2(self):
        value = self.slider2.value()
        self.slider2_label.setText(f"Second Track Weight: {value}%")
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
            self.audio_output_mixed = (audio1 * ratio1) + (audio2 * ratio2)

            # Save the mixed audio for similarity search
            sf.write("mixed_output.wav",
                     self.audio_output_mixed, self.sample_rate)

    def toggle_playback(self, button, track_source=None):
        is_playing = button.property("is_playing")

        if not is_playing:
            if track_source == 'track1':
                if self.file1_audio is not None:
                    self.audio_output = self.file1_audio
                    self.sample_rate = self.sample_rate # !!!!!!!!!!!!!
                else:
                    print("No track 1 selected")
                    return
            elif track_source == 'track2':
                if self.file2_audio is not None:
                    self.audio_output = self.file2_audio
                    self.sample_rate = self.sample_rate # !!!!!!!!!!!!!
                else:
                    print("No track 2 selected")
                    return
            elif track_source == 'mixed':
                self.audio_output = self.audio_output_mixed

            elif isinstance(track_source, str):  # A file path
                self.audio_output, self.sample_rate = librosa.load(
                    track_source, sr=None)
            else:
                return  # No valid audio selected

            # Start playback
            self.is_playing = True
            button.setIcon(QIcon("ico/pause.png"))
            button.setProperty("is_playing", True)
            self.start_playback()
        else:
            # Stop playback
            self.is_playing = False
            button.setIcon(QIcon("ico/play.png"))
            button.setProperty("is_playing", False)
            self.stop_playback()

    def start_playback(self):
        chunk_size = int(self.sample_rate * 0.1)  # 100ms chunks
        sd.play(self.audio_output, self.sample_rate)
        self.audio_timer.start(100)  # Update every 100ms

    def stop_playback(self):
        self.audio_timer.stop()
        sd.stop()
        self.is_playing = False
        self.play_btn.setText(" Mix")
        self.current_position = 0

    def play_audio_chunk(self):
        if not sd.get_stream().active:
            self.stop_playback()

    @staticmethod
    def get_file_type(filename):
        """Determine the type of audio file based on its name"""
        filename = filename.lower()
        if any(vocal_term in filename for vocal_term in ['vocals', 'vocal', 'lyrics']):
            return 'vocals'
        elif any(music_term in filename for music_term in ['music', 'instruments', 'instrumental']):
            return 'music'
        else:
            return 'original'
    
    @staticmethod
    def calculate_feature_similarity(features1, features2):
        """Calculate similarity between two sets of audio features"""
        similarities = {
            'mfcc': 1 - cosine(features1['mfcc'], features2['mfcc']),
            'chroma': 1 - cosine(features1['chroma'], features2['chroma']),
            'spectral_contrast': 1 - cosine(features1['spectral_contrast'], features2['spectral_contrast']),
            'spectral': 1 - abs(features1['spectral_centroid'] - features2['spectral_centroid']) / max(features1['spectral_centroid'], features2['spectral_centroid'])
        }
        
        # Weights for different feature types
        weights = {
            'mfcc': 0.4,  # MFCCs are good for timbre
            'chroma': 0.3,  # Chroma features capture harmony
            'spectral_contrast': 0.2,  # Captures tonal vs noise-like content
            'spectral': 0.1   # Basic spectral properties
        }
        
        return sum(similarities[k] * weights[k] for k in weights)
    
    @staticmethod
    def calculate_hash_similarity(hash1, hash2):
        """Calculate similarity between two perceptual hashes"""
        def calculate_hamming_similarity(hash1, hash2):
            """Calculate similarity between two hashes using Hamming distance"""
            try:
                # Convert hex strings to bytes
                hash1_bytes = bytes.fromhex(hash1)
                hash2_bytes = bytes.fromhex(hash2)
                
                # Count matching bits
                matches = sum(1 for a, b in zip(hash1_bytes, hash2_bytes) if a == b)
                return matches / len(hash1_bytes)
            except Exception as e:
                print(f"Error calculating hash similarity: {str(e)}")
                return 0
            
        similarities = {
            'mfcc': calculate_hamming_similarity(hash1['mfcc_hash'], hash2['mfcc_hash']),
            'chroma': calculate_hamming_similarity(hash1['chroma_hash'], hash2['chroma_hash']),
            'energy': calculate_hamming_similarity(hash1['energy_hash'], hash2['energy_hash']),
            'compact': calculate_hamming_similarity(hash1['compact_hash'], hash2['compact_hash'])
        }
        
        # Weights for different hash types
        weights = {
            'mfcc': 0.4,
            'chroma': 0.3,
            'energy': 0.2,
            'compact': 0.1
        }
        
        return sum(similarities[k] * weights[k] for k in weights)
    

    def search_similar_songs(self):
        """Enhanced search method with type-based filtering"""
        if not self.file1_path:
            self.results_table.setRowCount(0)
            return

        try:
            # Load databases
            with open("output/feature_hashes.json", "r") as f:
                hash_database = json.load(f)
            with open("output/all_features.json", "r") as f:
                feature_database = json.load(f)

            # Create lookup dictionary for features
            feature_lookup = {entry['song_name']: entry['features']
                              for entry in feature_database}
            hash_lookup = {entry['song_name']: entry['hash']
                              for entry in hash_database}

            # Determine search type based on input files
            file1_type = self.get_file_type(self.file1_path)
            file2_type = self.get_file_type(self.file2_path) if self.file2_path else file1_type

            # Filter database entries based on input types
            target_type = None
            if file1_type == 'vocals' and file2_type == 'vocals':
                target_type = 'vocals'
            elif file1_type == 'music' and file2_type == 'music':
                target_type = 'music'
            elif (file1_type == 'music' and file2_type == 'vocals') or (file1_type == 'vocals' and file2_type == 'music'):
                target_type = 'original'
            elif file1_type == 'original' or file2_type == 'original':
                target_type = 'original'

            # Process current audio
            if self.file2_path and self.audio_output is not None:
                temp_mix_path = "temp_mix.wav"
                sf.write(temp_mix_path, self.audio_output, self.sample_rate)
                query_path = temp_mix_path
            else:
                query_path = self.file1_path

            # Extract features and hash for query
            query_features = extract_features(query_path)
            query_hash = hash_features(query_path)

            # Calculate similarities with type filtering
            similarities = []
            for hash_entry in hash_database:
                song_name = hash_entry['song_name']
                entry_type = self.get_file_type(song_name)
                
                # Only process entries matching the target type
                if entry_type == target_type and song_name in feature_lookup:
                    hash_sim = self.calculate_hash_similarity(query_hash, hash_lookup[song_name])
                    feature_sim = self.calculate_feature_similarity(query_features, feature_lookup[song_name])
                    combined_sim = (0.7 * feature_sim) + (0.3 * hash_sim)

                    similarities.append({
                        'song_name': song_name,
                        'similarity': combined_sim
                    })

            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity'], reverse=True)

            # Update table
            self.results_table.setRowCount(len(similarities))
            self.results_table.setColumnCount(3)
            self.results_table.setHorizontalHeaderLabels(
                ["Song Name", "Similarity", ""])
            self.results_table.resizeRowsToContents()

            for i, result in enumerate(similarities):
                song_name_item = QTableWidgetItem(result['song_name'])
                similarity_item = QTableWidgetItem(
                    f"{result['similarity']:.2%}")
                self.results_table.setItem(i, 0, song_name_item)
                self.results_table.setItem(i, 1, similarity_item)

                res_play_button = QPushButton()
                res_play_button.setIcon(QIcon("ico/play.png"))
                res_play_button.setObjectName("res_play_btn")
                res_play_button.setProperty("is_playing", False)

                # correct file path for each track
                filepath = f"Music/{result['song_name']}.wav" if os.path.exists(
                    f"Music/{result['song_name']}.wav") else f"Music/{result['song_name']}.mp3"

                res_play_button.clicked.connect(
                    lambda _, btn=res_play_button, track_source=filepath: self.toggle_playback(btn, track_source=track_source))
                res_play_button.setProperty("is_playing", False)

                self.results_table.setCellWidget(i, 2, res_play_button)
            self.results_table.setSortingEnabled(True)

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
