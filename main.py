import sys
import os
import librosa
import numpy as np
import sounddevice as sd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QSlider,
    QFileDialog, QTableWidget, QTableWidgetItem, QWidget, QHBoxLayout, QLineEdit
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
import soundfile as sf
import json
from scipy.spatial.distance import cosine
from audioProcessor import extract_features, hash_features, search_similar_songs

class AudioSimilarityApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Similarity and Mixer")
        self.setGeometry(100, 100, 800, 800)  # Made taller to accommodate new controls

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
        self.central_widget.setLayout(self.layout)

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
        self.file1_label = QLabel("File 1: Not Selected")
        self.select_file1_btn = QPushButton("Select File 1")
        self.select_file1_btn.clicked.connect(self.select_file1)
        file1_layout.addWidget(self.file1_label)
        file1_layout.addWidget(self.select_file1_btn)
        self.layout.addLayout(file1_layout)

        # File 2 selection
        file2_layout = QHBoxLayout()
        self.file2_label = QLabel("File 2: Not Selected")
        self.select_file2_btn = QPushButton("Select File 2")
        self.select_file2_btn.clicked.connect(self.select_file2)
        file2_layout.addWidget(self.file2_label)
        file2_layout.addWidget(self.select_file2_btn)
        self.layout.addLayout(file2_layout)

    def setup_mixing_controls(self):
        # File 1 slider
        self.slider1_label = QLabel("File 1 Mix: 50%")
        self.layout.addWidget(self.slider1_label)
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(100)
        self.slider1.setValue(50)
        self.slider1.valueChanged.connect(self.update_slider1)
        self.layout.addWidget(self.slider1)

        # File 2 slider
        self.slider2_label = QLabel("File 2 Mix: 50%")
        self.layout.addWidget(self.slider2_label)
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setMinimum(0)
        self.slider2.setMaximum(100)
        self.slider2.setValue(50)
        self.slider2.valueChanged.connect(self.update_slider2)
        self.layout.addWidget(self.slider2)

    def setup_playback_controls(self):
        control_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("Play Mix")
        self.play_btn.clicked.connect(self.toggle_playback)
        control_layout.addWidget(self.play_btn)
        
        
        self.layout.addLayout(control_layout)

    def setup_search_section(self):
        # Search button
        self.search_btn = QPushButton("Search Similar Songs")
        self.search_btn.clicked.connect(self.search_similar_songs)
        self.layout.addWidget(self.search_btn)

        # Table for results
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Song Name", "Similarity Index"])
        self.layout.addWidget(self.results_table)

    def select_file1(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File 1", "", "Audio Files (*.wav *.mp3)")
        if file_path:
            self.file1_path = file_path
            self.file1_label.setText(f"File 1: {os.path.basename(file_path)}")
            self.file1_audio, self.sample_rate = librosa.load(file_path, sr=None)
            self.mix_audio()

    def select_file2(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File 2", "", "Audio Files (*.wav *.mp3)")
        if file_path:
            self.file2_path = file_path
            self.file2_label.setText(f"File 2: {os.path.basename(file_path)}")
            self.file2_audio, _ = librosa.load(file_path, sr=self.sample_rate if self.sample_rate else None)
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
        def get_file_type(filename):
            """Determine the type of audio file based on its name"""
            filename = filename.lower()
            if any(vocal_term in filename for vocal_term in ['vocals', 'vocal', 'lyrics']):
                return 'vocals'
            elif any(music_term in filename for music_term in ['music', 'instruments', 'instrumental']):
                return 'music'
            else:
                return 'original'
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

        def calculate_hash_similarity(hash1, hash2):
            """Calculate similarity between two perceptual hashes"""
            try:
                hash1_bytes = bytes.fromhex(hash1)
                hash2_bytes = bytes.fromhex(hash2)
                matches = sum(1 for a, b in zip(hash1_bytes, hash2_bytes) if a == b)
                return matches / len(hash1_bytes)
            except Exception as e:
                print(f"Error calculating hash similarity: {str(e)}")
                return 0
        """Enhanced search method with type-based filtering"""
        if not self.file1_path:
            self.results_table.setRowCount(0)
            return

        try:
            # Load databases
            with open("output/perceptual_hashes.json", "r") as f:
                hash_database = json.load(f)
            with open("output/all_features.json", "r") as f:
                feature_database = json.load(f)
                
            # Create lookup dictionary for features
            feature_lookup = {entry['song_name']: entry['features'] for entry in feature_database}
            
            # Determine search type based on input files
            file1_type = get_file_type(self.file1_path)
            file2_type = get_file_type(self.file2_path) if self.file2_path else file1_type

            # Filter database entries based on input types
            target_type = None
            if file1_type == 'vocals' and file2_type == 'vocals':
                target_type = 'vocals'
            elif file1_type == 'music' and file2_type == 'music':
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
            query_hash = hash_features(query_features)

            # Calculate similarities with type filtering
            similarities = []
            for hash_entry in hash_database:
                song_name = hash_entry['song_name']
                entry_type = get_file_type(song_name)
                
                # Only process entries matching the target type
                if entry_type == target_type and song_name in feature_lookup:
                    hash_sim = calculate_hash_similarity(query_hash, hash_entry['perceptual_audio_hash'])
                    feature_sim = calculate_feature_similarity(query_features, feature_lookup[song_name])
                    combined_sim = (0.6 * feature_sim) + (0.4 * hash_sim)
                    
                    similarities.append({
                        'song_name': song_name,
                        'similarity': combined_sim
                    })

            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity'], reverse=True)

            # Update table
            self.results_table.setRowCount(len(similarities))
            self.results_table.setColumnCount(2)
            self.results_table.setHorizontalHeaderLabels(["Song Name", "Similarity"])

            for i, result in enumerate(similarities):
                song_name_item = QTableWidgetItem(result['song_name'])
                similarity_item = QTableWidgetItem(f"{result['similarity']:.2%}")
                self.results_table.setItem(i, 0, song_name_item)
                self.results_table.setItem(i, 1, similarity_item)

            # Enable sorting and resize columns
            self.results_table.setSortingEnabled(True)
            self.results_table.resizeColumnsToContents()

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
