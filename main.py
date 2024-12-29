import sys
import os
import librosa
import numpy as np
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QSlider,
    QFileDialog, QTableWidget, QTableWidgetItem, QWidget, QHBoxLayout, QLineEdit
)
from PyQt5.QtCore import Qt
from audioProcessor import extract_features, hash_features, search_similar_songs
import soundfile as sf  # Add this import at the top of your script


class AudioSimilarityApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Similarity Checker")
        self.setGeometry(100, 100, 800, 600)

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # File selectors
        self.file1_label = QLabel("File 1: Not Selected")
        self.file2_label = QLabel("File 2: Not Selected")
        self.layout.addWidget(self.file1_label)
        self.layout.addWidget(self.file2_label)

        self.select_file1_btn = QPushButton("Select File 1")
        self.select_file1_btn.clicked.connect(self.select_file1)
        self.layout.addWidget(self.select_file1_btn)

        self.select_file2_btn = QPushButton("Select File 2")
        self.select_file2_btn.clicked.connect(self.select_file2)
        self.layout.addWidget(self.select_file2_btn)

        # Slider for blending
        self.slider_label = QLabel("Blending Percentage: 50%")
        self.layout.addWidget(self.slider_label)

        self.blend_slider = QSlider(Qt.Horizontal)
        self.blend_slider.setMinimum(0)
        self.blend_slider.setMaximum(100)
        self.blend_slider.setValue(50)
        self.blend_slider.valueChanged.connect(self.update_slider_label)
        self.layout.addWidget(self.blend_slider)

        # Search button
        self.search_btn = QPushButton("Search Similar Songs")
        self.search_btn.clicked.connect(self.search_similar_songs)
        self.layout.addWidget(self.search_btn)

        # Table for results
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Song Name", "Similarity Index"])
        self.layout.addWidget(self.results_table)

        # File paths
        self.file1_path = None
        self.file2_path = None

    def select_file1(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File 1", "", "Audio Files (*.wav *.mp3)")
        if file_path:
            self.file1_path = file_path
            self.file1_label.setText(f"File 1: {os.path.basename(file_path)}")

    def select_file2(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File 2", "", "Audio Files (*.wav *.mp3)")
        if file_path:
            self.file2_path = file_path
            self.file2_label.setText(f"File 2: {os.path.basename(file_path)}")

    def update_slider_label(self):
        value = self.blend_slider.value()
        self.slider_label.setText(f"Blending Percentage: {value}%")
    def blend_files(self):
        pass
        # if not self.file1_path or not self.file2_path:
        #     return None

        # y1, sr1 = librosa.load(self.file1_path, duration=30)
        # y2, sr2 = librosa.load(self.file2_path, duration=30)

        # if sr1 != sr2:
        #     raise ValueError("Sampling rates of the two files do not match.")

        # weight = self.blend_slider.value() / 100
        # blended = weight * y1 + (1 - weight) * y2

        # # Save the blended audio using soundfile
        # sf.write("blended_audio.wav", blended, sr1)
        # return "blended_audio.wav"

    def search_similar_songs(self):
        pass
        
        # if not self.file1_path:
        #     return

        # # Blend files if both are selected
        # if self.file2_path:
        #     blended, sr = self.blend_files()
        #     librosa.output.write_wav("blended_audio.wav", blended, sr)
        #     query_path = "blended_audio.wav"
        # else:
        #     query_path = self.file1_path

        # # Extract features and search for similar songs
        # query_features = extract_features(query_path)
        # query_hash = hash_features(query_features)

        # # Load hash database
        # with open("output/feature_hashes.json", "r") as f:
        #     hash_database = json.load(f)

        # results = search_similar_songs(query_hash, hash_database, top_n=5)

        # # Display results
        # self.results_table.setRowCount(len(results))
        # for i, (similarity, song_name) in enumerate(results):
        #     self.results_table.setItem(i, 0, QTableWidgetItem(song_name))
        #     self.results_table.setItem(i, 1, QTableWidgetItem(f"{similarity:.4f}"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioSimilarityApp()
    window.show()
    sys.exit(app.exec_())
