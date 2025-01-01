# Import necessary libraries
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from imagehash import phash
from PIL import Image
import json
import hashlib
from statistics import mean
from scipy.spatial.distance import cosine

# Function to generate and save spectrograms
def generate_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, duration=30)  # Load first 30 seconds of audio
    
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()

# Function to extract features from audio
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, duration=30)
    features = {
        "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "spectral_bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        "mfcc": np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1).tolist(),
        "chroma": np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1).tolist(),
        "spectral_contrast": np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1).tolist(),
    }
    return features

# Function to hash spectrogram image
def hash_spectrogram(image_path):
    return str(phash(Image.open(image_path)))

# Function to compute a perceptual hash of features
def hash_features(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr)
    # Extract key features for hashing
    hash_dict = {}
    
    # 1. MFCC-based hash
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfccs.mean(axis=1)
    mfcc_string = ''.join([str(int(abs(x) * 1000)) for x in mfcc_mean])
    hash_dict['mfcc_hash'] = hashlib.sha256(mfcc_string.encode()).hexdigest()
    
    # 2. Chroma-based hash
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_string = ''.join([str(int(x * 1000)) for x in chroma_mean])
    hash_dict['chroma_hash'] = hashlib.sha256(chroma_string.encode()).hexdigest()
    
    # 3. Energy-based hash
    rmse = librosa.feature.rms(y=y)[0]
    energy_string = ''.join([str(int(x * 1000)) for x in rmse[:100]])  # Use first 100 frames
    hash_dict['energy_hash'] = hashlib.sha256(energy_string.encode()).hexdigest()
    
    # 4. Compact hash (32-bit) combining multiple features
    compact_features = [
        int(mean(mfcc_mean) * 1000),
        int(mean(chroma_mean) * 1000),
        int(mean(rmse) * 1000),
        int(librosa.feature.zero_crossing_rate(y).mean() * 1000)
    ]
    compact_string = ''.join([str(x % 256) for x in compact_features])
    hash_dict['compact_hash'] = format(int(hashlib.md5(compact_string.encode()).hexdigest(), 16) % (2**32), '08x')
    
    return hash_dict

# Function to calculate similarity between hashes
def calculate_similarity(hash1, hash2):
    hash1_array = np.frombuffer(bytes.fromhex(hash1), dtype=np.uint8)
    hash2_array = np.frombuffer(bytes.fromhex(hash2), dtype=np.uint8)
    return 1 - cosine(hash1_array, hash2_array)

# Function to search for similar songs
def search_similar_songs(query_hash, hash_database, top_n=1):
    similarities = []
    for entry in hash_database:
        similarity = calculate_similarity(query_hash, entry["hash"])
        similarities.append((similarity, entry["song_name"]))

    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[:top_n]

# # Main processing script
# def process_songs(input_folder, output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     spectrogram_folder = os.path.join(output_folder, "spectrograms")
#     os.makedirs(spectrogram_folder, exist_ok=True)
    
#     all_features = []
#     all_hashes = []
#     feature_hashes = []

#     for file_name in os.listdir(input_folder):
#         if file_name.endswith(".wav") or file_name.endswith(".mp3"):
#             audio_path = os.path.join(input_folder, file_name)
            
#             # Generate spectrogram
#             spectrogram_path = os.path.join(spectrogram_folder, file_name.rsplit(".", 1)[0] + ".png")
#             generate_spectrogram(audio_path, spectrogram_path)
            
#             # Extract features
#             features = extract_features(audio_path)
#             song_name = file_name.split("_", 1)[-1].rsplit(".", 1)[0]  # Remove team number and file extension
            
#             all_features.append({"song_name": song_name, "features": features})
            
#             # Hash spectrogram
#             spectrogram_hash = hash_spectrogram(spectrogram_path)
#             all_hashes.append({"song_name": song_name, "hash": spectrogram_hash})

#             # Hash features
#             feature_hash = hash_features(features)
#             feature_hashes.append({"song_name": song_name, "hash": feature_hash})
#             print(f"Processed {file_name}: Spectrogram Hash = {spectrogram_hash}, Feature Hash = {feature_hash}")
    
#     # Save all features to a single JSON file
#     all_features_path = os.path.join(output_folder, "all_features.json")
#     with open(all_features_path, "w") as f:
#         json.dump(all_features, f, indent=4)
    
#     # Save all spectrogram hashes to a single JSON file
#     all_hashes_path = os.path.join(output_folder, "all_hashes.json")
#     with open(all_hashes_path, "w") as f:
#         json.dump(all_hashes, f, indent=4)

#     # Save all feature hashes to a single JSON file
#     feature_hashes_path = os.path.join(output_folder, "feature_hashes.json")
#     with open(feature_hashes_path, "w") as f:
#         json.dump(feature_hashes, f, indent=4)

# # Define input and output folders
# input_folder = "Music"  
# output_folder = "output"

# process_songs(input_folder, output_folder)
