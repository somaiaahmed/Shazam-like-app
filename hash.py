import librosa
import json
import hashlib
import os
from statistics import mean

def generate_audio_hash(audio_path, sr=22050):
    """
    Generate perceptual hash from an audio file.
    Returns multiple hash representations for robustness.
    """
    # Load the audio file
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

# def save_hashes_to_json(hashes, output_path):
#     """Save hashes to a JSON file."""
#     with open(output_path, 'w') as f:
#         json.dump(hashes, f, indent=4)

# def process_songs(input_folder, output_folder, sr=22050):
#     """Process all audio files in the input folder and save hashes."""
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     feature_hashes = []

#     for file_name in os.listdir(input_folder):
#         if file_name.lower().endswith(('.wav', '.mp3')):
#             audio_path = os.path.join(input_folder, file_name)
#             try:
#                 # Generate hashes
#                 feature_hash = generate_audio_hash(audio_path, sr=sr)
#                 feature_hashes.append({"song_name": file_name, "hash": feature_hash})
#                 print(f"Processed {file_name}")
#             except Exception as e:
#                 print(f"Error processing {file_name}: {e}")
    
#     # Save all feature hashes to a JSON file
#     feature_hashes_path = os.path.join(output_folder, "feature_hashes.json")
#     save_hashes_to_json(feature_hashes, feature_hashes_path)

# # Example usage
# input_folder = "Music"  
# output_folder = "output"

# process_songs(input_folder, output_folder)
