
import audio_setup
from file_input import opus_to_wav
from import_bird_sounds import extract_spectrogram
import os
import json
import numpy as np
from pydub import AudioSegment

'''
# Set FFmpeg paths directly in the code
ffmpeg_path = os.path.abspath(os.path.join('C:', 'ffmpeg', 'bin', 'ffmpeg.exe'))
ffprobe_path = os.path.abspath(os.path.join('C:', 'ffmpeg', 'bin', 'ffprobe.exe'))'''

'''if os.path.exists(ffmpeg_path) and os.path.exists(ffprobe_path):
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffmpeg = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path
else:
    print("Warning: FFmpeg not found at", ffmpeg_path)
    print("Please ensure FFmpeg is installed at C:\\ffmpeg\\bin\\")'''

def update_bird_types(spectrogram_data, json_file_path, bird_color):
    """
    Update bird types in the level json file based on audio analysis
    """
    # Read the current level configuration
    with open(json_file_path, 'r') as f:
        level_data = json.load(f)

    # Update bird types based on the detected bird color
    for bird in level_data['birds']:
        bird['type'] = f"{bird_color}_bird"

    # Save the updated configuration
    with open(json_file_path, 'w') as f:
        json.dump(level_data, f, indent=4)

print("Running audio main")
#print("Current working directory:", os.getcwd())
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(script_dir, "dataset")
# Get the correct path to level_1.json
level_json_path = os.path.join(script_dir, "..", "source", "data", "map", "level_1.json")
level_json_path = os.path.abspath(level_json_path)  # Convert to absolute path

print("Looking for files in:", dataset_dir)

# Define the bird color folders - WHITE BIRD DISABLED
bird_folders = ["blue", "red", "yellow"]  # Removed "white"

# Process each bird color folder
for bird_color in bird_folders:
    bird_folder_path = os.path.join(dataset_dir, bird_color)
    
    if not os.path.exists(bird_folder_path):
        print(f"Folder {bird_color} not found, skipping...")
        continue
    
    print(f"\nProcessing {bird_color} bird folder...")
    
    # Check if WAV files already exist
    wav_files = [f for f in os.listdir(bird_folder_path) if f.endswith(".wav")]
    
    if wav_files:
        print(f"Found {len(wav_files)} WAV file(s) in {bird_color} folder")
        # Process the first WAV file for bird type detection
        wav_path = os.path.join(bird_folder_path, wav_files[0])
        sample_t, sample_f, sample_spect = extract_spectrogram(wav_path, wav_files[0].split(".")[0])
        update_bird_types(sample_spect, level_json_path, bird_color)
        print(f"Updated bird types based on {bird_color} bird audio analysis")
    
    # Convert ALL OPUS files to WAV (regardless of whether WAV files exist)
    opus_files = [f for f in os.listdir(bird_folder_path) if f.endswith(".opus")]
    
    if opus_files:
        print(f"Found {len(opus_files)} OPUS file(s) to convert in {bird_color} folder")
        converted_count = 0
        
        for filename in opus_files:
            print(f"Converting: {filename}")
            output_filename = filename.split(".")[0] + ".wav"
            opus_path = os.path.join(bird_folder_path, filename)
            wav_path = os.path.join(bird_folder_path, output_filename)
            
            # Check if WAV already exists for this OPUS file
            if os.path.exists(wav_path):
                print(f"  WAV already exists for {filename}, skipping...")
                continue
            
            try:
                sample_wav = opus_to_wav(filename, opus_path)
                converted_count += 1
                print(f"  Successfully converted {filename} to {output_filename}")
            except Exception as e:
                print(f"  Error converting {filename}: {e}")
        
        print(f"Converted {converted_count} OPUS files in {bird_color} folder")
        
        # If we converted files and didn't have WAV files before, update bird types
        if not wav_files and converted_count > 0:
            # Use the first converted file for bird type detection
            first_converted = opus_files[0].split(".")[0] + ".wav"
            wav_path = os.path.join(bird_folder_path, first_converted)
            if os.path.exists(wav_path):
                sample_t, sample_f, sample_spect = extract_spectrogram(wav_path, first_converted.split(".")[0])
                update_bird_types(sample_spect, level_json_path, bird_color)
                print(f"Updated bird types based on {bird_color} bird audio analysis")
    else:
        print(f"No OPUS files found in {bird_color} folder")

print("\nFinished processing all bird folders")