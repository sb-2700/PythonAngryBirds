
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

def update_bird_types(spectrogram_data, json_file_path):
    """
    Update bird types in the level json file based on audio analysis
    """
    # Simple logic to determine bird type based on spectrogram characteristics
    def determine_bird_type(spect):
        # Example logic: Check the average intensity of the spectrogram
        '''print("Determining bird type from spectrogram data")
        avg_intensity = np.random.uniform(0, 1)  # Generate random number between 0 and 1
        if avg_intensity > 0.7:
            return "black_bird"  # high intensity sounds
        elif avg_intensity > 0.4:
            return "yellow_bird"  # medium intensity sounds
        else:
            return "red_bird"    # lower intensity sounds'''

    # Read the current level configuration
    with open(json_file_path, 'r') as f:
        level_data = json.load(f)

    # Update bird types based on audio analysis
    bird_type = determine_bird_type(spectrogram_data)
    for bird in level_data['birds']:
        bird['type'] = bird_type

    # Save the updated configuration
    with open(json_file_path, 'w') as f:
        json.dump(level_data, f, indent=4)

print("Running audio main")
#print("Current working directory:", os.getcwd())
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
sample_sounds_dir = os.path.join(script_dir, "sample_sounds")
# Get the correct path to level_1.json
level_json_path = os.path.join(script_dir, "..", "source", "data", "map", "level_1.json")
level_json_path = os.path.abspath(level_json_path)  # Convert to absolute path

print("Looking for files in:", sample_sounds_dir)
wav_exists = False
for filename in os.listdir(sample_sounds_dir):
    if filename.endswith(".wav"):
        wav_exists = True
        
wav_exists = False

if wav_exists:
    print("File already converted to .wav")
    wav_path = os.path.join(sample_sounds_dir, filename)
    sample_t, sample_f, sample_spect = extract_spectrogram(wav_path, filename.split(".")[0])
    update_bird_types(sample_spect, level_json_path)
    print(f"Updated bird types based on audio analysis")
else:
    for filename in os.listdir(sample_sounds_dir):
        if filename.endswith(".opus"):
            print("WAV not found, converting from OPUS")
            print("Target file:", filename)
            output_filename = filename.split(".")[0] + ".wav"
            opus_path = os.path.join(sample_sounds_dir, filename)
            wav_path = os.path.join(sample_sounds_dir, output_filename)
        
            #print(f"Target file path: {opus_path}")
            sample_wav = opus_to_wav(filename, opus_path)
            #print(f"Target wav path: {wav_path}")
        
            # Extract spectrogram and update birds
            sample_t, sample_f, sample_spect = extract_spectrogram(wav_path, filename.split(".")[0])
            update_bird_types(sample_spect, level_json_path)
            print(f"Updated bird types based on audio analysis")
        else:
            if filename == None:
                print("No file present")
    

'''
sample_audio = os.path()

processed_file = ogg_to_wav("input/human_sound.opus")
features = extract_features(processed_file)
'''