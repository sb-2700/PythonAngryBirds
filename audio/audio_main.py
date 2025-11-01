
from file_input import opus_to_wav
from import_bird_sounds import extract_spectrogram
import os
import json

def update_bird_types(spectrogram_data, json_file_path):
    """
    Update bird types in the level json file based on audio analysis
    """
    # Simple logic to determine bird type based on spectrogram characteristics
    def determine_bird_type(spect):
        # Example logic: Check the average intensity of the spectrogram
        avg_intensity = spect.mean()
        if avg_intensity > 0.7:
            return "black_bird"  # high intensity sounds
        elif avg_intensity > 0.4:
            return "yellow_bird"  # medium intensity sounds
        else:
            return "red_bird"    # lower intensity sounds

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

print("Running main")
level_json_path = os.path.join('..', 'source', 'data', 'map', 'level_1.json')

for filename in os.listdir("sample_sounds"):
    if filename.endswith(".opus"):
        print("Target file:", filename)
        output_filename = filename.split(".")[0] + ".wav"
        opus_path = os.path.join("sample_sounds", filename)
        wav_path = os.path.join("sample_sounds", output_filename)
        
        print(f"Target file path: {opus_path}")
        sample_wav = opus_to_wav(filename, opus_path)
        print(f"Target wav path: {wav_path}")
        
        # Extract spectrogram and update birds
        sample_t, sample_f, sample_spect = extract_spectrogram(wav_path, filename.split(".")[0])
        update_bird_types(sample_spect, level_json_path)
        print(f"Updated bird types based on audio analysis")
    else:
        if filename == None:
            print("No file present")
        elif filename.endswith(".wav"):
            print("File already converted to .wav")
            wav_path = os.path.join("sample_sounds", filename)
            sample_t, sample_f, sample_spect = extract_spectrogram(wav_path, filename.split(".")[0])
            update_bird_types(sample_spect, level_json_path)
            print(f"Updated bird types based on audio analysis")

'''
sample_audio = os.path()

processed_file = ogg_to_wav("input/human_sound.opus")
features = extract_features(processed_file)
'''