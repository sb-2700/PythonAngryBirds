
from pydub import AudioSegment
import librosa
import numpy as np
import os
'''
# Set FFmpeg paths directly
ffmpeg_path = "C:\\ffmpeg\\bin\\ffmpeg.exe"
ffprobe_path = "C:\\ffmpeg\\bin\\ffprobe.exe"

# Print debug information
print(f"Checking FFmpeg at {ffmpeg_path}")
print(f"Checking FFprobe at {ffprobe_path}")

if os.path.exists(ffmpeg_path) and os.path.exists(ffprobe_path):
    print("Found FFmpeg and FFprobe!")
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffmpeg = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path
else:
    print("ERROR: FFmpeg/FFprobe not found!")
    if not os.path.exists(ffmpeg_path):
        print(f"Missing: {ffmpeg_path}")
    if not os.path.exists(ffprobe_path):
        print(f"Missing: {ffprobe_path}")
    print("Please ensure FFmpeg is installed correctly")'''

def opus_to_wav(file, opus_path):
    '''print("\n=== Debug Information ===")
    print(f"Input file: {file}")
    print(f"Full path: {opus_path}")
    print(f"FFmpeg converter path: {AudioSegment.converter}")
    print(f"FFmpeg path: {AudioSegment.ffmpeg}")
    print(f"FFprobe path: {AudioSegment.ffprobe}")
    print(f"File exists? {os.path.exists(opus_path)}")'''
    
    try:
        print("Attempting to load audio file...")
        # Load the .opus file
        audio = AudioSegment.from_file(opus_path, format="ogg")  # try "ogg" instead of "opus"
        print("Successfully loaded audio file!")
        
        output_dir = opus_path.split(".")[0] + ".wav"
        print(f"Will save to: {output_dir}")
        
        # Export as .wav
        print("Attempting to export as WAV...")
        audio.export(output_dir, format="wav")
        print("Successfully exported to WAV!")
        return output_dir
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print(f"Error type: {type(e)}")
        raise


    '''if file.endswith(".opus"):
        sound = AudioSegment.from_file(file, format="opus")
        wav_path = file.replace(".opus", ".wav")
        sound.export(wav_path, format="wav")
        return wav_path
    return file
'''
def extract_features(file):
    y, sr = librosa.load(file)
    # Basic audio descriptors

    # MFCCs - Mel-Frequency Cepstral Coefficients
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    # Other spectral features
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))

    # Combine into single feature vector
    features_vector = np.hstack([mfccs_mean, mfccs_std, centroid, bandwidth, zcr, rms])
    return features_vector