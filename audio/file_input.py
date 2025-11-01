
from pydub import AudioSegment
import librosa
import numpy as np
import os

def opus_to_wav(file, opus_path):
    
    # Load the .opus file
    audio = AudioSegment.from_file(opus_path, format="ogg")  # try "ogg" instead of "opus"
    output_dir = opus_path.split(".")[0] + ".wav"
    # Export as .wav
    audio.export(output_dir, format="wav")


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