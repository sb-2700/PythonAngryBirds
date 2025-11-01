
from file_input import opus_to_wav, extract_features
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
import os
from pydub import AudioSegment
import matplotlib.pyplot as plt




def extract_spectrogram(wav_path, filename):
    # Load audio file
    sample_rate, data = wavfile.read(wav_path)

    # Convert stereo to mono if needed
    if len(data.shape) == 2:
        data = data.mean(axis=1)

    # Compute spectrogram
    f, t, Sxx = spectrogram(data, fs=sample_rate)

    # Convert to log scale (dB)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)  # avoid log(0)

    '''#print("try plotting")
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, Sxx_log, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram (Log Scale)')
    plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    plt.show()
    #plt.savefig(f"Spectrogram_{filename}.png")
    print("Printed")
'''

    return t, f, Sxx_log

def main():
    #Folder definition
    bird_sounds_folder = "Bird_audios"
    output_dir = "Bird_audios_wav"


    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)


    for filename in os.listdir(bird_sounds_folder):
        if filename.endswith(".mp3"):
            output_filename = filename.split(".")[0] + ".wav"
            mp3_path = os.path.join(bird_sounds_folder, filename)
            wav_path = os.path.join(output_dir, output_filename)
            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(wav_path, format="wav")
        print(f"WAV saved to: {wav_path}")

    for filename in os.listdir(output_dir):
        if filename.endswith(".wav"):
            wav_path = os.path.join(output_dir, filename)
            print(wav_path)
            bird_name = filename.split(".")[0]
            spect_t, spect_f, Spect_Sxx = extract_spectrogram(wav_path, filename)

if __name__ == "__main__":
    main()