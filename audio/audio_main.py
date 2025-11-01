
from file_input import opus_to_wav
from import_bird_sounds import extract_spectrogram
import os

#Import Audio input from whatsapp web
print("Running main")
#Samples for mats audio
for filename in os.listdir("sample_sounds"):
    if filename.endswith(".opus"):
        print("Target file:",filename)
        output_filename = filename.split(".")[0] + ".wav"
        opus_path = os.path.join("sample_sounds", filename)
        wav_path = os.path.join("sample_sounds", output_filename)
        #audio = AudioSegment.from_mp3(mp3_path)
        #audio.export(wav_path, format="wav")
        print(f"Target file path: {opus_path}")
        sample_wav = opus_to_wav(filename,opus_path)
        print(f"Target wav path: {wav_path}")
        sample_t, sample_f, sample_spect = extract_spectrogram(wav_path, filename.split(".")[0])
    else:
        if filename == None:
            print("No file present")
        elif filename.endswith(".wav"):
            print("File already converted to .wav")

'''
sample_audio = os.path()

processed_file = ogg_to_wav("input/human_sound.opus")
features = extract_features(processed_file)
'''