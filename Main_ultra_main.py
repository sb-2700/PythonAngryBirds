import os
import sys
from source.main import main

# Compute project root and ensure audio folder is on sys.path so we can import audio helpers
project_root = os.path.abspath(os.path.dirname(__file__))
audio_dir = os.path.join(project_root, 'audio')
if audio_dir not in sys.path:
	sys.path.insert(0, audio_dir)

import audio_setup 
from file_input import opus_to_wav

# Original plan / notes (keep all comments):
#Recieve twilio audio file
#
#pass .opus file to audio_main.py to convert to .wav
#call proto_infer.py to get bird type
#use bird_type.py to update level_1.json (bird type change)
#
#Recieve second twilio audio file

# Convert a sample opus file in audio/power_sounds to WAV and save next to it
power_folder = os.path.join(audio_dir, 'power_sounds')
opus_name = 'power_sound_2.opus'
opus_path = os.path.join(power_folder, opus_name)
if os.path.exists(opus_path):
	try:
		wav_path = opus_to_wav(opus_name, opus_path)
		print('Converted to WAV:', wav_path)
	except Exception as e:
		print('Failed to convert opus to wav:', e)
else:
	print('Opus file not found at', opus_path)

#Run main.py which will calculate the correct shot parameters as game loads.
#use main.py to run the game with updated level_1.json
main()

# Next steps (outline):
# - Call audio_main / proto_infer to process wav and get bird type
# - Update level_1.json or send live updates to game
# - Use shot_calc to calculate shot params from a second audio file
# - Launch game via main.py or communicate updates to running instance