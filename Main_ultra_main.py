#main_ultra_main
import os
import sys
import shutil
import time
import json
import numpy as np
from pydub import AudioSegment
import subprocess

# Compute project root and ensure audio folder is on sys.path so we can import audio helpers
project_root = os.path.abspath(os.path.dirname(__file__))
audio_dir = os.path.join(project_root, 'audio')
if audio_dir not in sys.path:
    sys.path.insert(0, audio_dir)

from shot_calc import get_pitch
import audio.audio_setup 
from audio.file_input import opus_to_wav

print("Project root:", project_root)
print("Audio dir:", audio_dir)
print("Imported:", opus_to_wav)

file_found = False

def update_bird_types(bird_type,json_file_path):
    """
    Update bird types in the level json file based on audio analysis
    """
    
    # Read the current level configuration
    with open(json_file_path, 'r') as f:
        level_data = json.load(f)

    # Update bird types based on audio analysis
    level_data['birds'][0]['type'] = bird_type
    '''for bird in level_data['birds']:
        bird['type'] = bird_type'''

    # Save the updated configuration
    with open(json_file_path, 'w') as f:
        json.dump(level_data, f, indent=4)

start = time.time()
while not file_found:
	end = time.time()
	if (end - start > 40):  # Timeout after 20 seconds
		print("Timeout waiting for .ogg file.")
		break
	time.sleep(1)
	trilio_saved_folder = os.path.join(project_root, 'downloaded_media')
	# Find the most recent .ogg file in the folder
	try:
		ogg_files = [f for f in os.listdir(trilio_saved_folder) if f.lower().endswith('.ogg')]
		if ogg_files:
			# Get the most recent file by modification time
			ogg_files_full = [os.path.join(trilio_saved_folder, f) for f in ogg_files]
			most_recent_ogg = max(ogg_files_full, key=os.path.getmtime)
			bird_impression_name = most_recent_ogg
			print(f"Bird impression name: {bird_impression_name}")
			most_recent_wav = opus_to_wav(os.path.basename(most_recent_ogg), most_recent_ogg)
			# Copy to sample_sounds folder
			sample_sounds_folder = os.path.join(audio_dir, 'sample_sounds')
			if not os.path.exists(sample_sounds_folder):
				os.makedirs(sample_sounds_folder)
			dest_path = os.path.join(sample_sounds_folder, 'Mat_sample.wav')
			shutil.copy2(most_recent_wav, dest_path)
			print(f"Copied {most_recent_wav} to {dest_path}")
			file_found = True
	except Exception as e:
		print(f"Error searching for .ogg files: {e}")

'''
input_pitch = get_pitch(dest_path)
print(f"Detected pitch: {input_pitch} Hz for {dest_path}")
# Original plan / notes (keep all comments):
#Recieve twilio audio file
#Write file to folder from local storage on computer
#
#pass .opus file to audio_main.py to convert to .wav
#call proto_infer.py to get bird type
#use bird_type.py to update level_1.json (bird type change)
#

scaled_pitch = np.clip(np.interp(input_pitch, [450, 850], [2700, 3300]), 2700, 3300)
if scaled_pitch < 2900:
     decided_bird = "red" #red freq = 2867Hz
elif scaled_pitch > 3100:
        decided_bird = "yellow" #yellow freq = 3175Hz 
else:
     decided_bird= "blue"

insert_bird = decided_bird +"_bird"
'''

# ========== BIRD CLASSIFICATION USING proto_infer_head.py ==========
if file_found:
    try:
        print(f"Running bird classification on: {dest_path}")
        
        # Change working directory to audio folder so proto_infer_head.py can find its files
        original_cwd = os.getcwd()
        os.chdir(audio_dir)
        
        # Import proto_infer_head functions after changing to audio directory
        sys.path.insert(0, audio_dir)
        import proto_infer_head
        
        # Run the complete classification process
        result = proto_infer_head.classify(dest_path, use_head=True)
        
        # Extract the bird type (HEAD_CLASSES[i] equivalent)
        decided_bird = result["bird"]  # This is HEAD_CLASSES[i] from your proto_infer_head
        confidence = result["confidence"]
        method = result["method"]

    
        print(f"🐦 Classification complete!")
        print(f"   Method: {method}")
        print(f"   Detected bird: {decided_bird}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Full result: {result}")
        
        # Convert to format needed for JSON update
        insert_bird = decided_bird + "_bird"
        
        # Change back to original directory
        os.chdir(original_cwd)
        
    except Exception as e:
        print(f"Error running bird classification: {e}")
        print(f"Error details: {str(e)}")
        # Fallback to default
        decided_bird = "red"
        insert_bird = "red_bird"
        confidence = 0.0
        print("Using fallback bird type: red")
   # Make sure to change back to original directory even on error
        os.chdir(original_cwd)

else:
    # No file found, use fallback
    decided_bird = "red"
    insert_bird = "red_bird"
    confidence = 0.0
    print("No audio file found, using fallback bird type: red")

print(f"Final decision: {insert_bird} (confidence: {confidence:.3f})")


#Update json level_1.json with new bird type
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the correct path to level_1.json
level_json_path = os.path.join(script_dir, "source", "data", "map", "level_1.json")
level_json_path = os.path.abspath(level_json_path)  # Convert to absolute path

#Carry out JSON update
update_bird_types(insert_bird, level_json_path)

#Recieve second twilio audio file

power_file_found = False
while not power_file_found:
    end = time.time()
    if (end - start) > 40:  # Timeout after 40 seconds
        print("Timeout waiting for .ogg file.")
        break

    time.sleep(1)
    # Find the most recent .ogg file in the folder
    try:
        # Update files in case new one arrived
        ogg_files = [f for f in os.listdir(trilio_saved_folder) if f.lower().endswith('.ogg')]
        if ogg_files:
            # Get the most recent file by modification time
            ogg_files_full = [os.path.join(trilio_saved_folder, f) for f in ogg_files]
            most_recent_ogg = max(ogg_files_full, key=os.path.getmtime)
            if most_recent_ogg == bird_impression_name:
                continue
            most_recent_wav = opus_to_wav(os.path.basename(most_recent_ogg), most_recent_ogg)
            print(f"Most recent ogg power: {most_recent_ogg}")
            # Copy to sample_sounds folder
            sample_sounds_folder = os.path.join(audio_dir, 'power_sounds')
            if not os.path.exists(sample_sounds_folder):
                os.makedirs(sample_sounds_folder)
            dest_path = os.path.join(sample_sounds_folder, 'power_sound_2.wav')
            shutil.copy2(most_recent_wav, dest_path)
            print(f"Copied {most_recent_wav} to {dest_path}")
            power_file_found = True
    except Exception as e:
        print(f"Error searching for .ogg files: {e}")

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
time.sleep(2)

from source.main import main
main()
print("Main run")


try:
    subprocess.run(['python', "wav_delete.py"], check=True)
    print("wav_delete.py executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error running wav_delete.py: {e}")


# Next steps (outline):
# - Call audio_main / proto_infer to process wav and get bird type
# - Update level_1.json or send live updates to game
# - Use shot_calc to calculate shot params from a second audio file
# - Launch game via main.py or communicate updates to running instance