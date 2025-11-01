#Recieve twilio audio file 

#pass .opus file to audio_main.py to convert to .wav
#call proto_infer.py to get bird type

#Recieve second twilio audio file
#Calculate shot parameters based on simple audio features (loudness, duration, pitch)
#Power - dB of audio
#Angle - pitch of audio

#use bird_type.py to update level_1.json (bird type change)
#use shot_calc.py to calculate shot parameters and update level_1.json

#use main.py to run the game with updated level_1.json