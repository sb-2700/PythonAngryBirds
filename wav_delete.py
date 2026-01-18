#wav_delete.py

import os

# List of target folders
folders = [
    os.path.join('audio', 'power_sounds'),
    os.path.join('audio', 'sample_sounds'),
    'downloaded_media'
]

# Loop through each folder and delete .wav files
for folder in folders:
    folder_path = os.path.join(os.getcwd(), folder)
    print(f"Checking folder: {folder_path}")
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith('.wav') or file_name.lower().endswith('.ogg'):
                file_path = os.path.join(folder_path, file_name)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
    else:
        
        print(f"Folder not found: {folder_path}")
