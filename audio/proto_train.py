# proto_train.py
import os, json, numpy as np
from embeddings import wav_to_embedding

CLASSES = ["red","blue","yellow","white"]

FILE_TO_CLASS = {
    "Red_bird_fly.wav": "red",
    "Blue_bird_fly.wav": "blue", 
    "Chuck_yellow_bird_fly.wav": "yellow",
    "Matilda_white_bird_fly.wav": "white"
}

def build_prototypes(data_dir="Bird_audios_wav"):
    protos = {}
    
    # Group files by class
    class_files = {cls: [] for cls in CLASSES}
    
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".wav") and filename in FILE_TO_CLASS:
            cls = FILE_TO_CLASS[filename]
            class_files[cls].append(os.path.join(data_dir, filename))
    
    for cls in CLASSES:
        paths = class_files[cls]
        if not paths:
            print(f"Warning: No files found for class '{cls}'")
            continue
            
        E = np.stack([wav_to_embedding(p) for p in paths])  # [N, 2048]
        c = E.mean(axis=0)
        c /= (np.linalg.norm(c) + 1e-10)
        protos[cls] = c.tolist()
        print(f"Created prototype for {cls} using {len(paths)} file(s)")
    
    with open("prototypes.json","w") as f:
        json.dump(protos, f)
    print("Saved prototypes.json")

if __name__ == "__main__":
    build_prototypes()
