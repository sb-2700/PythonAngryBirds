# proto_train.py
import os, json, numpy as np
from embeddings import wav_to_embedding

CLASSES = ["red","blue","yellow","black","white"]

def build_prototypes(data_dir="data"):
    protos = {}
    for cls in CLASSES:
        folder = os.path.join(data_dir, cls)
        paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".wav")]
        assert paths, f"No wavs found for class '{cls}' in {folder}"
        E = np.stack([wav_to_embedding(p) for p in paths])  # [N, 2048]
        c = E.mean(axis=0)
        c /= (np.linalg.norm(c) + 1e-10)
        protos[cls] = c.tolist()
    with open("prototypes.json","w") as f:
        json.dump(protos, f)
    print("Saved prototypes.json")

if __name__ == "__main__":
    build_prototypes()
