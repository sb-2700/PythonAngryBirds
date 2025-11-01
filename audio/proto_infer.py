# proto_infer.py
import json, sys, numpy as np
from embeddings import wav_to_embedding

with open("prototypes.json","r") as f:
    PROTOS = {k: np.array(v, dtype=np.float32) for k,v in json.load(f).items()}
CLASSES = list(PROTOS.keys())

def classify_prototype(wav_path, temperature=0.05):
    v = wav_to_embedding(wav_path)
    sims = np.array([v @ PROTOS[c] for c in CLASSES])     # cosine (L2-normed vectors)
    z = sims / temperature                                # sharpen confidences
    p = np.exp(z - z.max()); p /= p.sum()
    i = int(p.argmax())
    return {"bird": CLASSES[i], "confidence": float(p[i]), "probs": p.tolist()}

if __name__ == "__main__":
    print(classify_prototype(sys.argv[1]))
