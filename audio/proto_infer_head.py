# proto_infer.py
import json, sys, numpy as np, os
from scipy.special import softmax
from embeddings import wav_to_embedding

# ---- Load prototypes (fallback path)
PROTOS, CLASSES = {}, []
if os.path.exists("prototypes.json"):
    with open("prototypes.json","r") as f:
        PROTOS = {k: np.array(v, dtype=np.float32) for k,v in json.load(f).items()}
    CLASSES = list(PROTOS.keys())
    MAT = np.stack([PROTOS[c] for c in CLASSES], axis=0) if CLASSES else None
else:
    MAT = None

# ---- Load trained head if available
HEAD_PATH = "bird_head.joblib"
HEAD = None
if os.path.exists(HEAD_PATH):
    import joblib
    B = joblib.load(HEAD_PATH)
    CLF, HEAD_CLASSES = B["clf"], B["classes"]
    HEAD = (CLF, HEAD_CLASSES)

def classify_prototype(wav_path, temperature=0.10):
    if not CLASSES:
        raise RuntimeError("No prototypes.json found.")
    v = wav_to_embedding(wav_path)      # [D]
    sims = MAT @ v                      # [C] cosine (vectors are L2-normalized)
    p = softmax(sims / temperature)     # convert to probabilities
    i = int(np.argmax(p))
    return {"method":"prototype","bird":CLASSES[i],"confidence":float(p[i]),"probs":p.tolist()}

def classify_head(wav_path):
    if HEAD is None:
        raise RuntimeError("No bird_head.joblib found. Train with head_train.py.")
    CLF, HEAD_CLASSES = HEAD
    v = wav_to_embedding(wav_path).reshape(1, -1)     # [1, D]
    logits = CLF.decision_function(v)                 # [1, C]
    p = softmax(logits, axis=1)[0]
    i = int(np.argmax(p))
    return {"method":"head","bird":HEAD_CLASSES[i],"confidence":float(p[i]),"probs":p.tolist()}

def classify(wav_path, use_head=True):
    if use_head and HEAD is not None:
        return classify_head(wav_path)
    elif CLASSES:
        return classify_prototype(wav_path)
    else:
        raise RuntimeError("Neither bird_head.joblib nor prototypes.json available.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python proto_infer.py path/to/clip.wav [--no-head]")
        sys.exit(1)
    use_head = not ("--no-head" in sys.argv)
    print(classify(sys.argv[1], use_head=use_head))
