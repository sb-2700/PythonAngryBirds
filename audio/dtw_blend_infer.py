# dtw_blend_infer.py
# Blends PANNs-embedding cosine similarity with DTW-on-mel score.
# Usage: python dtw_blend_infer.py path/to/sample.wav

import json
import sys
import numpy as np
import librosa

from embeddings import load_mono_16k, wav_to_embedding  # from your embeddings.py

# --- Load embedding prototypes (from proto_train.py)
PROTOS = {k: np.array(v, dtype=np.float32)
          for k, v in json.load(open("prototypes.json")).items()}
CLASSES = list(PROTOS.keys())
MAT = np.stack([PROTOS[c] for c in CLASSES], axis=0)  # [C, 2048]

# --- Load mel prototypes (from proto_train_mel.py)
MEL_PROTOS = {k: np.array(v, dtype=np.float32)
              for k, v in json.load(open("mel_prototypes.json")).items()}

def wav_to_logmel(
    y: np.ndarray,
    sr: int = 16000,
    n_mels: int = 64,
    win: int = 1024,
    hop: int = 256,
) -> np.ndarray:
    """Waveform -> log-mel spectrogram (per-mel z-normalized)."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=win, hop_length=hop,
        n_mels=n_mels, fmin=50, fmax=sr // 2
    )
    L = librosa.power_to_db(S, ref=np.max)  # [mels, T]
    mu = L.mean(axis=1, keepdims=True)
    sigma = L.std(axis=1, keepdims=True) + 1e-6
    L = (L - mu) / sigma
    return L.astype(np.float32)

def _softmax(x: np.ndarray, T: float = 0.25) -> np.ndarray:
    z = x / T
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()

def classify_blend(wav_path: str, alpha: float = 0.6) -> dict:
    """
    alpha ∈ [0,1]: weight for embedding similarity; (1-alpha) for DTW-mel score.
    Returns dict with bird, confidence, probs, and component scores.
    """
    # 1) Preprocess once (your loader: trim -> rms -> ~2s from first sound)
    y = load_mono_16k(wav_path)

    # 2) Embedding cosine sims (higher is better)
    v = wav_to_embedding(wav_path)      # uses same loader internally
    emb_sims = MAT @ v                  # [C]

    # 3) DTW on mel against each class template (lower distance -> higher score)
    M = wav_to_logmel(y, sr=16000)      # [mels, T]
    dtw_scores = []
    for cls in CLASSES:
        Tmpl = MEL_PROTOS[cls]          # [mels, T0]
        # cosine metric inside DTW; returns accumulated cost matrix D
        D, _ = librosa.sequence.dtw(X=Tmpl, Y=M, metric="cosine")
        dist = float(D[-1, -1])
        dtw_scores.append(-dist)        # negate so "higher is better"

    dtw_scores = np.array(dtw_scores, dtype=np.float32)

    # 4) Normalize components (z-score) to comparable scales, then blend
    emb_n = (emb_sims - emb_sims.mean()) / (emb_sims.std() + 1e-6)
    dtw_n = (dtw_scores - dtw_scores.mean()) / (dtw_scores.std() + 1e-6)
    blended = alpha * emb_n + (1.0 - alpha) * dtw_n

    # 5) Softmax to probabilities
    probs = _softmax(blended, T=0.25)
    idx = int(np.argmax(probs))

    return {
        "bird": CLASSES[idx],
        "confidence": float(probs[idx]),
        "probs": probs.tolist(),
        "emb_scores": emb_sims.tolist(),
        "dtw_scores": dtw_scores.tolist(),
        "alpha": alpha,
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dtw_blend_infer.py path/to/sample.wav")
        sys.exit(1)
    print(classify_blend(sys.argv[1]))
