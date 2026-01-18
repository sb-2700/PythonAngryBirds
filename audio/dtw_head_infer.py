# dtw_head_infer.py
# Combines DTW-on-mel with trained linear head (instead of cosine similarity)
import json, sys, joblib
import numpy as np
import librosa
from embeddings import load_mono_16k, wav_to_embedding

# Load trained head classifier
HEAD_DATA = joblib.load("bird_head.joblib")
HEAD_CLF = HEAD_DATA["clf"]
HEAD_CLASSES = HEAD_DATA["classes"]

# Load mel prototypes for DTW
MEL_PROTOS = {k: np.array(v, dtype=np.float32)
              for k, v in json.load(open("mel_prototypes.json")).items()}

def wav_to_logmel(y, sr=16000, n_mels=64, win=1024, hop=256):
    """Waveform -> log-mel spectrogram (per-mel z-normalized)."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=win, hop_length=hop,
        n_mels=n_mels, fmin=50, fmax=sr // 2
    )
    L = librosa.power_to_db(S, ref=np.max)
    mu = L.mean(axis=1, keepdims=True)
    sigma = L.std(axis=1, keepdims=True) + 1e-6
    L = (L - mu) / sigma
    return L.astype(np.float32)

def _softmax(x, T=0.25):
    z = x / T
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()

def classify_dtw_head(wav_path, alpha=0.7):
    """
    Combines trained linear head (alpha weight) with DTW mel matching (1-alpha weight)
    Much better than cosine similarity!
    """
    y = load_mono_16k(wav_path)
    
    # 1) Linear Head Prediction (TRAINED on multiple samples)
    embedding = wav_to_embedding(wav_path)
    embedding = embedding.reshape(1, -1)
    head_probs = HEAD_CLF.predict_proba(embedding)[0]  # Proper learned probabilities
    
    # 2) DTW on mel spectrograms
    M = wav_to_logmel(y, sr=16000)
    dtw_scores = []
    for cls in HEAD_CLASSES:  # Use same class order as head
        if cls in MEL_PROTOS:
            Tmpl = MEL_PROTOS[cls]
            D, _ = librosa.sequence.dtw(X=Tmpl, Y=M, metric="cosine")
            dist = float(D[-1, -1])
            dtw_scores.append(-dist)  # Higher is better
        else:
            dtw_scores.append(0.0)  # Default if no mel prototype
    
    dtw_scores = np.array(dtw_scores, dtype=np.float32)
    
    # 3) Normalize DTW scores to probabilities
    dtw_probs = _softmax(dtw_scores, T=0.25)
    
    # 4) Blend: Trained head + DTW (no need for z-score, both are probabilities)
    final_probs = alpha * head_probs + (1.0 - alpha) * dtw_probs
    
    idx = int(np.argmax(final_probs))
    predicted_class = HEAD_CLASSES[idx]
    confidence = float(final_probs[idx])
    
    return {
        "bird": predicted_class,
        "confidence": confidence,
        "final_probs": final_probs.tolist(),
        "head_probs": head_probs.tolist(),
        "dtw_probs": dtw_probs.tolist(),
        "alpha": alpha,
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dtw_head_infer.py path/to/sample.wav")
        sys.exit(1)
    
    result = classify_dtw_head(sys.argv[1])
    print(f"Predicted: {result['bird']} (confidence: {result['confidence']:.3f})")
    print(f"Head contribution: {result['alpha']:.1%}, DTW contribution: {1-result['alpha']:.1%}")