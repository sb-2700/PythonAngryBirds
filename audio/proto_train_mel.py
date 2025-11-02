# proto_train_mel.py
import os, json, numpy as np
import librosa
from embeddings import load_mono_16k

# --- Reuse your filename->class mapping
FILE_TO_CLASS = {
    "Red_bird_fly.wav": "red",
    "Blue_bird_fly.wav": "blue",
    "Chuck_yellow_bird_fly.wav": "yellow",
    "Matilda_white_bird_fly.wav": "white",
}

REF_DIR = "Bird_audios_wav"

def wav_to_logmel(y, sr=16000, n_mels=64, win=1024, hop=256):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=win, hop_length=hop, n_mels=n_mels, fmin=50, fmax=sr//2
    )
    L = librosa.power_to_db(S, ref=np.max)
    # Per-frequency normalize (helps room/mic variance)
    mu = L.mean(axis=1, keepdims=True)
    sigma = L.std(axis=1, keepdims=True) + 1e-6
    L = (L - mu) / sigma
    return L.astype(np.float32)  # [mels, T]

def main():
    # 1) group reference paths by class
    class_files = {}
    for fn in os.listdir(REF_DIR):
        if not fn.lower().endswith(".wav"):
            continue
        if fn in FILE_TO_CLASS:
            cls = FILE_TO_CLASS[fn].lower()
            class_files.setdefault(cls, []).append(os.path.join(REF_DIR, fn))

    if not class_files:
        raise SystemExit("No reference files found for mel prototypes.")

    mel_protos = {}

    # 2) build a mel template per class (median of aligned mels)
    for cls, paths in class_files.items():
        mels = []
        for p in paths:
            y = load_mono_16k(p)           # same preprocessing
            M = wav_to_logmel(y)                   # [mels, T]
            mels.append(M)

        # time-align by center-cropping to the minimum T
        min_T = min(M.shape[1] for M in mels)
        stack = np.stack([M[:, :min_T] for M in mels], axis=0)  # [N, mels, T]
        proto = np.median(stack, axis=0)                        # robust center
        mel_protos[cls] = proto.tolist()
        print(f"Built mel prototype for {cls} using {len(paths)} file(s), T={min_T}")

    # 3) save
    with open("mel_prototypes.json", "w") as f:
        json.dump(mel_protos, f)
    print("Saved mel_prototypes.json")

if __name__ == "__main__":
    main()
