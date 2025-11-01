# embeddings.py
import numpy as np, soundfile as sf, librosa, torch
from panns_inference import AudioTagging

DEVICE = "cpu"  # or "cuda" if available
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = AudioTagging(checkpoint_path=None, device=DEVICE)  # downloads weights on first use
    return _model

''' Audio Preprocessing'''
def load_mono_16k(path, target_sr=16000, trim_db=30):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1: y = y.mean(axis=1)
    if sr != target_sr: y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    # optional silence trim
    yt, _ = librosa.effects.trim(y, top_db=trim_db)
    if len(yt) == 0: yt = y
    return yt

def wav_to_embedding(path):
    """Return L2-normalized 2048-D embedding for a wav file."""
    y = load_mono_16k(path)
    model = _get_model()

    # Add batch dimension: model expects shape [batch_size, samples]
    y_batch = y[None, :]  # Convert from [samples] to [1, samples]

    with torch.no_grad():
        _, emb = model.inference(y_batch)   # emb shape: [T, 2048]
    v = emb.mean(axis=0)              # temporal average to fixed-size
    v = v / (np.linalg.norm(v) + 1e-10)
    return v.astype(np.float32)
