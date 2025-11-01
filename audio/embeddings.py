# embeddings.py
import numpy as np, soundfile as sf, librosa, torch
from panns_inference import AudioTagging

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Auto-detect GPU
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = AudioTagging(checkpoint_path=None, device=DEVICE)  # downloads weights on first use
    return _model

def _rms_db(y):
    """Calculate RMS in dB"""
    rms = float(np.sqrt(np.mean(y**2) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)

def _apply_gain_db(y, gain_db):
    """Apply gain in dB to audio"""
    return y * (10.0 ** (gain_db / 20.0))

''' Audio Preprocessing'''
def load_mono_16k(path, target_sr=16000, trim_db=30.0, rms_target_db=-20.0, duration_s=2.0, start_at_first_sound=True):
    """Load -> mono -> 16 kHz -> trim silence -> loudness normalize -> cut/pad"""
    # 1) Load audio
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1: 
        y = y.mean(axis=1)
    
    # 2) Resample if needed
    if sr != target_sr: 
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    # 3) Trim leading/trailing silence
    yt, idx = librosa.effects.trim(y, top_db=trim_db)
    if len(yt) == 0:
        # All silence -> return 2s of zeros
        return np.zeros(int(target_sr * duration_s), dtype=np.float32)
    
    if start_at_first_sound:
        # Start window at first non-silent sample
        start = int(idx[0])
        y = y[start:]
    else:
        y = yt
    
    # 4) Loudness normalize by RMS
    current_db = _rms_db(y)
    gain_db = float(rms_target_db - current_db)
    y = _apply_gain_db(y, gain_db)
    
    # 5) Fix duration: cut/pad to exactly duration_s
    T = int(target_sr * duration_s)
    if len(y) < T:
        y = np.pad(y, (0, T - len(y)))
    else:
        y = y[:T]
    
    return y.astype(np.float32)

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

def cosine_sim(a, b):
    """Calculate cosine similarity between two embeddings"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))