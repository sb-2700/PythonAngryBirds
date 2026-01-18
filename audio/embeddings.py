# emb# =============================================================================
# MODEL SELECTION - CHANGE THIS LINE TO SWITCH MODELS
# =============================================================================
USE_MODEL = "PANNS"        # Options: "PANNS", "AST", or "EFFICIENTAT"
# USE_MODEL = ""          # Audio Spectrogram Transformer
# USE_MODEL = "EFFICIENTAT"  # EfficientAT-B2 (fast and accurate)
# =============================================================================s.py
import numpy as np, soundfile as sf, librosa, torch
from panns_inference import AudioTagging
from transformers import AutoFeatureExtractor, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Auto-detect GPU

# Model variables
_panns_model = None
_ast_model = None
_ast_extractor = None
_efficientat_model = None
_efficientat_extractor = None

AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
# EfficientAT model - using HuBERT for reliable efficient audio processing
EFFICIENTAT_MODEL_NAME = "facebook/hubert-base-ls960"  # HuBERT efficient audio model

def _get_panns_model():
    global _panns_model
    if _panns_model is None:
        _panns_model = AudioTagging(checkpoint_path=None, device=DEVICE)
    return _panns_model

def _get_ast_model():
    global _ast_model, _ast_extractor
    if _ast_model is None:
        _ast_extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL_NAME)
        _ast_model = AutoModel.from_pretrained(AST_MODEL_NAME).to(DEVICE).eval()
    return _ast_extractor, _ast_model

def _get_efficientat_model():
    global _efficientat_model, _efficientat_extractor
    if _efficientat_model is None:
        # Use HuBERT - efficient, reliable, and great for audio classification
        print("Loading HuBERT efficient audio model...")
        _efficientat_extractor = AutoFeatureExtractor.from_pretrained(EFFICIENTAT_MODEL_NAME)
        _efficientat_model = AutoModel.from_pretrained(EFFICIENTAT_MODEL_NAME).to(DEVICE).eval()
        print("✓ HuBERT model loaded successfully")
    return _efficientat_extractor, _efficientat_model

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
    """Return L2-normalized embedding for a wav file."""
    y = load_mono_16k(path)
    
    if USE_MODEL == "PANNS":
        # PANNs model inference
        model = _get_panns_model()
        y_batch = y[None, :]  # Add batch dimension for PANNs
        
        with torch.no_grad():
            _, emb = model.inference(y_batch)   # emb shape: [T, 2048]
        v = emb.mean(axis=0)              # temporal average to fixed-size
        
    elif USE_MODEL == "AST":
        # AST model inference
        extractor, model = _get_ast_model()
        inputs = extractor(y, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
            v = embeddings.mean(dim=1).squeeze(0)  # Pool over time, remove batch dim
        
        v = v.cpu().numpy()  # Convert to numpy for AST
    
    elif USE_MODEL == "EFFICIENTAT":
        # EfficientAT model inference (using HuBERT)
        processor, model = _get_efficientat_model()
        inputs = processor(y, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # HuBERT outputs last_hidden_state
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
                v = embeddings.mean(dim=1).squeeze(0)  # Pool over time, remove batch dim
            elif hasattr(outputs, 'extract_features'):
                # Some HuBERT models have this
                v = outputs.extract_features.mean(dim=1).squeeze(0)
            else:
                # Fallback: use first tensor output
                v = list(outputs.values())[0]
                if v.dim() > 1:
                    v = v.mean(dim=1).squeeze(0)
                else:
                    v = v.squeeze(0)
        
        v = v.cpu().numpy()  # Convert to numpy for EfficientAT
    
    else:
        raise ValueError(f"Unknown model: {USE_MODEL}")
    
    # L2 normalize
    v = v / (np.linalg.norm(v) + 1e-10)
    return v.astype(np.float32)

def cosine_sim(a, b):
    """Calculate cosine similarity between two embeddings"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))