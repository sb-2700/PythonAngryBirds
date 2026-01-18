# audio_setup.py
import os
from pydub import AudioSegment
from pydub.utils import which

# Prefer system-installed ffmpeg if available, otherwise fall back to C:\ffmpeg\bin
ffmpeg = which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"
ffprobe = which("ffprobe") or r"C:\ffmpeg\bin\ffprobe.exe"

# Ensure PATH contains the ffmpeg folder (helps subprocess lookups)
ffmpeg_dir = os.path.dirname(ffmpeg) if os.path.exists(ffmpeg) else r"C:\ffmpeg\bin"
os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + ffmpeg_dir

# Set pydub configuration
AudioSegment.converter = ffmpeg
AudioSegment.ffmpeg = ffmpeg
AudioSegment.ffprobe = ffprobe

# Optional debug output (remove when stable)
print("audio_setup: ffmpeg ->", ffmpeg)
print("audio_setup: ffprobe ->", ffprobe)
print("audio_setup: PATH contains ffmpeg_dir ->", ffmpeg_dir in os.environ["PATH"])