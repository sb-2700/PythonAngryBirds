import numpy as np
import os
import librosa

def get_time_averaged_db(audio_file_path, ref_level=1.0):
    """
    Calculate the time-averaged dB level from an audio file.
    
    Parameters:
        audio_file_path (str): Path to the audio file
        ref_level (float): Reference level for dB calculation (default: 1.0)
    
    Returns:
        float: Time-averaged dB level
    """
    # Load the audio file
    y, sr = librosa.load(audio_file_path)
    
    # Convert amplitude to power
    power = np.abs(y)**2
    
    # Calculate dB values (10 * log10 for power)
    db_values = 10 * np.log10(power + 1e-10)
    
    # Calculate time-averaged dB
    avg_db = np.mean(db_values)
    
    return avg_db

def get_pitch(audio_path):
    """
    Get the average pitch of an audio file.
    Returns frequency in Hz.
    """
    y, sr = librosa.load(audio_path)
    # Use librosa's pitch detection
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    # Get the highest magnitude pitch at each time
    pitch_vals = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch_vals.append(pitches[index, t])
    # Return the average of non-zero pitches
    valid_pitches = [p for p in pitch_vals if p > 0]
    return np.mean(valid_pitches) if valid_pitches else 0

def get_angle():
    default_angle = 135
    """
    Calculate the angle based on pitch.
    Returns angle in radians.
    """
    sample_sound_2_available = False  # Debug flag
    if not sample_sound_2_available:
        print("Debug mode: Standard angle returned")
        return ( default_angle / 360) * 2 * np.pi  # 45 degrees in radians

    try:
        audio_path = "audio/power_sounds/sample_sound_2.opus"
        if os.path.exists(audio_path):
            pitch = get_pitch(audio_path)
            # Map pitch to angle (from -45° to 90°)
            # Lower frequencies will give negative angles (upward shots)
            angle_degrees = np.clip(np.interp(pitch, [100, 1000], [90, 180]), 90, 180)
            return (angle_degrees / 360) * 2 * np.pi
        else:
            print("Audio file not found, using default angle")
            return (default_angle / 360) * 2 * np.pi
    except Exception as e:
        print(f"Error calculating angle: {e}")
        return (default_angle / 360) * 2 * np.pi

def get_distance():
    dB_range = (0, 90)
    """
    Calculate the pull distance based on time-averaged dB.
    Returns distance in pixels scaled quadratically (0-90).
    """
    sample_sound_2_available = False  # Debug flag
    if not sample_sound_2_available:
        print("Debug mode: Standard power returned")
        return 45

    try:
        audio_path = "audio/power_sounds/sample_sound_2.opus"
        if os.path.exists(audio_path):
            avg_dB = get_time_averaged_db(audio_path)
            # Map dB to 0-1 range first
            scaling_factor = (dB_range[1] - dB_range[0]) / 90 # *90 for full range of sling shot
            distance = avg_dB * scaling_factor 
            return distance
        else:
            print("Audio file not found, using default distance")
            return 45
    except Exception as e:
        print(f"Error calculating distance: {e}")
        return 45

if __name__ == "__main__":
    # Test the audio-based calculations
    angle = get_angle()
    distance = get_distance()
    
    print(f"Calculated angle: {angle:.2f} radians ({(angle * 360 / (2 * np.pi)):.2f} degrees)")
    print(f"Calculated distance: {distance:.2f} pixels")