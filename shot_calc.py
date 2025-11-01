import numpy as np
import os
import librosa
from matplotlib import pyplot as plt

def get_time_averaged_db(audio_file_path, ref_level=1.0):
    """
    Calculate the time-averaged dB level from an audio file.
    Preserves relative loudness between whispers and shouts.
    
    Parameters:
        audio_file_path (str): Path to the audio file
        ref_level (float): Reference level for dB calculation (default: 1.0)
    
    Returns:
        float: Time-averaged dB level, higher values indicate louder sounds
    """
    # Load the audio file
    y, sr = librosa.load(audio_file_path)
    
    # Convert amplitude to power
    power = np.abs(y)**2
    
    # Calculate dB values (20 * log10 for amplitude)
    # Add large offset to make values positive for game mechanics
    db_values = 20 * np.log10(np.sqrt(power) + 1e-12) + 60
    
    # Remove any remaining negative values
    db_values = np.maximum(db_values, 0)
    
    # Calculate time-averaged dB
    avg_db = np.mean(db_values)

    '''plt.figure(figsize=(10, 4))
    plt.plot(db_values, color='blue')
    plt.title(f'dB Values Over Time (avg: {avg_db:.2f} dB)')
    plt.ylabel('dB')
    plt.xlabel('Time (samples)')
    plt.show()'''
    
    return int(avg_db)

def get_duration(audio_path):
    """
    Get the duration of an audio file in seconds.
    
    Parameters:
        audio_path (str): Path to the audio file
    
    Returns:
        float: Duration of the audio in seconds
    """
    # Load the audio file
    y, sr = librosa.load(audio_path)
    
    # Calculate duration
    duration = librosa.get_duration(y=y, sr=sr)
    
    return duration

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
    power_sound_2_available = True#False#True #False  # Debug flag
    if not power_sound_2_available:
        print("Debug mode: Standard angle returned")
        return ( default_angle / 360) * 2 * np.pi  # 45 degrees in radians

    try:
        print("Calculating angle from pitch...")
        audio_path = "audio/power_sounds/power_sound_2.wav"
        if os.path.exists(audio_path):
            print("Path found")
            pitch = get_pitch(audio_path)
            # Map pitch to angle (from -45° to 90°)
            # Lower frequencies will give negative angles (upward shots)
            angle_degrees = np.clip(np.interp(pitch, [100, 1000], [180, 90]), 90, 180)
            print(f"Pitch detected: {pitch} Hz, Angle: {angle_degrees} degrees")
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
    power_sound_2_available = True #True #False  # Debug flag
    if not power_sound_2_available:
        print("Debug mode: Standard power returned")
        return 45

    try:
        audio_path = "audio/power_sounds/power_sound_2.wav"
        if os.path.exists(audio_path):
            T_period = get_duration(audio_path)
            # Map dB to 0-1 range first
            scaled_factor = T_period /3 # *90 for full range of sling shot
            distance = 90 * scaled_factor 
            print(f"Time of recording: {T_period}, Power of shot 1-3s is time range: {distance}")
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