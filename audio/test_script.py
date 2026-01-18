#!/usr/bin/env python3
"""
Automated testing script for sample_sounds folder.
Runs all .wav files through proto_infer_head.py and compares predictions with actual labels.
"""

import os
import sys
import subprocess
import re
from pathlib import Path
from collections import defaultdict, Counter

def extract_true_label(filename):
    """Extract the true bird label from filename (first word before underscore)"""
    # Remove file extension
    name = Path(filename).stem
    
    # Extract first word (bird color)
    parts = name.split('_')
    if len(parts) > 0:
        return parts[0].lower()
    return None

def run_inference(wav_file):
    """Run proto_infer_head.py on a single wav file and parse the result"""
    try:
        # Run the inference script
        result = subprocess.run(
            [sys.executable, "proto_infer_head.py", wav_file],
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        if result.returncode != 0:
            print(f"Error running inference on {wav_file}:")
            print(result.stderr)
            return None, None
        
        # Parse the output to extract prediction and confidence
        output = result.stdout.strip()
        
        # Look for patterns like "Predicted: red (confidence: 0.856)"
        prediction_match = re.search(r"Predicted:\s*(\w+)", output, re.IGNORECASE)
        confidence_match = re.search(r"confidence:\s*([\d.]+)", output, re.IGNORECASE)
        
        if prediction_match:
            prediction = prediction_match.group(1).lower()
            confidence = float(confidence_match.group(1)) if confidence_match else 0.0
            return prediction, confidence
        else:
            print(f"Could not parse prediction from output: {output}")
            return None, None
            
    except subprocess.TimeoutExpired:
        print(f"Timeout running inference on {wav_file}")
        return None, None
    except Exception as e:
        print(f"Error running inference on {wav_file}: {e}")
        return None, None

def main():
    # Check if sample_sounds folder exists
    sample_dir = "sample_sounds"
    if not os.path.exists(sample_dir):
        print(f"Error: {sample_dir} folder not found!")
        return
    
    # Check if proto_infer_head.py exists
    if not os.path.exists("proto_infer_head.py"):
        print("Error: proto_infer_head.py not found!")
        return
    
    # Find all .wav files in sample_sounds
    wav_files = []
    for file in os.listdir(sample_dir):
        if file.lower().endswith('.wav'):
            wav_files.append(file)
    
    if not wav_files:
        print(f"No .wav files found in {sample_dir} folder!")
        return
    
    print(f"Found {len(wav_files)} .wav files in {sample_dir}")
    print("=" * 60)
    
    # Track results
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    # Stats by class
    class_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'predictions': []})
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    # Process each file
    for i, filename in enumerate(sorted(wav_files), 1):
        wav_path = os.path.join(sample_dir, filename)
        true_label = extract_true_label(filename)
        
        if true_label is None:
            print(f"⚠️  {i:2d}/{len(wav_files)} {filename:<25} - Could not extract true label")
            continue
        
        print(f"🔍 {i:2d}/{len(wav_files)} {filename:<25} (true: {true_label:<6}) - ", end="")
        
        # Run inference
        prediction, confidence = run_inference(wav_path)
        
        if prediction is None:
            print("❌ Failed")
            continue
        
        # Check if correct
        is_correct = prediction == true_label
        if is_correct:
            correct_predictions += 1
            status = "✅"
        else:
            status = "❌"
        
        total_predictions += 1
        
        # Update stats
        class_stats[true_label]['total'] += 1
        class_stats[true_label]['predictions'].append(prediction)
        if is_correct:
            class_stats[true_label]['correct'] += 1
        
        confusion_matrix[true_label][prediction] += 1
        
        # Store result
        results.append({
            'filename': filename,
            'true_label': true_label,
            'prediction': prediction,
            'confidence': confidence,
            'correct': is_correct
        })
        
        print(f"{status} pred: {prediction:<6} (conf: {confidence:.3f})")
    
    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if total_predictions > 0:
        overall_accuracy = correct_predictions / total_predictions
        print(f"Overall Accuracy: {correct_predictions}/{total_predictions} = {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
        print()
        
        # Per-class accuracy
        print("Per-Class Accuracy:")
        print("-" * 30)
        for cls in sorted(class_stats.keys()):
            stats = class_stats[cls]
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                print(f"{cls:<8}: {stats['correct']:2d}/{stats['total']:2d} = {accuracy:.3f} ({accuracy*100:.1f}%)")
        print()
        
        # Confusion Matrix
        print("Confusion Matrix:")
        print("-" * 20)
        all_classes = sorted(set(list(confusion_matrix.keys()) + 
                                [pred for preds in confusion_matrix.values() for pred in preds.keys()]))
        
        header_label = "True\\Pred"
        print(f"{header_label:<10}", end="")
        for pred_cls in all_classes:
            print(f"{pred_cls:<8}", end="")
        print()
        
        for true_cls in all_classes:
            print(f"{true_cls:<10}", end="")
            for pred_cls in all_classes:
                count = confusion_matrix[true_cls][pred_cls]
                print(f"{count:<8}", end="")
            print()
        print()
        
        # Most confused cases
        print("Incorrect Predictions:")
        print("-" * 25)
        for result in results:
            if not result['correct']:
                print(f"{result['filename']:<25} {result['true_label']} → {result['prediction']} (conf: {result['confidence']:.3f})")
    
    else:
        print("No valid predictions made!")

if __name__ == "__main__":
    main()