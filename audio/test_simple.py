#!/usr/bin/env python3
"""
Simple test script to debug the issue
"""

import os
import sys
import subprocess

def main():
    print("=== STARTING TEST SCRIPT ===")
    
    # Check current directory
    print(f"Current directory: {os.getcwd()}")
    
    # Check if sample_sounds folder exists
    sample_dir = "sample_sounds"
    print(f"Checking if {sample_dir} exists: {os.path.exists(sample_dir)}")
    
    if not os.path.exists(sample_dir):
        print(f"ERROR: {sample_dir} folder not found!")
        return
    
    # Check if proto_infer_head.py exists
    print(f"Checking if proto_infer_head.py exists: {os.path.exists('proto_infer_head.py')}")
    
    if not os.path.exists("proto_infer_head.py"):
        print("ERROR: proto_infer_head.py not found!")
        return
    
    # List all files in sample_sounds
    print(f"\nListing files in {sample_dir}:")
    all_files = os.listdir(sample_dir)
    print(f"Total files found: {len(all_files)}")
    
    for i, file in enumerate(all_files[:10]):  # Show first 10
        print(f"  {i+1}. {file}")
    
    # Find .wav files specifically
    wav_files = [f for f in all_files if f.lower().endswith('.wav')]
    print(f"\nWAV files found: {len(wav_files)}")
    
    if not wav_files:
        print("ERROR: No .wav files found!")
        return
    
    for i, file in enumerate(wav_files[:5]):  # Show first 5 WAV files
        print(f"  WAV {i+1}. {file}")
    
    # Test running inference on one file
    test_file = wav_files[0]
    test_path = os.path.join(sample_dir, test_file)
    print(f"\nTesting inference on: {test_path}")
    
    try:
        print("Running command:", [sys.executable, "proto_infer_head.py", test_path])
        
        result = subprocess.run(
            [sys.executable, "proto_infer_head.py", test_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT length: {len(result.stdout)}")
        print(f"STDERR length: {len(result.stderr)}")
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
    except Exception as e:
        print(f"Exception occurred: {e}")
    
    print("=== TEST COMPLETE ===")

if __name__ == "__main__":
    main()