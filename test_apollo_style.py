#!/usr/bin/env python3
"""
Test script to verify Apollo-style recognition is working
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import cv2
import numpy as np
from tlr.pipeline import load_pipeline

def test_single_frame():
    """Test Apollo-style recognition on a single frame"""
    
    # Load pipeline
    pipeline = load_pipeline('cuda:0')
    print("✅ Pipeline loaded with Apollo-style recognition")
    
    # Test with a few frames from your video
    test_frames = [
        'test_original_video_nuevo/frame_0192.jpg',  # Should be red/yellow
        'test_original_video_nuevo/frame_0200.jpg',  # Should be red
        'test_original_video_nuevo/frame_0050.jpg',  # Should be green
    ]
    
    # Load projection boxes
    bboxes_per_frame = {}
    with open('test_original_video_nuevo/projection_bboxes_master.txt', 'r') as f:
        for line in f:
            parts = line.strip().split(",")
            frame = parts[0]
            bbox = list(map(int, parts[1:]))
            bboxes_per_frame.setdefault(frame, []).append(bbox)
    
    for frame_path in test_frames:
        if not os.path.exists(frame_path):
            print(f"❌ Frame not found: {frame_path}")
            continue
            
        frame_name = os.path.basename(frame_path)
        if frame_name not in bboxes_per_frame:
            print(f"❌ No bboxes for {frame_name}")
            continue
            
        print(f"\n🔍 Testing {frame_name}")
        
        # Load image
        image_np = cv2.imread(frame_path)
        image_tensor = torch.from_numpy(image_np.astype(np.float32)).to('cuda:0')
        
        # Get projection boxes for this frame
        bboxes = bboxes_per_frame[frame_name]
        print(f"  📊 Projection boxes: {len(bboxes)}")
        
        # Run pipeline with debug logging enabled
        # Temporarily enable debug
        temp_debug_line = """
            # Apollo-style logging (uncomment for debug)
            print(f"Light status recognized as {['BLACK', 'RED', 'YELLOW', 'GREEN'][color_id]}")
            print(f"Color Prob: {output_probs.tolist()}")
            print(f"Max prob: {max_prob:.4f}, Threshold: {threshold}")
        """
        
        # Process
        valid_detections, recognitions, assignments, invalid_detections, revised_states = pipeline(
            image_tensor, bboxes, frame_ts=0.0
        )
        
        print(f"  ✅ Valid detections: {len(valid_detections)}")
        print(f"  ✅ Recognitions shape: {recognitions.shape}")
        
        # Decode recognition results
        if len(recognitions) > 0:
            colors = ['BLACK', 'RED', 'YELLOW', 'GREEN']
            for i, recognition in enumerate(recognitions):
                color_idx = torch.argmax(recognition).item()
                confidence = recognition[color_idx].item()
                print(f"    🚦 Detection {i}: {colors[color_idx]} (confidence: {confidence:.3f})")
        
        print("  " + "="*50)

if __name__ == "__main__":
    test_single_frame()
    print("\n🎯 Test completed! Check if you now see YELLOW detections.")