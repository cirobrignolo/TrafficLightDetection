#!/usr/bin/env python3
"""
Test script to verify ID switching behavior during yellow blinking
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import cv2
import numpy as np
from tlr.pipeline import load_pipeline
import shutil

def create_id_switched_test():
    """
    Create a test where IDs switch in the middle of yellow blinking
    """
    
    # Load original projection file
    original_file = 'test_original_video_nuevo/projection_bboxes_master.txt'
    test_file = 'test_id_switching_bboxes.txt'
    
    print("🔧 Creating ID-switched projection file...")
    
    # Read original projections
    projections = []
    with open(original_file, 'r') as f:
        for line in f:
            parts = line.strip().split(",")
            frame = parts[0]
            bbox = list(map(int, parts[1:]))  # [x1, y1, x2, y2, proj_id]
            projections.append((frame, bbox))
    
    # Define switching point (middle of yellow blinking)
    switch_frame = 215  # Around frame_0215.jpg (middle of first yellow sequence)
    
    # Write modified projections
    with open(test_file, 'w') as f:
        for frame, bbox in projections:
            frame_num = int(frame.replace('frame_', '').replace('.jpg', ''))
            
            if frame_num >= switch_frame:
                # SWITCH IDs: 0 -> 1, 1 -> 0 (or -1 -> -2, -2 -> -1)
                if bbox[4] == 0:
                    bbox[4] = 1
                elif bbox[4] == -1:  # Second traffic light
                    bbox[4] = -2
                # Add more cases if needed
            
            # Write line
            f.write(f"{frame},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{bbox[4]}\n")
    
    print(f"✅ Created {test_file} with ID switch at frame {switch_frame}")
    return test_file

def run_id_switching_test():
    """
    Run the test with ID switching
    """
    
    # Create test file
    bbox_file = create_id_switched_test()
    
    # Setup
    input_dir = 'test_original_video_nuevo'
    output_dir = 'test_id_switching_results'
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Running ID switching test...")
    print(f"📂 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Load pipeline
    pipeline = load_pipeline('cuda:0')
    
    # Read projections
    entries = {}
    with open(bbox_file, 'r') as f:
        for line in f:
            parts = line.strip().split(",")
            frame = parts[0]
            bbox = list(map(int, parts[1:]))
            entries.setdefault(frame, []).append(bbox)
    
    # Process frames around the switch point
    test_frames = [
        'frame_0210.jpg', 'frame_0211.jpg', 'frame_0212.jpg', 'frame_0213.jpg', 'frame_0214.jpg',
        'frame_0215.jpg',  # ← SWITCH POINT
        'frame_0216.jpg', 'frame_0217.jpg', 'frame_0218.jpg', 'frame_0219.jpg', 'frame_0220.jpg'
    ]
    
    results = []
    
    for frame_idx, frame_name in enumerate(test_frames):
        if frame_name not in entries:
            continue
            
        print(f"\n🔍 Processing {frame_name} (frame {frame_idx})")
        
        # Load image
        frame_path = os.path.join(input_dir, frame_name)
        if not os.path.exists(frame_path):
            continue
            
        image_np = cv2.imread(frame_path)
        image_tensor = torch.from_numpy(image_np.astype(np.float32)).to('cuda:0')
        
        # Get bboxes
        bboxes = entries[frame_name]
        print(f"  📊 Projections: {bboxes}")
        
        # Process
        frame_ts = frame_idx * (1.0/29)  # Simulate frame timing
        valid_detections, recognitions, assignments, invalid_detections, revised_states = pipeline(
            image_tensor, bboxes, frame_ts=frame_ts
        )
        
        # Analyze results
        print(f"  ✅ Valid detections: {len(valid_detections)}")
        print(f"  ✅ Assignments: {assignments.tolist() if len(assignments) > 0 else []}")
        if revised_states:
            print(f"  ✅ Revised states: {revised_states}")
        
        # Decode results
        if len(recognitions) > 0:
            colors = ['BLACK', 'RED', 'YELLOW', 'GREEN']
            assign_map = {int(assignment[0]): int(assignment[1]) for assignment in assignments}
            
            for det_idx, recognition in enumerate(recognitions):
                color_idx = torch.argmax(recognition).item()
                confidence = recognition[color_idx].item()
                proj_id = assign_map.get(det_idx, -999)
                
                # Get revised state
                revised_info = ""
                if revised_states and proj_id in revised_states:
                    revised_color, is_blinking = revised_states[proj_id]
                    revised_info = f" -> {revised_color.upper()}" + (" (BLINK)" if is_blinking else "")
                
                print(f"    🚦 Det {det_idx} -> Proj {proj_id}: {colors[color_idx]}{revised_info} (conf: {confidence:.3f})")
                
                results.append({
                    'frame': frame_name,
                    'frame_idx': frame_idx,
                    'det_idx': det_idx,
                    'proj_id': proj_id,
                    'detected_color': colors[color_idx],
                    'confidence': confidence,
                    'revised_color': revised_states[proj_id][0] if revised_states and proj_id in revised_states else colors[color_idx],
                    'is_blinking': revised_states[proj_id][1] if revised_states and proj_id in revised_states else False
                })
    
    # Save results
    import csv
    with open(f'{output_dir}/id_switching_results.csv', 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\n📄 Results saved to {output_dir}/id_switching_results.csv")
    
    # Analysis
    print(f"\n🔍 ANALYSIS:")
    print(f"   Before frame 215: Should see proj_id=0 with YELLOW->RED (BLINK)")
    print(f"   After frame 215:  Should see proj_id=1 with YELLOW (normal, no blink)")
    print(f"   This tests if blinking detection survives ID changes")
    
    return results

if __name__ == "__main__":
    results = run_id_switching_test()