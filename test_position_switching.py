#!/usr/bin/env python3
"""
Test script to verify historical tracking behavior by swapping physical positions
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import cv2
import numpy as np
from modified_pipeline_for_test import create_test_pipeline
import shutil

def create_assignment_switched_test():
    """
    This test will use the original projection file but intercept assignments post-Hungarian
    to swap detection IDs in the middle of yellow blinking
    """
    
    # Use original projection file directly
    original_file = 'test_original_video_nuevo/projection_bboxes_master.txt'
    
    print("🔧 Using original projection file for assignment switching test...")
    print(f"📄 Will use: {original_file}")
    
    # Define switching point (middle of yellow blinking)
    switch_frame = 215  # Around frame_0215.jpg (middle of first yellow sequence)
    
    print(f"🔄 Will swap assignment IDs at frame {switch_frame}")
    print("   This means: detection that was assigned to proj_id=0 will get proj_id=1 and vice versa")
    
    return original_file, switch_frame

def run_assignment_switching_test():
    """
    Run the test with assignment ID switching post-Hungarian algorithm
    """
    
    # Get test setup
    bbox_file, switch_frame = create_assignment_switched_test()
    
    # Setup
    input_dir = 'test_original_video_nuevo'
    output_dir = 'test_assignment_switching_results'
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Running assignment switching test...")
    print(f"📂 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Load modified pipeline for testing
    pipeline = create_test_pipeline()
    pipeline.set_swap_frame(switch_frame)
    
    # Read projections
    entries = {}
    with open(bbox_file, 'r') as f:
        for line in f:
            parts = line.strip().split(",")
            frame = parts[0]
            bbox = list(map(int, parts[1:]))
            entries.setdefault(frame, []).append(bbox)
    
    # Process frames from beginning through first blinking sequence
    # This ensures we have proper history buildup before the swap
    test_frames = []
    for i in range(0, 226):  # frame_0000.jpg to frame_0225.jpg
        test_frames.append(f'frame_{i:04d}.jpg')
    
    print(f"📋 Processing {len(test_frames)} frames (0-225) to capture full blinking sequence")
    print(f"🔄 Swap will occur at frame 215 (middle of yellow blink sequence)")
    
    results = []
    
    for frame_idx, frame_name in enumerate(test_frames):
        if frame_name not in entries:
            continue
            
        # Only show detailed output for critical frames  
        frame_num = int(frame_name.replace('frame_', '').replace('.jpg', ''))
        show_details = frame_num in [205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225]
        
        if show_details:
            print(f"\n🔍 Processing {frame_name} (frame {frame_idx})")
        elif frame_idx % 50 == 0:
            print(f"⏳ Processing frame {frame_idx}...")
        
        # Load image
        frame_path = os.path.join(input_dir, frame_name)
        if not os.path.exists(frame_path):
            continue
            
        image_np = cv2.imread(frame_path)
        image_tensor = torch.from_numpy(image_np.astype(np.float32)).to('cuda:0')
        
        # Get bboxes
        bboxes = entries[frame_name]
        if show_details:
            print(f"  📊 Projections: {bboxes}")
        
        # Process with modified pipeline (handles swapping internally)
        frame_ts = frame_idx * (1.0/29)  # Simulate frame timing
        
        valid_detections, recognitions, assignments, invalid_detections, revised_states = pipeline(
            image_tensor, bboxes, frame_ts=frame_ts, current_frame_num=frame_num
        )
        
        if revised_states and show_details:
            print(f"  ✅ Revised states: {revised_states}")
        
        # Decode results
        if len(recognitions) > 0:
            colors = ['BLACK', 'RED', 'YELLOW', 'GREEN']
            assign_map = {int(assignment[1]): int(assignment[0]) for assignment in assignments}
            
            for det_idx, recognition in enumerate(recognitions):
                color_idx = torch.argmax(recognition).item()
                confidence = recognition[color_idx].item()
                proj_id = assign_map.get(det_idx, -999)
                
                # Get revised state
                revised_info = ""
                if revised_states and proj_id in revised_states:
                    revised_color, is_blinking = revised_states[proj_id]
                    revised_info = f" -> {revised_color.upper()}" + (" (BLINK)" if is_blinking else "")
                
                if show_details:
                    print(f"    🚦 Det {det_idx} -> Proj {proj_id}: {colors[color_idx]}{revised_info} (conf: {confidence:.3f})")
                
                results.append({
                    'frame': frame_name,
                    'frame_idx': frame_idx,
                    'det_idx': det_idx,
                    'proj_id': proj_id,
                    'detected_color': colors[color_idx],
                    'confidence': confidence,
                    'revised_color': revised_states[proj_id][0] if revised_states and proj_id in revised_states else colors[color_idx],
                    'is_blinking': revised_states[proj_id][1] if revised_states and proj_id in revised_states else False,
                    'frame_num': frame_num,
                    'post_switch': frame_num >= switch_frame
                })
    
    # Save results
    import csv
    with open(f'{output_dir}/assignment_switching_results.csv', 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\n📄 Results saved to {output_dir}/assignment_switching_results.csv")
    
    # Analysis
    print(f"\n🔍 ANALYSIS:")
    print(f"   Test concept:")
    print(f"   - Before frame {switch_frame}: Normal assignment (Det0->Proj0, Det1->Proj1)")
    print(f"   - After frame {switch_frame}:  Swapped assignment (Det0->Proj1, Det1->Proj0)")
    print(f"   - Key question: Does the BLINK history stay with the detection or follow the proj_id?")
    print(f"   - Expected: If history follows proj_id, blinking should transfer to different detection")
    
    return results

if __name__ == "__main__":
    results = run_assignment_switching_test()