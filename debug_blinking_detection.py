#!/usr/bin/env python3
"""
Debug script to understand why blinking is not being detected
Uses original pipeline + visual outputs
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import cv2
import numpy as np
from tlr.pipeline import load_pipeline
import time

def debug_blinking_detection():
    """
    Debug blinking detection using original pipeline with visual outputs
    """
    
    # Setup
    input_dir = 'test_original_video_nuevo'
    output_dir = 'debug_blinking_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for visual outputs
    detection_dir = os.path.join(output_dir, 'detections')
    recognition_dir = os.path.join(output_dir, 'recognitions')
    final_dir = os.path.join(output_dir, 'final')
    
    for dir_path in [detection_dir, recognition_dir, final_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    print("🚀 Running blinking detection debug...")
    print(f"📂 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Load ORIGINAL pipeline (no wrapper)
    pipeline = load_pipeline('cuda:0')
    print("✅ Loaded original pipeline")
    
    # Read projections
    bbox_file = os.path.join(input_dir, 'projection_bboxes_master.txt')
    entries = {}
    with open(bbox_file, 'r') as f:
        for line in f:
            parts = line.strip().split(",")
            frame = parts[0]
            bbox = list(map(int, parts[1:]))
            entries.setdefault(frame, []).append(bbox)
    
    # Focus on critical frames where blinking should occur
    critical_frames = [
        'frame_0205.jpg', 'frame_0206.jpg', 'frame_0207.jpg', 'frame_0208.jpg', 'frame_0209.jpg',
        'frame_0210.jpg', 'frame_0211.jpg', 'frame_0212.jpg', 'frame_0213.jpg', 'frame_0214.jpg',
        'frame_0215.jpg', 'frame_0216.jpg', 'frame_0217.jpg', 'frame_0218.jpg', 'frame_0219.jpg',
        'frame_0220.jpg', 'frame_0221.jpg', 'frame_0222.jpg', 'frame_0223.jpg', 'frame_0224.jpg', 'frame_0225.jpg'
    ]
    
    results = []
    
    print(f"\n📋 Processing {len(critical_frames)} critical frames")
    
    for frame_idx, frame_name in enumerate(critical_frames):
        if frame_name not in entries:
            continue
            
        frame_num = int(frame_name.replace('frame_', '').replace('.jpg', ''))
        print(f"\n🔍 Processing {frame_name} (frame_num={frame_num})")
        
        # Load image
        frame_path = os.path.join(input_dir, frame_name)
        if not os.path.exists(frame_path):
            print(f"   ❌ Image not found: {frame_path}")
            continue
            
        image_np = cv2.imread(frame_path)
        image_tensor = torch.from_numpy(image_np.astype(np.float32)).to('cuda:0')
        
        # Get bboxes
        bboxes = entries[frame_name]
        print(f"  📊 Projections: {bboxes}")
        
        # Process with ORIGINAL pipeline
        frame_ts = time.time()  # Use real timestamp like original
        valid_detections, recognitions, assignments, invalid_detections, revised_states = pipeline(
            image_tensor, bboxes, frame_ts=frame_ts
        )
        
        # === SWAPPING LOGIC ===
        # Apply detection swap at frame 215 (middle of blinking sequence)
        swap_frame = 215
        swapped = False
        
        if frame_num >= swap_frame and len(valid_detections) >= 2:
            print(f"  🔄 SWAPPING detections at frame {frame_num}")
            print(f"     Before swap: Det0=({int(valid_detections[0][1])},{int(valid_detections[0][2])},{int(valid_detections[0][3])},{int(valid_detections[0][4])})")
            print(f"                   Det1=({int(valid_detections[1][1])},{int(valid_detections[1][2])},{int(valid_detections[1][3])},{int(valid_detections[1][4])})")
            
            # Swap assignments manually (not detections, to preserve recognition results)
            if len(assignments) >= 2:
                # Store original assignments
                original_assign_map = {int(assignment[1]): int(assignment[0]) for assignment in assignments}
                
                # Swap assignment proj_ids
                new_assignments = assignments.clone()
                if len(assignments) >= 2:
                    new_assignments[0][0], new_assignments[1][0] = assignments[1][0], assignments[0][0]
                
                # Update assign_map for results
                assignments = new_assignments
                print(f"     Swapped assignments: {assignments.tolist()}")
                
                # Rebuild revised_states with swapped proj_ids
                if revised_states:
                    new_revised_states = {}
                    # Map each detection to its new proj_id
                    for det_idx in range(len(recognitions)):
                        if det_idx < len(assignments):
                            new_proj_id = int(assignments[det_idx][0])  # proj_id from assignment
                            # Find which proj_id this detection had before
                            old_proj_id = original_assign_map.get(det_idx, -1)
                            if old_proj_id in revised_states:
                                new_revised_states[new_proj_id] = revised_states[old_proj_id]
                                print(f"     Moved revised_state from proj {old_proj_id} to proj {new_proj_id}: {revised_states[old_proj_id]}")
                    
                    revised_states = new_revised_states
                
                swapped = True
            
            print(f"     After swap assignments: {assignments.tolist()}")
        
        print(f"  ✅ Valid detections: {len(valid_detections)}")
        print(f"  ✅ Assignments: {assignments.tolist() if len(assignments) > 0 else []}")
        if revised_states:
            print(f"  ✅ Revised states: {revised_states}")
        
        # === VISUAL OUTPUT 1: DETECTIONS ===
        img_detections = image_np.copy()
        for det_idx, det in enumerate(valid_detections):
            x1, y1, x2, y2 = map(int, det[1:5])
            cv2.rectangle(img_detections, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_detections, f'Det{det_idx}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw projections
        for proj_idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox[:4]
            cv2.rectangle(img_detections, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.putText(img_detections, f'Proj{bbox[4]}', (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        cv2.imwrite(os.path.join(detection_dir, frame_name), img_detections)
        
        # === VISUAL OUTPUT 2: RECOGNITIONS ===
        img_recognition = image_np.copy()
        colors = ['BLACK', 'RED', 'YELLOW', 'GREEN']
        color_bgr = [(0, 0, 0), (0, 0, 255), (0, 255, 255), (0, 255, 0)]
        
        for det_idx, (det, recognition) in enumerate(zip(valid_detections, recognitions)):
            x1, y1, x2, y2 = map(int, det[1:5])
            color_idx = torch.argmax(recognition).item()
            confidence = recognition[color_idx].item()
            color_name = colors[color_idx]
            
            cv2.rectangle(img_recognition, (x1, y1), (x2, y2), color_bgr[color_idx], 2)
            cv2.putText(img_recognition, f'{color_name}', (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr[color_idx], 2)
            cv2.putText(img_recognition, f'{confidence:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr[color_idx], 1)
        
        cv2.imwrite(os.path.join(recognition_dir, frame_name), img_recognition)
        
        # === VISUAL OUTPUT 3: FINAL WITH ASSIGNMENTS ===
        img_final = image_np.copy()
        
        if len(recognitions) > 0:
            assign_map = {int(assignment[1]): int(assignment[0]) for assignment in assignments}
            
            for det_idx, recognition in enumerate(recognitions):
                if det_idx >= len(valid_detections):
                    continue
                    
                det = valid_detections[det_idx]
                x1, y1, x2, y2 = map(int, det[1:5])
                color_idx = torch.argmax(recognition).item()
                confidence = recognition[color_idx].item()
                proj_id = assign_map.get(det_idx, -999)
                
                # Get revised state
                revised_info = ""
                if revised_states and proj_id in revised_states:
                    revised_color, is_blinking = revised_states[proj_id]
                    revised_info = f" -> {revised_color.upper()}"
                    if is_blinking:
                        revised_info += " (BLINK)"
                        # Special visual for blinking - thick red border
                        cv2.rectangle(img_final, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 255), 5)
                
                # Draw detection box
                cv2.rectangle(img_final, (x1, y1), (x2, y2), color_bgr[color_idx], 2)
                
                # Text labels
                cv2.putText(img_final, f'Det{det_idx}->Proj{proj_id}', (x1, y1-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(img_final, f'{colors[color_idx]}{revised_info}', (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr[color_idx], 2)
                cv2.putText(img_final, f'{confidence:.3f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                print(f"    🚦 Det {det_idx} -> Proj {proj_id}: {colors[color_idx]}{revised_info} (conf: {confidence:.3f})")
                
                # Store results
                results.append({
                    'frame': frame_name,
                    'frame_num': frame_num,
                    'det_idx': det_idx,
                    'proj_id': proj_id,
                    'detected_color': colors[color_idx],
                    'confidence': confidence,
                    'revised_color': revised_states[proj_id][0] if revised_states and proj_id in revised_states else colors[color_idx],
                    'is_blinking': revised_states[proj_id][1] if revised_states and proj_id in revised_states else False,
                    'post_swap': frame_num >= swap_frame
                })
        
        cv2.imwrite(os.path.join(final_dir, frame_name), img_final)
    
    # Save results to CSV
    import csv
    with open(f'{output_dir}/debug_results.csv', 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\n📄 Results saved to {output_dir}/debug_results.csv")
    print(f"🖼️  Visual outputs saved to:")
    print(f"   📁 Detections: {detection_dir}")
    print(f"   📁 Recognitions: {recognition_dir}")
    print(f"   📁 Final: {final_dir}")
    
    # Analysis
    blinking_frames = [r for r in results if r['is_blinking']]
    yellow_frames = [r for r in results if r['detected_color'] == 'YELLOW']
    
    print(f"\n🔍 ANALYSIS:")
    print(f"   Total frames processed: {len(results)//2 if results else 0}")  # Divided by 2 because 2 detections per frame
    print(f"   Frames with YELLOW detection: {len(yellow_frames)}")
    print(f"   Frames with BLINKING: {len(blinking_frames)}")
    
    if blinking_frames:
        print(f"   Blinking detected in frames: {sorted(set(r['frame_num'] for r in blinking_frames))}")
    
    return results

if __name__ == "__main__":
    results = debug_blinking_detection()