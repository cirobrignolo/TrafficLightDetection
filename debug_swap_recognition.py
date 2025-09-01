#!/usr/bin/env python3
"""
Debug recognition after physical swap to understand why both detections become BLACK
"""
import sys
import os
import torch
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, 'src')
from tlr.pipeline import load_pipeline

def debug_swap_recognition():
    """
    Test frame 215 specifically to debug recognition issue
    """
    print("🔧 Debugging swap recognition...")
    
    # Load pipeline
    pipeline = load_pipeline('cuda:0')
    print("✅ Pipeline loaded")
    
    # Load frame 215
    frame_path = "test_original_video_nuevo/frame_0215.jpg"
    if not os.path.exists(frame_path):
        print(f"❌ Frame not found: {frame_path}")
        return
    
    image = cv2.imread(frame_path)
    image_tensor = torch.tensor(image, dtype=torch.float32).cuda()
    
    # Projections
    bboxes = [[421, 165, 460, 223, 0], [466, 165, 511, 256, 1]]
    
    print(f"📸 Processing {frame_path}")
    print(f"📊 Projections: {bboxes}")
    
    # Run normal pipeline first
    print("\n=== NORMAL PIPELINE ===")
    valid_detections, recognitions, assignments, _, _ = pipeline(image_tensor, bboxes, frame_ts=7.4)
    
    print(f"Detections: {len(valid_detections)}")
    for i, det in enumerate(valid_detections):
        x1, y1, x2, y2 = map(int, det[1:5])
        tl_type_scores = det[5:9]
        tl_type = int(torch.argmax(tl_type_scores))
        print(f"  Det {i}: ({x1},{y1},{x2},{y2}) type_scores={tl_type_scores.tolist()} type={tl_type}")
    
    print(f"Recognitions: {len(recognitions)}")
    colors = ['BLACK', 'RED', 'YELLOW', 'GREEN']
    for i, rec in enumerate(recognitions):
        color = colors[torch.argmax(rec)]
        print(f"  Rec {i}: {color} scores={rec.tolist()}")
    
    print(f"Assignments: {assignments.tolist()}")
    
    # Now do the swap
    print("\n=== PHYSICAL SWAP ===")
    if len(valid_detections) >= 2:
        print("Before swap:")
        for i, det in enumerate(valid_detections):
            x1, y1, x2, y2 = map(int, det[1:5])
            tl_type_scores = det[5:9]
            print(f"  Det {i}: ({x1},{y1},{x2},{y2}) type_scores={tl_type_scores.tolist()}")
        
        # SWAP
        temp_det = valid_detections[0].clone()
        valid_detections[0] = valid_detections[1].clone()
        valid_detections[1] = temp_det
        
        print("After swap:")
        for i, det in enumerate(valid_detections):
            x1, y1, x2, y2 = map(int, det[1:5])
            tl_type_scores = det[5:9]
            print(f"  Det {i}: ({x1},{y1},{x2},{y2}) type_scores={tl_type_scores.tolist()}")
        
        # Re-recognize with swapped detections
        print("\n=== RE-RECOGNITION WITH SWAPPED COORDINATES ===")
        tl_types = [int(torch.argmax(det[5:9])) + 1 for det in valid_detections]
        print(f"TL Types from swapped detections: {tl_types}")
        
        recognitions_swapped = pipeline.recognize(image_tensor, valid_detections, tl_types)
        
        print(f"Swapped Recognitions: {len(recognitions_swapped)}")
        for i, rec in enumerate(recognitions_swapped):
            color = colors[torch.argmax(rec)]
            print(f"  Swapped Rec {i}: {color} scores={rec.tolist()}")
        
        # Test with original tl_types
        print("\n=== TEST: RE-RECOGNITION WITH ORIGINAL TL_TYPES ===")
        original_tl_types = [int(torch.argmax(det[5:9])) + 1 for det in [temp_det, valid_detections[0]]]  # Reverse order
        print(f"Original TL Types (pre-swap order): {original_tl_types}")
        
        recognitions_orig_types = pipeline.recognize(image_tensor, valid_detections, original_tl_types)
        
        print(f"Recognition with original types: {len(recognitions_orig_types)}")
        for i, rec in enumerate(recognitions_orig_types):
            color = colors[torch.argmax(rec)]
            print(f"  Orig-type Rec {i}: {color} scores={rec.tolist()}")
            
        # Test with manual extraction from image
        print("\n=== MANUAL IMAGE EXTRACTION TEST ===")
        for i, det in enumerate(valid_detections):
            x1, y1, x2, y2 = map(int, det[1:5])
            roi = image[y1:y2, x1:x2]
            avg_color = np.mean(roi, axis=(0,1))
            print(f"  Det {i} at ({x1},{y1},{x2},{y2}): avg_BGR={avg_color}")
            
            # Save ROI for visual inspection
            cv2.imwrite(f"debug_det_{i}_roi_swapped.jpg", roi)
        
        # Compare with original positions before swap
        print("\n=== ORIGINAL POSITIONS CHECK ===")
        # Original det0 position (should be right/yellow)
        x1, y1, x2, y2 = 476, 175, 501, 247
        roi_orig_0 = image[y1:y2, x1:x2]
        avg_color_orig_0 = np.mean(roi_orig_0, axis=(0,1))
        print(f"  Original Det0 pos (right): ({x1},{y1},{x2},{y2}) avg_BGR={avg_color_orig_0}")
        cv2.imwrite(f"debug_original_det0_roi.jpg", roi_orig_0)
        
        # Original det1 position (should be left/green)  
        x1, y1, x2, y2 = 432, 176, 452, 212
        roi_orig_1 = image[y1:y2, x1:x2]
        avg_color_orig_1 = np.mean(roi_orig_1, axis=(0,1))
        print(f"  Original Det1 pos (left): ({x1},{y1},{x2},{y2}) avg_BGR={avg_color_orig_1}")
        cv2.imwrite(f"debug_original_det1_roi.jpg", roi_orig_1)

if __name__ == "__main__":
    debug_swap_recognition()