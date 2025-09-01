#!/usr/bin/env python3
"""
Debug the preprocessing step after physical swap to identify recognition failure
"""
import sys
import os
import torch
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, 'src')
from tlr.pipeline import load_pipeline
from tlr.tools.utils import preprocess4rec

def debug_preprocessing():
    """
    Debug preprocessing for recognition after swap
    """
    print("🔧 Debugging preprocessing after swap...")
    
    # Load pipeline
    pipeline = load_pipeline('cuda:0')
    print("✅ Pipeline loaded")
    
    # Load frame 215
    frame_path = "test_original_video_nuevo/frame_0215.jpg"
    image = cv2.imread(frame_path)
    image_tensor = torch.tensor(image, dtype=torch.float32).cuda()
    
    # Get normal detections first
    bboxes = [[421, 165, 460, 223, 0], [466, 165, 511, 256, 1]]
    valid_detections, recognitions, assignments, _, _ = pipeline(image_tensor, bboxes, frame_ts=7.4)
    
    print("=== ORIGINAL DETECTIONS ===")
    for i, det in enumerate(valid_detections):
        x1, y1, x2, y2 = map(int, det[1:5])
        print(f"Det {i}: ({x1},{y1},{x2},{y2})")
        print(f"  Detection tensor shape: {det.shape}")
        print(f"  Detection dtype: {det.dtype}")
        print(f"  Detection device: {det.device}")
        
        # Test preprocessing manually
        det_box = det[1:5].type(torch.long)
        tl_type = int(torch.argmax(det[5:9])) + 1
        recognizer, shape = pipeline.classifiers[tl_type-1]
        
        print(f"  TL Type: {tl_type}, Shape: {shape}")
        print(f"  det_box dtype: {det_box.dtype}")
        print(f"  det_box: {det_box}")
        
        # Manual preprocessing
        try:
            input_processed = preprocess4rec(image_tensor, det_box, shape, pipeline.means_rec)
            print(f"  Preprocessing SUCCESS: {input_processed.shape}, dtype: {input_processed.dtype}")
            print(f"  Preprocessed values range: [{input_processed.min():.3f}, {input_processed.max():.3f}]")
            
            # Save preprocessed image for inspection
            input_vis = (input_processed.cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(f"debug_preprocessed_orig_{i}.jpg", input_vis)
            
        except Exception as e:
            print(f"  Preprocessing FAILED: {e}")
    
    print("\n=== AFTER PHYSICAL SWAP ===")
    # Do the swap
    if len(valid_detections) >= 2:
        temp_det = valid_detections[0].clone()
        valid_detections[0] = valid_detections[1].clone()
        valid_detections[1] = temp_det
        
        for i, det in enumerate(valid_detections):
            x1, y1, x2, y2 = map(int, det[1:5])
            print(f"Swapped Det {i}: ({x1},{y1},{x2},{y2})")
            print(f"  Detection tensor shape: {det.shape}")
            print(f"  Detection dtype: {det.dtype}")
            print(f"  Detection device: {det.device}")
            
            # Test preprocessing after swap
            det_box = det[1:5].type(torch.long)
            tl_type = int(torch.argmax(det[5:9])) + 1
            recognizer, shape = pipeline.classifiers[tl_type-1]
            
            print(f"  TL Type: {tl_type}, Shape: {shape}")
            print(f"  det_box dtype: {det_box.dtype}")
            print(f"  det_box: {det_box}")
            
            # Manual preprocessing
            try:
                input_processed = preprocess4rec(image_tensor, det_box, shape, pipeline.means_rec)
                print(f"  Preprocessing SUCCESS: {input_processed.shape}, dtype: {input_processed.dtype}")
                print(f"  Preprocessed values range: [{input_processed.min():.3f}, {input_processed.max():.3f}]")
                
                # Save preprocessed image for inspection
                input_vis = (input_processed.cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(f"debug_preprocessed_swapped_{i}.jpg", input_vis)
                
                # Try running the actual recognizer
                input_scaled = input_processed.permute(2, 0, 1).unsqueeze(0)  # NCHW format
                with torch.no_grad():
                    raw_output = recognizer(input_scaled)
                    print(f"  Recognizer raw output: {raw_output}")
                    
                    # Apply softmax to get probabilities
                    probabilities = torch.softmax(raw_output, dim=1)
                    print(f"  Probabilities: {probabilities}")
                    
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    colors = ['BLACK', 'RED', 'YELLOW', 'GREEN']
                    print(f"  Predicted: {colors[predicted_class]}")
                
            except Exception as e:
                print(f"  Preprocessing/Recognition FAILED: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    debug_preprocessing()