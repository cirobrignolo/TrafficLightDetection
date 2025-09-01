#!/usr/bin/env python3
"""
Modified pipeline for detection swapping test
Based on tlr.pipeline but with interception capability
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from tlr.pipeline import load_pipeline
from tlr.tools.utils import boxes2projections
from tlr.selector import select_tls

class ModifiedPipelineForTest:
    """
    Wrapper around normal pipeline that allows detection swapping
    """
    def __init__(self, original_pipeline):
        self.pipeline = original_pipeline
        self.swap_frame = None
        
    def set_swap_frame(self, frame_num):
        """Set the frame number where detection swapping should start"""
        self.swap_frame = frame_num
        
    def __call__(self, image_tensor, bboxes, frame_ts=None, current_frame_num=None):
        """
        Modified pipeline call that can swap detections at specified frame
        """
        # Step 1: Detection (normal)
        detections = self.pipeline.detect(image_tensor, bboxes)
        valid_detections = [det for det in detections if len(det) > 0]
        
        print(f"  📍 Detected: {len(valid_detections)} traffic lights")
        if len(valid_detections) >= 2:
            for i, det in enumerate(valid_detections[:2]):
                x1, y1, x2, y2 = map(int, det[1:5])
                print(f"     Det {i}: bbox=({x1},{y1},{x2},{y2})")
        
        # Step 2: INTERCEPTION - Swap detections if needed
        if (self.swap_frame is not None and 
            current_frame_num is not None and 
            current_frame_num >= self.swap_frame and 
            len(valid_detections) >= 2):
            
            print(f"  🔄 SWAPPING detections at frame {current_frame_num}")
            print(f"     Before swap: Det0=({int(valid_detections[0][1])},{int(valid_detections[0][2])},{int(valid_detections[0][3])},{int(valid_detections[0][4])})")
            print(f"                   Det1=({int(valid_detections[1][1])},{int(valid_detections[1][2])},{int(valid_detections[1][3])},{int(valid_detections[1][4])})")
            
            # Swap the first two detections
            valid_detections[0], valid_detections[1] = valid_detections[1], valid_detections[0]
            
            print(f"     After swap:  Det0=({int(valid_detections[0][1])},{int(valid_detections[0][2])},{int(valid_detections[0][3])},{int(valid_detections[0][4])})")
            print(f"                   Det1=({int(valid_detections[1][1])},{int(valid_detections[1][2])},{int(valid_detections[1][3])},{int(valid_detections[1][4])})")
        
        # Step 3: Recognition (on potentially swapped detections)
        recognitions = []
        if len(valid_detections) > 0:
            # Get tl_types from detection data (columns 5-8)
            tl_types = [int(torch.argmax(det[5:9])) + 1 for det in valid_detections]  # +1 because types are 1-indexed
            recognitions = self.pipeline.recognize(image_tensor, valid_detections, tl_types)
        
        # Step 4: Selection/Assignment
        assignments = torch.empty([0, 2])
        invalid_detections = []
        
        if len(valid_detections) > 0 and len(bboxes) > 0:
            # Convert bboxes to projections
            projections = boxes2projections(bboxes)
            
            # Run selector (Hungarian algorithm)
            assignments = select_tls(
                self.pipeline.ho,  # Hungarian optimizer
                valid_detections,
                projections,
                image_tensor.shape[:2]
            )
            
            # Separate valid/invalid based on assignments
            assigned_det_indices = set(int(a[1]) for a in assignments) if len(assignments) > 0 else set()
            valid_final = [det for i, det in enumerate(valid_detections) if i in assigned_det_indices]
            invalid_detections = [det for i, det in enumerate(valid_detections) if i not in assigned_det_indices]
            valid_detections = valid_final
        
        # Step 5: Tracking
        revised_states = {}
        if len(assignments) > 0 and frame_ts is not None:
            # Convert assignments to list of tuples
            assignment_list = [(int(a[0]), int(a[1])) for a in assignments]
            revised_states = self.pipeline.tracker.track(frame_ts, assignment_list, recognitions)
        
        return valid_detections, recognitions, assignments, invalid_detections, revised_states


def create_test_pipeline():
    """Create a modified pipeline for testing"""
    original_pipeline = load_pipeline('cuda:0')
    return ModifiedPipelineForTest(original_pipeline)