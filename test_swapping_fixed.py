#!/usr/bin/env python3
"""
Fixed swapping test with real timestamps and proper blinking detection
Based on test_position_switching.py but using conditions that work
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import cv2
import numpy as np
from tlr.pipeline import load_pipeline
import time

def run_fixed_swapping_test():
    """
    Run the swapping test with real timestamps and original pipeline to ensure blinking detection
    """
    
    # Use original projection file
    bbox_file = 'test_original_video_nuevo/projection_bboxes_master.txt'
    
    print("🔧 Running FIXED swapping test...")
    print(f"📄 Using: {bbox_file}")
    
    # Define switching point (middle of yellow blinking)
    switch_frame = 215  # Around frame_0215.jpg (middle of first yellow sequence)
    
    print(f"🔄 Will swap assignments at frame {switch_frame}")
    print("   This swaps which detection gets assigned to which proj_id")
    
    # Setup
    input_dir = 'test_original_video_nuevo'
    output_dir = 'test_swapping_fixed_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create organized folder structure like run_pipeline_debug
    detection_dir = os.path.join(output_dir, '1_detection')
    recognition_dir = os.path.join(output_dir, '2_recognition')
    final_dir = os.path.join(output_dir, '3_final')
    
    # Create directories
    for stage_dir in [detection_dir, recognition_dir, final_dir]:
        os.makedirs(stage_dir, exist_ok=True)
    
    # CSV files in main output directory
    detection_csv = os.path.join(output_dir, '1_detection_results.csv')
    recognition_csv = os.path.join(output_dir, '2_recognition_results.csv')
    final_csv = os.path.join(output_dir, '3_final_results.csv')
    
    # Clean start: remove existing files
    for csv_file in [detection_csv, recognition_csv, final_csv]:
        if os.path.exists(csv_file):
            os.remove(csv_file)
    
    # Write headers for each file
    with open(detection_csv, 'w') as out:
        out.write("frame,det_idx,x1,y1,x2,y2,tl_type,det_vert,det_quad,det_hori,det_bg,post_swap\n")
    
    with open(recognition_csv, 'w') as out:
        out.write("frame,det_idx,x1,y1,x2,y2,tl_type,pred_color,rec_black,rec_red,rec_yellow,rec_green,post_swap\n")
    
    with open(final_csv, 'w') as out:
        out.write("frame,det_idx,proj_id,x1,y1,x2,y2,tl_type,pred_color,revised_color,blink,det_vert,det_quad,det_hori,det_bg,rec_black,rec_red,rec_yellow,rec_green,post_swap\n")
    
    print("🚀 Running fixed swapping test...")
    print(f"📂 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Load ORIGINAL pipeline (no wrapper)
    pipeline = load_pipeline('cuda:0')
    print("✅ Loaded original pipeline")
    
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
    print(f"🔄 Swap will occur at frame {switch_frame} (middle of yellow blink sequence)")
    
    results = []
    
    # Track real time progression for proper timestamps
    start_time = time.time()
    
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
        
        # Process with ORIGINAL pipeline using REAL timestamps
        # This maintains proper temporal continuity for tracking
        frame_ts = start_time + (frame_idx * (1.0/29))  # Real progression at 29fps
        
        valid_detections, recognitions, assignments, invalid_detections, revised_states = pipeline(
            image_tensor, bboxes, frame_ts=frame_ts
        )
        
        # === SWAPPING LOGIC ===
        # Apply assignment swap at frame 215 (middle of blinking sequence)
        original_assignments = assignments.clone() if len(assignments) > 0 else assignments
        original_revised_states = revised_states.copy() if revised_states else {}
        
        if frame_num >= switch_frame and len(assignments) >= 2:
            if show_details:
                print(f"  🔄 SWAPPING assignments at frame {frame_num}")
                print(f"     Before swap: {assignments.tolist()}")
            
            # Swap assignment proj_ids (keep detection indices the same)
            new_assignments = assignments.clone()
            new_assignments[0][0], new_assignments[1][0] = assignments[1][0], assignments[0][0]
            assignments = new_assignments
            
            if show_details:
                print(f"     After swap:  {assignments.tolist()}")
            
            # Rebuild revised_states with swapped proj_ids
            if revised_states and len(assignments) >= 2:
                new_revised_states = {}
                
                # Map each detection to its new proj_id, but keep the historical state
                for det_idx in range(min(len(recognitions), len(assignments))):
                    new_proj_id = int(assignments[det_idx][0])  # New proj_id assignment
                    old_proj_id = int(original_assignments[det_idx][0])  # Original proj_id
                    
                    # Transfer the revised state from the old proj_id to the new one
                    if old_proj_id in original_revised_states:
                        new_revised_states[new_proj_id] = original_revised_states[old_proj_id]
                        if show_details:
                            print(f"     Transferred state: proj {old_proj_id} -> proj {new_proj_id}: {original_revised_states[old_proj_id]}")
                
                revised_states = new_revised_states
        
        if revised_states and show_details:
            print(f"  ✅ Revised states: {revised_states}")
        
        # === STAGE OUTPUT GENERATION (like run_pipeline_debug) ===
        
        # Construct assignment map
        assign_list = assignments.cpu().tolist() if len(assignments) > 0 else []
        assign_map = {det_idx: proj_id for proj_id, det_idx in assign_list}
        
        # ═══ ETAPA 1: DETECCIÓN CRUDA ═══
        img_detection = image_np.copy()
        detection_lines = []
        
        for det_idx, det in enumerate(valid_detections):
            x1, y1, x2, y2 = map(int, det[1:5])
            det_scores = det[5:9].tolist()
            tl_type = int(torch.argmax(det[5:9]))
            type_names = ['vert', 'quad', 'hori', 'bg']
            
            detection_lines.append(
                f"{frame_name},{det_idx},{x1},{y1},{x2},{y2},"
                f"{type_names[tl_type]},"
                f"{det_scores[0]:.4f},{det_scores[1]:.4f},{det_scores[2]:.4f},{det_scores[3]:.4f},"
                f"{frame_num >= switch_frame}\n"
            )
            
            # Visualizar detección
            color = (0, 255, 0) if frame_num < switch_frame else (255, 0, 255)  # Verde antes, magenta después del swap
            cv2.rectangle(img_detection, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img_detection, f'Det{det_idx}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        # Draw projections
        for bbox in bboxes:
            x1, y1, x2, y2, proj_id = bbox
            cv2.rectangle(img_detection, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.putText(img_detection, f'Proj{proj_id}', (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
        # Add swap indicator
        if frame_num >= switch_frame:
            cv2.putText(img_detection, 'POST-SWAP', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            
        cv2.imwrite(os.path.join(detection_dir, frame_name), img_detection)
        
        if detection_lines:
            with open(detection_csv, 'a') as out:
                out.writelines(detection_lines)
        
        # ─── ETAPA 2: RECONOCIMIENTO ────────────────────────────────────────────────
        img_recognition = image_np.copy()
        recognition_lines = []
        
        for det_idx, det in enumerate(valid_detections):
            x1, y1, x2, y2 = map(int, det[1:5])
            if det_idx < len(recognitions):
                pred_cls = int(torch.argmax(recognitions[det_idx]))
                pred_color = ['black','red','yellow','green'][pred_cls]
                tl_type = int(torch.argmax(det[5:9]))
                type_names = ['vert', 'quad', 'hori', 'bg']
                rec_scores = recognitions[det_idx].tolist()
                
                recognition_lines.append(
                    f"{frame_name},{det_idx},{x1},{y1},{x2},{y2},"
                    f"{type_names[tl_type]},{pred_color},"
                    f"{rec_scores[0]:.4f},{rec_scores[1]:.4f},{rec_scores[2]:.4f},{rec_scores[3]:.4f},"
                    f"{frame_num >= switch_frame}\n"
                )
                
                # Visualizar reconocimiento
                color_map = {'black': (0,0,0), 'red': (0,0,255), 'yellow': (0,255,255), 'green': (0,255,0)}
                color = color_map.get(pred_color, (255,255,255))
                cv2.rectangle(img_recognition, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img_recognition, pred_color, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add swap indicator
        if frame_num >= switch_frame:
            cv2.putText(img_recognition, 'POST-SWAP', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            
        cv2.imwrite(os.path.join(recognition_dir, frame_name), img_recognition)
        
        if recognition_lines:
            with open(recognition_csv, 'a') as out:
                out.writelines(recognition_lines)
        
        # ─── ETAPA 3: RESULTADO FINAL CON TRACKING ─────────────────────────────────
        img_final = image_np.copy()
        final_lines = []
        
        # Decode results
        if len(recognitions) > 0:
            colors = ['BLACK', 'RED', 'YELLOW', 'GREEN']
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
                revised_color = colors[color_idx]
                is_blinking = False
                revised_info = ""
                
                if revised_states and proj_id in revised_states:
                    revised_color, is_blinking = revised_states[proj_id]
                    revised_info = f" -> {revised_color.upper()}" + (" (BLINK)" if is_blinking else "")
                
                if show_details:
                    print(f"    🚦 Det {det_idx} -> Proj {proj_id}: {colors[color_idx]}{revised_info} (conf: {confidence:.3f})")
                
                # Create final CSV line
                tl_type = int(torch.argmax(det[5:9]))
                type_names = ['vert', 'quad', 'hori', 'bg']
                det_scores = det[5:9].tolist()
                rec_scores = recognition.tolist()
                
                final_lines.append(
                    f"{frame_name},{det_idx},{proj_id},{x1},{y1},{x2},{y2},"
                    f"{type_names[tl_type]},{colors[color_idx]},{revised_color},"
                    f"{'(BLINK)' if is_blinking else ''},"
                    f"{det_scores[0]:.4f},{det_scores[1]:.4f},{det_scores[2]:.4f},{det_scores[3]:.4f},"
                    f"{rec_scores[0]:.4f},{rec_scores[1]:.4f},{rec_scores[2]:.4f},{rec_scores[3]:.4f},"
                    f"{frame_num >= switch_frame}\n"
                )
                
                # Visualizar resultado final
                color_map = {'BLACK': (0,0,0), 'RED': (0,0,255), 'YELLOW': (0,255,255), 'GREEN': (0,255,0)}
                vis_color = color_map.get(revised_color.upper(), (255,255,255))
                
                # Special handling for blinking - thick red border
                if is_blinking:
                    cv2.rectangle(img_final, (x1-3, y1-3), (x2+3, y2+3), (0,0,255), 5)
                
                cv2.rectangle(img_final, (x1, y1), (x2, y2), vis_color, 2)
                cv2.putText(img_final, f'Det{det_idx}->Proj{proj_id}', (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.putText(img_final, f'{revised_color.upper()}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, vis_color, 2)
                if is_blinking:
                    cv2.putText(img_final, 'BLINK', (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                
                results.append({
                    'frame': frame_name,
                    'frame_idx': frame_idx,
                    'frame_num': frame_num,
                    'det_idx': det_idx,
                    'proj_id': proj_id,
                    'detected_color': colors[color_idx],
                    'confidence': confidence,
                    'revised_color': revised_color,
                    'is_blinking': is_blinking,
                    'post_swap': frame_num >= switch_frame
                })
        
        # Add swap indicator to final image
        if frame_num >= switch_frame:
            cv2.putText(img_final, 'POST-SWAP', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            
        cv2.imwrite(os.path.join(final_dir, frame_name), img_final)
        
        if final_lines:
            with open(final_csv, 'a') as out:
                out.writelines(final_lines)
    
    # Save results
    import csv
    with open(f'{output_dir}/swapping_fixed_results.csv', 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\n📄 Results saved to {output_dir}/swapping_fixed_results.csv")
    print(f"\n🖼️  Visual outputs saved to:")
    print(f"   📁 Detections: {detection_dir}")
    print(f"   📁 Recognitions: {recognition_dir}")
    print(f"   📁 Final results: {final_dir}")
    print(f"\n📊 Stage CSV files:")
    print(f"   📄 Detection results: {detection_csv}")
    print(f"   📄 Recognition results: {recognition_csv}")
    print(f"   📄 Final results: {final_csv}")
    
    # Analysis
    blinking_frames = [r for r in results if r['is_blinking']]
    yellow_frames = [r for r in results if r['detected_color'] == 'YELLOW']
    pre_swap_blink = [r for r in blinking_frames if not r['post_swap']]
    post_swap_blink = [r for r in blinking_frames if r['post_swap']]
    
    print(f"\n🔍 ANALYSIS:")
    print(f"   Total results: {len(results)}")
    print(f"   Frames with YELLOW detection: {len(yellow_frames)}")
    print(f"   Total BLINKING detections: {len(blinking_frames)}")
    print(f"   BLINKING before swap (< frame {switch_frame}): {len(pre_swap_blink)}")
    print(f"   BLINKING after swap (≥ frame {switch_frame}): {len(post_swap_blink)}")
    
    if blinking_frames:
        blink_frames_nums = sorted(set(r['frame_num'] for r in blinking_frames))
        print(f"   Blinking detected in frames: {blink_frames_nums}")
        
        # Check for blinking transfer
        if pre_swap_blink and post_swap_blink:
            pre_proj_ids = set(r['proj_id'] for r in pre_swap_blink)
            post_proj_ids = set(r['proj_id'] for r in post_swap_blink)
            print(f"   Pre-swap blinking proj_ids: {pre_proj_ids}")
            print(f"   Post-swap blinking proj_ids: {post_proj_ids}")
            
            if pre_proj_ids != post_proj_ids:
                print(f"   ✅ BLINKING TRANSFERRED between proj_ids after swap!")
            else:
                print(f"   ❓ Blinking stayed with same proj_ids")
    
    # Expected behavior summary
    print(f"\n📋 EXPECTED BEHAVIOR:")
    print(f"   - Before frame {switch_frame}: Normal assignment, proj_id=1 should have YELLOW->RED (BLINK)")
    print(f"   - After frame {switch_frame}:  Swapped assignment, check if blinking transfers")
    print(f"   - Key test: Does BLINK history follow the proj_id or stay with physical detection?")
    
    return results

if __name__ == "__main__":
    results = run_fixed_swapping_test()