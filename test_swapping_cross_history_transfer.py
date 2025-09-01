#!/usr/bin/env python3
"""
Physical detection swapping test WITH CROSS history transfer
This test physically swaps the detection coordinates after detection but before recognition,
then cross-transfers the tracking histories so each physical location gets the other's history.

Expected behavior:
- Right TL (yellow) gets left TL's green history → should see YELLOW without blink
- Left TL (green) gets right TL's blink history → should see GREEN (blink stops naturally)
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import cv2
import numpy as np
from tlr.pipeline import load_pipeline
import time

def run_swapping_cross_history_transfer():
    """
    Run swapping test where IDs change but history stays with the physical detection
    """
    
    # Use original projection file
    bbox_file = 'test_original_video_nuevo/projection_bboxes_master.txt'
    
    print("🔧 Running swapping test WITH CROSS history transfer...")
    print(f"📄 Using: {bbox_file}")
    
    # Define switching point (middle of yellow blinking)
    switch_frame = 215  # Around frame_0215.jpg (middle of first yellow sequence)
    
    print(f"🔄 Will swap assignments AND histories at frame {switch_frame}")
    print("   Each traffic light will receive the OTHER traffic light's history")
    
    # Setup
    input_dir = 'test_original_video_nuevo'
    output_dir = 'test_swapping_cross_history_transfer_results'
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
    
    print("🚀 Running swapping test without history transfer...")
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
        
        # Process with ORIGINAL pipeline using REAL timestamps - BUT INTERCEPT BEFORE RECOGNITION
        frame_ts = start_time + (frame_idx * (1.0/29))  # Real progression at 29fps
        
        # Process with ORIGINAL pipeline normally FIRST to get baseline
        valid_detections, recognitions, assignments, _, revised_states = pipeline(
            image_tensor, bboxes, frame_ts=frame_ts
        )
        
        # === PHYSICAL SWAP LOGIC ===
        # If we're at or after the swap frame AND we have 2+ detections, swap them
        if frame_num >= switch_frame and len(valid_detections) >= 2:
            if show_details:
                print(f"  🔄 PHYSICALLY SWAPPING detections at frame {frame_num}")
                print(f"     Before swap: Det0=({int(valid_detections[0][1])},{int(valid_detections[0][2])},{int(valid_detections[0][3])},{int(valid_detections[0][4])})")
                print(f"                   Det1=({int(valid_detections[1][1])},{int(valid_detections[1][2])},{int(valid_detections[1][3])},{int(valid_detections[1][4])})")
            
            # SWAP: Physically exchange the complete detection tensors (using clone to avoid shared references)
            temp_det = valid_detections[0].clone()
            valid_detections[0] = valid_detections[1].clone()
            valid_detections[1] = temp_det
            
            if show_details:
                print(f"     After swap:  Det0=({int(valid_detections[0][1])},{int(valid_detections[0][2])},{int(valid_detections[0][3])},{int(valid_detections[0][4])})")
                print(f"                   Det1=({int(valid_detections[1][1])},{int(valid_detections[1][2])},{int(valid_detections[1][3])},{int(valid_detections[1][4])})")
            
            # Re-run recognition with swapped detection coordinates
            if len(valid_detections) > 0:
                tl_types = [int(torch.argmax(det[5:9])) + 1 for det in valid_detections]
                recognitions = pipeline.recognize(image_tensor, valid_detections, tl_types)
                if show_details:
                    colors = ['BLACK', 'RED', 'YELLOW', 'GREEN']
                    print(f"     Re-recognized: Det0={colors[torch.argmax(recognitions[0])]}, Det1={colors[torch.argmax(recognitions[1])]}")
            
            # Re-run assignment with swapped detections
            if len(valid_detections) > 0 and len(bboxes) > 0:
                from tlr.tools.utils import boxes2projections
                projections = boxes2projections(bboxes)
                from tlr.selector import select_tls
                assignments = select_tls(
                    pipeline.ho,
                    valid_detections,
                    projections,
                    image_tensor.shape[:2]
                )
                if show_details:
                    print(f"     Re-assigned: {assignments.tolist()}")
            
            # Re-run tracking with swapped detections but CROSS-TRANSFER histories
            if len(assignments) > 0:
                # Build the assignment list for tracking
                assignment_list = [(int(a[0]), int(a[1])) for a in assignments]
                
                # Cross-transfer histories: 
                # Before swap: proj_id=0 had Det0's history, proj_id=1 had Det1's history
                # After swap: Det0 is physically at Det1's position, Det1 is at Det0's position
                # So we need to give the new positions the old histories
                temp_revised_states = pipeline.tracker.track(frame_ts, assignment_list, recognitions)
                
                # Apply cross-history transfer manually
                if frame_num == switch_frame:
                    # At the exact swap frame, manually transfer histories
                    if 0 in pipeline.tracker.semantic.history and 1 in pipeline.tracker.semantic.history:
                        # Swap the tracking histories between projection 0 and 1
                        semantic_0 = pipeline.tracker.semantic.history[0]
                        semantic_1 = pipeline.tracker.semantic.history[1]
                        
                        # Exchange their historical data (swap all attributes)
                        temp_color = semantic_0.color
                        temp_time_stamp = semantic_0.time_stamp
                        temp_last_bright = semantic_0.last_bright_time
                        temp_last_dark = semantic_0.last_dark_time
                        temp_blink = semantic_0.blink
                        temp_hyst_color = semantic_0.hysteretic_color
                        temp_hyst_count = semantic_0.hysteretic_count
                        
                        semantic_0.color = semantic_1.color
                        semantic_0.time_stamp = semantic_1.time_stamp
                        semantic_0.last_bright_time = semantic_1.last_bright_time
                        semantic_0.last_dark_time = semantic_1.last_dark_time
                        semantic_0.blink = semantic_1.blink
                        semantic_0.hysteretic_color = semantic_1.hysteretic_color
                        semantic_0.hysteretic_count = semantic_1.hysteretic_count
                        
                        semantic_1.color = temp_color
                        semantic_1.time_stamp = temp_time_stamp
                        semantic_1.last_bright_time = temp_last_bright
                        semantic_1.last_dark_time = temp_last_dark
                        semantic_1.blink = temp_blink
                        semantic_1.hysteretic_color = temp_hyst_color
                        semantic_1.hysteretic_count = temp_hyst_count
                        
                        if show_details:
                            print(f"  💾 CROSS-TRANSFERRED histories between proj 0 and 1")
                            print(f"      Proj 0 now has: color={semantic_0.color}, blink={semantic_0.blink}")
                            print(f"      Proj 1 now has: color={semantic_1.color}, blink={semantic_1.blink}")
                
                revised_states = temp_revised_states
        
        if revised_states and show_details:
            print(f"  ✅ Revised states: {revised_states}")
        
        # === STAGE OUTPUT GENERATION ===
        
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
            color = (0, 255, 0) if frame_num < switch_frame else (255, 165, 0)  # Verde antes, naranja después del swap
            cv2.rectangle(img_detection, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img_detection, f'Det{det_idx}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        # Draw projections
        for bbox in bboxes:
            x1, y1, x2, y2, proj_id = bbox
            cv2.rectangle(img_detection, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.putText(img_detection, f'Proj{proj_id}', (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
        # Add swap indicator
        if frame_num >= switch_frame:
            cv2.putText(img_detection, 'POST-SWAP (CROSS HISTORY TRANSFER)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
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
            cv2.putText(img_recognition, 'POST-SWAP (CROSS HISTORY TRANSFER)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
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
                
                # Special handling for blinking - thick orange border (different from first test)
                if is_blinking:
                    cv2.rectangle(img_final, (x1-3, y1-3), (x2+3, y2+3), (0,165,255), 5)
                
                cv2.rectangle(img_final, (x1, y1), (x2, y2), vis_color, 2)
                cv2.putText(img_final, f'Det{det_idx}->Proj{proj_id}', (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.putText(img_final, f'{revised_color.upper()}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, vis_color, 2)
                if is_blinking:
                    cv2.putText(img_final, 'BLINK', (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)
                
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
            cv2.putText(img_final, 'POST-SWAP (CROSS HISTORY TRANSFER)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
        cv2.imwrite(os.path.join(final_dir, frame_name), img_final)
        
        if final_lines:
            with open(final_csv, 'a') as out:
                out.writelines(final_lines)
    
    # Save results
    import csv
    with open(f'{output_dir}/swapping_cross_history_results.csv', 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\n📄 Results saved to {output_dir}/swapping_cross_history_results.csv")
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
    
    print(f"\n🔍 ANALYSIS (PHYSICAL SWAP + CROSS HISTORY TRANSFER):")
    print(f"   Total results: {len(results)}")
    print(f"   Frames with YELLOW detection: {len(yellow_frames)}")
    print(f"   Total BLINKING detections: {len(blinking_frames)}")
    print(f"   BLINKING before swap (< frame {switch_frame}): {len(pre_swap_blink)}")
    print(f"   BLINKING after swap (≥ frame {switch_frame}): {len(post_swap_blink)}")
    
    if blinking_frames:
        blink_frames_nums = sorted(set(r['frame_num'] for r in blinking_frames))
        print(f"   Blinking detected in frames: {blink_frames_nums}")
        
        # Analyze post-swap yellow detections specifically
        post_swap_yellow = [r for r in yellow_frames if r['post_swap']]
        post_swap_yellow_no_blink = [r for r in post_swap_yellow if not r['is_blinking']]
        
        print(f"   Post-swap YELLOW detections: {len(post_swap_yellow)}")
        print(f"   Post-swap YELLOW detections WITHOUT blink: {len(post_swap_yellow_no_blink)}")
        
        if post_swap_yellow_no_blink:
            print(f"   ✅ SUCCESS: Yellow detected without blink after cross-history transfer")
            for r in post_swap_yellow_no_blink[:3]:  # Show first few examples
                print(f"      Frame {r['frame_num']}: Det{r['det_idx']}->Proj{r['proj_id']} sees YELLOW, no blink")
        
        # Check for blinking transfer behavior
        if pre_swap_blink and post_swap_blink:
            pre_proj_ids = set(r['proj_id'] for r in pre_swap_blink)
            post_proj_ids = set(r['proj_id'] for r in post_swap_blink)
            print(f"   Pre-swap blinking proj_ids: {pre_proj_ids}")
            print(f"   Post-swap blinking proj_ids: {post_proj_ids}")
        elif pre_swap_blink and not post_swap_blink:
            print(f"   ✅ BLINKING STOPPED after swap (cross-transfer broke the blink pattern)")
        elif not pre_swap_blink and post_swap_blink:
            print(f"   ❓ BLINKING APPEARED after swap")
    
    # Expected behavior summary
    print(f"\n📋 EXPECTED BEHAVIOR (CROSS HISTORY TRANSFER):")
    print(f"   - Before frame {switch_frame}: Normal assignment, proj_id=1 should have YELLOW->RED (BLINK)")
    print(f"   - After frame {switch_frame}:  Swapped assignment AND cross-transferred histories")
    print(f"   - Key test: Right TL (yellow) gets left TL's history (green), left TL (green) gets right TL's history (blink)")
    print(f"   - Expected result: Right TL sees YELLOW but has GREEN history → should detect YELLOW (no blink)")
    print(f"   - Expected result: Left TL sees GREEN but has BLINK history → should detect GREEN (blink stops)")
    
    return results

if __name__ == "__main__":
    results = run_swapping_cross_history_transfer()