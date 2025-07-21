import cv2
import os
import sys
import torch
import numpy as np

# 🔧 Fix para importar tlr.pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from tlr.pipeline import load_pipeline

# ─── CONFIG ───────────────────────────────────────────────────────────────────
input_dir = 'frames_auto_labeled'
bbox_file = os.path.join(input_dir, 'projection_bboxes_master.txt')

# ─── CARGA DEL PIPELINE ─────────────────────────────────────────────────────────
pipeline = load_pipeline('cuda:0')
print("🔍 Debug: Checking raw recognition scores")

# ─── LEER PROYECCIONES ───────────────────────────────────────────────────────────
entries = {}
with open(bbox_file, 'r') as f:
    for line in f:
        parts = line.strip().split(",")
        frame = parts[0]
        bbox = list(map(int, parts[1:]))
        entries.setdefault(frame, []).append(bbox)

# ─── DEBUG ESPECÍFICO DE ALGUNOS FRAMES ──────────────────────────────────────────
test_frames = ['frame_0000.jpg', 'frame_0002.jpg', 'frame_0030.jpg']

for frame_name in test_frames:
    print(f"\n🔍 ═══ RAW SCORES DEBUG: {frame_name} ═══")
    
    bboxes = entries[frame_name]
    frame_path = os.path.join(input_dir, frame_name)
    image_np = cv2.imread(frame_path)
    image_tensor = torch.from_numpy(image_np.astype(np.float32)).to('cuda:0')
    
    # BYPASS THE PIPELINE AND CALL COMPONENTS DIRECTLY
    
    # 1) Raw detection
    detections = pipeline.detect(image_tensor, bboxes)
    print(f"📊 Raw detections shape: {detections.shape}")
    
    # 2) Filter valid detections
    tl_types = torch.argmax(detections[:, 5:], dim=1)
    valid_mask = tl_types != 0
    valid_detections = detections[valid_mask]
    print(f"📊 Valid detections: {len(valid_detections)}")
    
    if len(valid_detections) > 0:
        # 3) Raw recognition WITHOUT the confidence threshold modification
        print(f"🎯 CALLING RECOGNIZER DIRECTLY (NO THRESHOLD):")
        
        for det_idx, detection in enumerate(valid_detections):
            det_box = detection[1:5].type(torch.long)
            tl_type = tl_types[valid_mask][det_idx]
            
            # Get the recognizer for this type
            recognizer, shape = pipeline.classifiers[tl_type-1]
            
            # Call preprocessing manually
            from tlr.tools.utils import preprocess4rec
            input_tensor = preprocess4rec(image_tensor, det_box, shape, pipeline.means_rec)
            
            # Call recognizer directly to get RAW scores
            raw_output = recognizer(input_tensor.permute(2, 0, 1).unsqueeze(0))
            raw_scores = raw_output[0].detach().cpu().tolist()
            
            max_prob = max(raw_scores)
            max_idx = raw_scores.index(max_prob)
            color_names = ['black', 'red', 'yellow', 'green']
            
            print(f"   Detection {det_idx}:")
            print(f"   📍 Box: {det_box.tolist()}")
            print(f"   🎨 Type: {['bg','vert','quad','hori'][tl_type]}")
            print(f"   📊 RAW recognition scores: {[f'{s:.4f}' for s in raw_scores]}")
            print(f"   🏆 Max confidence: {max_prob:.4f} → {color_names[max_idx]}")
            print(f"   ✅ Passes 0.5 threshold: {max_prob >= 0.5}")
            print()
    
    print(f"═══ END {frame_name} ═══")

print("\n🔍 This shows the RAW model outputs before any thresholding or modifications")