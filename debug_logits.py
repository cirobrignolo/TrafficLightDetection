import cv2
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

# 🔧 Fix para importar tlr.pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from tlr.pipeline import load_pipeline

# ─── LOAD PIPELINE ─────────────────────────────────────────────────────────────
pipeline = load_pipeline('cuda:0')
print("🔍 Debug: Checking raw logits (before softmax)")

# ─── TEST ONE FRAME ─────────────────────────────────────────────────────────────
input_dir = 'frames_auto_labeled'
bbox_file = os.path.join(input_dir, 'projection_bboxes_master.txt')

# Read one projection
with open(bbox_file, 'r') as f:
    line = f.readline().strip()
    parts = line.split(",")
    frame_name = parts[0]
    bbox = list(map(int, parts[1:]))

print(f"🎯 Testing frame: {frame_name}")

# Load image
frame_path = os.path.join(input_dir, frame_name)
image_np = cv2.imread(frame_path)
image_tensor = torch.from_numpy(image_np.astype(np.float32)).to('cuda:0')

# Get detection
detections = pipeline.detect(image_tensor, [bbox])
tl_types = torch.argmax(detections[:, 5:], dim=1)
valid_mask = tl_types != 0
valid_detections = detections[valid_mask]

if len(valid_detections) > 0:
    detection = valid_detections[0]
    det_box = detection[1:5].type(torch.long)
    tl_type = tl_types[valid_mask][0]
    
    # Get recognizer
    recognizer, shape = pipeline.classifiers[tl_type-1]
    
    # Preprocess
    from tlr.tools.utils import preprocess4rec
    input_tensor = preprocess4rec(image_tensor, det_box, shape, pipeline.means_rec)
    
    # MANUAL FORWARD PASS TO CHECK LOGITS
    print(f"🧠 Manual forward pass through recognizer:")
    
    x = input_tensor.permute(2, 0, 1).unsqueeze(0)
    print(f"📊 Input shape: {x.shape}")
    
    # Forward through network layers
    conv1 = F.max_pool2d(F.relu(recognizer.conv1(x)), kernel_size=3, stride=2, padding=1)
    conv2 = F.max_pool2d(F.relu(recognizer.conv2(conv1)), kernel_size=3, stride=2, padding=1)
    conv3 = F.max_pool2d(F.relu(recognizer.conv3(conv2)), kernel_size=3, stride=2, padding=1)
    conv4 = F.max_pool2d(F.relu(recognizer.conv4(conv3)), kernel_size=3, stride=2, padding=1)
    conv5 = recognizer.pool5(F.relu(recognizer.conv5(conv4)))
    ft = F.relu(recognizer.ft(conv5))
    
    # RAW LOGITS (before softmax)
    raw_logits = recognizer.logits(ft.reshape(-1, recognizer.logits.in_features))
    logits_list = raw_logits[0].detach().cpu().tolist()
    
    # SOFTMAX PROBABILITIES
    probs = F.softmax(raw_logits, dim=1)
    probs_list = probs[0].detach().cpu().tolist()
    
    print(f"🎯 RAW LOGITS (before softmax): {[f'{l:.4f}' for l in logits_list]}")
    print(f"📊 SOFTMAX PROBS: {[f'{p:.4f}' for p in probs_list]}")
    print(f"📈 Logit range: {min(logits_list):.4f} to {max(logits_list):.4f}")
    print(f"📉 Difference between max and second: {sorted(logits_list, reverse=True)[0] - sorted(logits_list, reverse=True)[1]:.4f}")
    
    color_names = ['black', 'red', 'yellow', 'green']
    max_idx = logits_list.index(max(logits_list))
    print(f"🏆 Predicted: {color_names[max_idx]}")
    
else:
    print("❌ No valid detections found")