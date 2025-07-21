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
print("🔍 Debug: Frame 27 (Red+Yellow light)")

# ─── LOAD FRAME 27 ─────────────────────────────────────────────────────────────
input_dir = 'frames_auto_labeled'
bbox_file = os.path.join(input_dir, 'projection_bboxes_master.txt')

# Find frame 27 projection
entries = {}
with open(bbox_file, 'r') as f:
    for line in f:
        parts = line.strip().split(",")
        frame = parts[0]
        bbox = list(map(int, parts[1:]))
        entries.setdefault(frame, []).append(bbox)

frame_name = 'frame_0027.jpg'
bboxes = entries[frame_name]

print(f"🎯 Analyzing {frame_name}")
print(f"📦 Projection boxes: {bboxes}")

# Load image
frame_path = os.path.join(input_dir, frame_name)
image_np = cv2.imread(frame_path)
image_tensor = torch.from_numpy(image_np.astype(np.float32)).to('cuda:0')

print(f"🖼️  Original image shape: {image_np.shape}")

# Get detection
detections = pipeline.detect(image_tensor, bboxes)
tl_types = torch.argmax(detections[:, 5:], dim=1)
valid_mask = tl_types != 0
valid_detections = detections[valid_mask]

print(f"📊 Valid detections: {len(valid_detections)}")

if len(valid_detections) > 0:
    detection = valid_detections[0]
    det_box = detection[1:5].type(torch.long)
    x1, y1, x2, y2 = det_box.tolist()
    tl_type = tl_types[valid_mask][0]
    
    print(f"📍 Detection box: [{x1}, {y1}, {x2}, {y2}]")
    print(f"🎨 Traffic light type: {['bg','vert','quad','hori'][tl_type]}")
    
    # EXTRACT AND SAVE THE CROP THAT GOES TO RECOGNITION
    crop_region = image_np[y1:y2, x1:x2]
    print(f"✂️  Cropped region shape: {crop_region.shape}")
    
    # Save the crop for visual inspection
    crop_filename = f"debug_frame27_crop_{x1}_{y1}_{x2}_{y2}.jpg"
    cv2.imwrite(crop_filename, crop_region)
    print(f"💾 Saved crop to: {crop_filename}")
    
    # Also save the full image with bounding box drawn
    debug_image = image_np.copy()
    
    # Draw projection box (blue)
    for x1_proj, y1_proj, x2_proj, y2_proj, pid in bboxes:
        cv2.rectangle(debug_image, (x1_proj, y1_proj), (x2_proj, y2_proj), (255, 0, 0), 2)
        cv2.putText(debug_image, f"Proj{pid}", (x1_proj, y1_proj-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Draw detection box (green)
    cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(debug_image, "DETECTION", (x1, y1-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    full_filename = "debug_frame27_full_with_boxes.jpg"
    cv2.imwrite(full_filename, debug_image)
    print(f"💾 Saved full image with boxes to: {full_filename}")
    
    # GET RECOGNITION SCORES
    recognizer, shape = pipeline.classifiers[tl_type-1]
    
    from tlr.tools.utils import preprocess4rec
    input_tensor = preprocess4rec(image_tensor, det_box, shape, pipeline.means_rec)
    
    # Get processed input crop (after preprocessing)
    processed_crop = input_tensor.detach().cpu().numpy().astype(np.uint8)
    processed_filename = f"debug_frame27_processed_crop.jpg"
    cv2.imwrite(processed_filename, processed_crop)
    print(f"💾 Saved processed crop to: {processed_filename}")
    
    # RAW LOGITS AND SCORES
    raw_output = recognizer(input_tensor.permute(2, 0, 1).unsqueeze(0))
    
    # Manual forward to get logits
    x = input_tensor.permute(2, 0, 1).unsqueeze(0)
    conv1 = F.max_pool2d(F.relu(recognizer.conv1(x)), kernel_size=3, stride=2, padding=1)
    conv2 = F.max_pool2d(F.relu(recognizer.conv2(conv1)), kernel_size=3, stride=2, padding=1)
    conv3 = F.max_pool2d(F.relu(recognizer.conv3(conv2)), kernel_size=3, stride=2, padding=1)
    conv4 = F.max_pool2d(F.relu(recognizer.conv4(conv3)), kernel_size=3, stride=2, padding=1)
    conv5 = recognizer.pool5(F.relu(recognizer.conv5(conv4)))
    ft = F.relu(recognizer.ft(conv5))
    raw_logits = recognizer.logits(ft.reshape(-1, recognizer.logits.in_features))
    
    logits_list = raw_logits[0].detach().cpu().tolist()
    probs_list = raw_output[0].detach().cpu().tolist()
    
    color_names = ['black', 'red', 'yellow', 'green']
    
    print(f"\n🎯 RECOGNITION RESULTS:")
    for i, (color, logit, prob) in enumerate(zip(color_names, logits_list, probs_list)):
        print(f"   {color:6}: logit={logit:8.4f}, prob={prob:.4f}")
    
    max_idx = logits_list.index(max(logits_list))
    print(f"\n🏆 Model prediction: {color_names[max_idx]} (confidence: {max(probs_list):.4f})")
    
    # Check if it's red+yellow situation
    red_logit = logits_list[1]
    yellow_logit = logits_list[2]
    logit_diff = abs(red_logit - yellow_logit)
    
    print(f"\n🚦 RED+YELLOW ANALYSIS:")
    print(f"   Red logit:    {red_logit:.4f}")
    print(f"   Yellow logit: {yellow_logit:.4f}")
    print(f"   Difference:   {logit_diff:.4f}")
    print(f"   Close scores? {logit_diff < 10}")  # Arbitrary threshold
    
else:
    print("❌ No valid detections found")

print(f"\n📸 Check the saved images to see what the model is actually seeing:")
print(f"   - {crop_filename}: Raw crop from original image")
print(f"   - {processed_filename}: Processed crop fed to recognizer")  
print(f"   - {full_filename}: Full image with detection boxes")