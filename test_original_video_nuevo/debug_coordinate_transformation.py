#!/usr/bin/env python3
"""
Debug script to trace coordinate transformations in Apollo TLR pipeline
"""

# Simulate the coordinate transformation step by step
print("=" * 80)
print("APOLLO TLR COORDINATE TRANSFORMATION DEBUG")
print("=" * 80)

# Input projection: [511, 349, 761, 592] with ID=0
projection_input = [511, 349, 761, 592, 0]
print(f"🔹 INPUT PROJECTION: {projection_input}")
print(f"   Format: [xmin, ymin, xmax, ymax, id]")

# Step 1: Convert to ProjectionROI (from box2projection)
x = projection_input[0]  # 511
y = projection_input[1]  # 349  
w = projection_input[2] - projection_input[0]  # 761 - 511 = 250
h = projection_input[3] - projection_input[1]  # 592 - 349 = 243

print(f"\n🔹 STEP 1: box2projection() conversion")
print(f"   x={x}, y={y}, w={w}, h={h}")

# ProjectionROI calculation (from utils.py lines 157-167)
xl = x  # 511
yt = y  # 349
xr = x + w - 1  # 511 + 250 - 1 = 760
yb = y + h - 1  # 349 + 243 - 1 = 591
center_x = int((xl + xr) / 2)  # (511 + 760) / 2 = 635.5 → 635
center_y = int((yt + yb) / 2)  # (349 + 591) / 2 = 470

print(f"   ProjectionROI bounds:")
print(f"   xl={xl}, yt={yt}, xr={xr}, yb={yb}")
print(f"   center_x={center_x}, center_y={center_y}")

# Step 2: Crop calculation (from crop function)
print(f"\n🔹 STEP 2: crop() calculation")
print(f"   Assuming image_shape = (1080, 1920, 3)")  # Typical video resolution

image_shape = (1080, 1920, 3)  # height, width, channels
width = image_shape[1]   # 1920
height = image_shape[0]  # 1080
crop_scale = 2.5
min_crop_size = 270

resize = crop_scale * max(w, h)  # 2.5 * max(250, 243) = 2.5 * 250 = 625
resize = max(resize, min_crop_size)  # max(625, 270) = 625
resize = min(resize, width)   # min(625, 1920) = 625
resize = min(resize, height)  # min(625, 1080) = 625

print(f"   crop_scale={crop_scale}, min_crop_size={min_crop_size}")
print(f"   resize calculation: 2.5 * max({w}, {h}) = {resize}")

xl_crop = center_x - resize/2 + 1  # 635 - 625/2 + 1 = 635 - 312.5 + 1 = 323.5
xl_crop = 0 if xl_crop < 0 else xl_crop  # 323.5

yt_crop = center_y - resize/2 + 1  # 470 - 625/2 + 1 = 470 - 312.5 + 1 = 158.5  
yt_crop = 0 if yt_crop < 0 else yt_crop  # 158.5

xr_crop = xl_crop + resize - 1  # 323.5 + 625 - 1 = 947.5
yb_crop = yt_crop + resize - 1  # 158.5 + 625 - 1 = 782.5

print(f"   Initial crop calculation:")
print(f"   xl_crop={xl_crop}, yt_crop={yt_crop}")
print(f"   xr_crop={xr_crop}, yb_crop={yb_crop}")

# Boundary adjustments
if xr_crop >= width - 1:  # 947.5 >= 1919? No
    xl_crop -= xr_crop - width + 1
    xr_crop = width - 1

if yb_crop >= height - 1:  # 782.5 >= 1079? No
    yt_crop -= yb_crop - height + 1
    yb_crop = height - 1

crop_result = [int(xl_crop), int(xr_crop + 1), int(yt_crop), int(yb_crop + 1)]
print(f"   Final crop coordinates: [xl, xr, yt, yb] = {crop_result}")

# Step 3: Detection processing (detector runs on cropped 270x270 region)
print(f"\n🔹 STEP 3: Detection processing")
print(f"   Detector input: 270x270 resized crop from {crop_result}")
print(f"   Detector outputs bounding boxes relative to 270x270 crop")

# Step 4: restore_boxes_to_full_image
print(f"\n🔹 STEP 4: restore_boxes_to_full_image()")
print(f"   This should add crop offset back to detection coordinates:")
print(f"   detection[:, 1] += xl_crop  # x1 coordinate")
print(f"   detection[:, 2] += yt_crop  # y1 coordinate") 
print(f"   detection[:, 3] += xl_crop  # x2 coordinate")
print(f"   detection[:, 4] += yt_crop  # y2 coordinate")

print(f"\n🔹 EXPECTED vs ACTUAL RESULTS:")
print(f"   Input projection: [511, 349, 761, 592]")
print(f"   Actual detection: [466, 260, 489, 327]")
print(f"   Expected detection should be INSIDE the projection or very close to it")

print(f"\n🔹 OFFSET ANALYSIS:")
x_offset = 511 - 466  # 45 pixels left
y_offset = 349 - 260  # 89 pixels up
print(f"   X offset: {x_offset} pixels (detection is LEFT of projection)")
print(f"   Y offset: {y_offset} pixels (detection is ABOVE projection)")

print(f"\n🔹 SUSPECTED ISSUES:")
print(f"   1. The crop calculation might have rounding errors")
print(f"   2. The restore_boxes_to_full_image offset calculation might be wrong")
print(f"   3. There might be a coordinate system mismatch (0-based vs 1-based indexing)")
print(f"   4. The 270x270 resizing might introduce scaling artifacts")