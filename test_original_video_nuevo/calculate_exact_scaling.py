#!/usr/bin/env python3
"""
Calculate the exact scaling factor needed to fix the coordinate bug
"""

import math

print("=" * 80)
print("EXACT SCALING FACTOR CALCULATION")
print("=" * 80)

# From the crop function, we know:
# 1. Input projection: [511, 349, 761, 592] 
# 2. Crop region: [323, 948, 158, 783] (xl, xr, yt, yb)
# 3. Crop size: 625 x 625
# 4. Detector input: 270 x 270

crop_xl = 323
crop_yt = 158
crop_width = 948 - 323   # 625
crop_height = 783 - 158  # 625
detector_size = 270

print(f"🔹 COORDINATE TRANSFORMATION CHAIN:")
print(f"   1. Crop region: [{crop_xl}, {crop_xl + crop_width}, {crop_yt}, {crop_yt + crop_height}]")
print(f"   2. Crop size: {crop_width} x {crop_height}")
print(f"   3. Detector input: {detector_size} x {detector_size}")

# Calculate scaling factors  
scale_x = crop_width / detector_size   # 625 / 270 = 2.315
scale_y = crop_height / detector_size  # 625 / 270 = 2.315

print(f"   4. Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")

# Current output coordinates (buggy)
current_output = [466, 260, 489, 327]

# Reverse calculation to find what detector actually output
detector_x1 = (current_output[0] - crop_xl) / scale_x
detector_y1 = (current_output[1] - crop_yt) / scale_y
detector_x2 = (current_output[2] - crop_xl) / scale_x
detector_y2 = (current_output[3] - crop_yt) / scale_y

print(f"\n🔹 REVERSE ENGINEERING DETECTOR OUTPUT:")
print(f"   Current output: {current_output}")
print(f"   Implied detector output (270x270 space): [{detector_x1:.1f}, {detector_y1:.1f}, {detector_x2:.1f}, {detector_y2:.1f}]")

# This should be within 0-270 range
print(f"   Detector output range check: x∈[0,270], y∈[0,270]")
print(f"   x1={detector_x1:.1f} ✓, y1={detector_y1:.1f} ✓, x2={detector_x2:.1f} ✓, y2={detector_y2:.1f} ✓")

# Calculate what the correct output should be with proper scaling
correct_x1 = detector_x1 * scale_x + crop_xl
correct_y1 = detector_y1 * scale_y + crop_yt
correct_x2 = detector_x2 * scale_x + crop_xl
correct_y2 = detector_y2 * scale_y + crop_yt

print(f"\n🔹 CORRECT COORDINATES WITH PROPER SCALING:")
print(f"   Should be: [{correct_x1:.0f}, {correct_y1:.0f}, {correct_x2:.0f}, {correct_y2:.0f}]")
print(f"   Actually: {current_output}")
print(f"   These are identical! So the coordinates ARE being scaled...")

print(f"\n🔹 WAIT - THE BUG MIGHT BE ELSEWHERE!")
print(f"   Since the math works out, the coordinates ARE being properly scaled.")
print(f"   The bug must be in a different part of the pipeline.")

# Let me check if the issue is in the crop calculation itself
# Maybe the crop region is calculated incorrectly?

print(f"\n🔹 CHECKING CROP CALCULATION BUG:")
original_proj = [511, 349, 761, 592, 0]
proj_center_x = (511 + 760) // 2  # 635 (using 760 = 761-1)
proj_center_y = (349 + 591) // 2  # 470 (using 591 = 592-1)

print(f"   Input projection center: ({proj_center_x}, {proj_center_y})")

# The crop should be centered around this point
crop_center_x = (crop_xl + crop_xl + crop_width) / 2  # (323 + 948) / 2 = 635.5
crop_center_y = (crop_yt + crop_yt + crop_height) / 2  # (158 + 783) / 2 = 470.5

print(f"   Crop region center: ({crop_center_x:.1f}, {crop_center_y:.1f})")
print(f"   Center difference: ({crop_center_x - proj_center_x:.1f}, {crop_center_y - proj_center_y:.1f})")

print(f"\n🔹 POTENTIAL ISSUE: CROP + 1 OFFSET")
print(f"   In crop() function lines 220, 222:")
print(f"   xl = projection.center_x - resize/2 + 1")
print(f"   yt = projection.center_y - resize/2 + 1")
print(f"   The +1 offset might be causing systematic coordinate drift!")

xl_without_offset = proj_center_x - 625/2      # 635 - 312.5 = 322.5
yt_without_offset = proj_center_y - 625/2      # 470 - 312.5 = 157.5

print(f"   Without +1 offset: xl={xl_without_offset}, yt={yt_without_offset}")
print(f"   With +1 offset: xl={xl_without_offset + 1}, yt={yt_without_offset + 1}")
print(f"   Difference: (+1, +1) in crop coordinates")

scale_factor = 625 / 270
offset_in_final_coords = 1 * scale_factor
print(f"   This +1 offset becomes {offset_in_final_coords:.1f} pixels in final coordinates")
print(f"   But we see {current_output[0] - original_proj[0]} = {current_output[0] - original_proj[0]} pixel X offset")
print(f"   And {current_output[1] - original_proj[1]} = {current_output[1] - original_proj[1]} pixel Y offset")

print(f"\n🔹 CONCLUSION:")
print(f"   The coordinate offset is NOT just a simple scaling issue.")
print(f"   There may be multiple contributing factors including:")
print(f"   1. The +1 offset in crop calculation")
print(f"   2. Rounding errors during coordinate transformations")  
print(f"   3. Possible tensor indexing inconsistencies")
print(f"   4. Integer vs float coordinate handling discrepancies")