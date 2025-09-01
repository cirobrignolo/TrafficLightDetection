#!/usr/bin/env python3
"""
Debug script to identify the exact coordinate transformation bug
"""

# From the analysis, I suspect there's an issue in restore_boxes_to_full_image
# Let's check the coordinate restoration step by step

print("=" * 80)
print("APOLLO TLR COORDINATE BUG ANALYSIS")
print("=" * 80)

# Input: projection [511, 349, 761, 592, 0]
# Output: detection [466, 260, 489, 327]

# The crop function calculates [xl, xr, yt, yb] = [323, 948, 158, 783]
xl_crop = 323
yt_crop = 158

print(f"🔹 CROP COORDINATES:")
print(f"   xl_crop = {xl_crop}")
print(f"   yt_crop = {yt_crop}")

print(f"\n🔹 DETECTION COORDINATE RESTORATION:")
print(f"   restore_boxes_to_full_image() adds crop offset:")
print(f"   detection[:, 1] += xl_crop  # x1 += {xl_crop}")
print(f"   detection[:, 2] += yt_crop  # y1 += {yt_crop}")
print(f"   detection[:, 3] += xl_crop  # x2 += {xl_crop}")
print(f"   detection[:, 4] += yt_crop  # y2 += {yt_crop}")

print(f"\n🔹 POTENTIAL ISSUE ANALYSIS:")

# If the detection came out as [466, 260, 489, 327] after restoration,
# then the original detection (before restoration) would have been:
original_x1 = 466 - xl_crop  # 466 - 323 = 143
original_y1 = 260 - yt_crop  # 260 - 158 = 102  
original_x2 = 489 - xl_crop  # 489 - 323 = 166
original_y2 = 327 - yt_crop  # 327 - 158 = 169

print(f"   If detection [466, 260, 489, 327] is AFTER restoration,")
print(f"   then BEFORE restoration it would be: [{original_x1}, {original_y1}, {original_x2}, {original_y2}]")
print(f"   Size: {original_x2 - original_x1} x {original_y2 - original_y1} = {original_x2 - original_x1} x {original_y2 - original_y1}")

print(f"\n🔹 CHECKING FOR COORDINATE ORDER BUG:")
print(f"   Standard bbox format: [x1, y1, x2, y2]")
print(f"   Detection tensor format: det[1:5] = [?, ?, ?, ?]")

# Let's check if there's a coordinate swap issue
# Looking at the selector.py center calculation:
print(f"\n🔹 SELECTOR.PY CENTER CALCULATION BUG:")
print(f"   Line 20: center_refine = [int((detection[3] + detection[1])/2), int((detection[4] + detection[2])/2)]")
print(f"   This suggests: detection[1]=x1, detection[2]=y1, detection[3]=x2, detection[4]=y2")
print(f"   BUT this is WRONG! It should be:")
print(f"   center_x = (detection[1] + detection[3])/2  # (x1 + x2)/2")
print(f"   center_y = (detection[2] + detection[4])/2  # (y1 + y2)/2")

print(f"\n🔹 COORDINATE SWAP BUG FOUND!")
print(f"   In selector.py line 20, the center calculation swaps x and y coordinates:")
print(f"   BUGGY:  center_refine = [(x2+x1)/2, (y2+y1)/2] ← x and y swapped!")
print(f"   CORRECT: center_refine = [(x1+x2)/2, (y1+y2)/2]")

# Let's also check if there are other coordinate order issues
print(f"\n🔹 CHECKING RESTORE_BOXES_TO_FULL_IMAGE:")
print(f"   Lines 270-273: detection[:, start_col+i] += offset")
print(f"   start_col=1, so:")
print(f"   detection[:, 1] += xl  # Should be x1 coordinate")
print(f"   detection[:, 2] += yt  # Should be y1 coordinate") 
print(f"   detection[:, 3] += xl  # Should be x2 coordinate")
print(f"   detection[:, 4] += yt  # Should be y2 coordinate")
print(f"   This looks correct IF detection format is [?, x1, y1, x2, y2, ...]")

print(f"\n🔹 SUMMARY OF COORDINATE BUGS:")
print(f"   1. selector.py line 20: center calculation has x/y coordinate swap")
print(f"   2. Need to verify detection tensor format consistency across all modules")
print(f"   3. The 45/89 pixel offset suggests there might be additional issues")