#!/usr/bin/env python3
"""
Debug the exact selector.py coordinate bug
"""

print("=" * 80)
print("SELECTOR.PY COORDINATE BUG FOUND!")
print("=" * 80)

print(f"🔹 DETECTION TENSOR FORMAT:")
print(f"   detection[0] = 0 (dummy)")
print(f"   detection[1] = x1")
print(f"   detection[2] = y1")
print(f"   detection[3] = x2")
print(f"   detection[4] = y2")
print(f"   detection[5:9] = scores")

print(f"\n🔹 SELECTOR.PY LINE 20 BUG:")
print(f"   CURRENT (BUGGY): center_refine = [int((detection[3] + detection[1])/2), int((detection[4] + detection[2])/2)]")
print(f"   MEANING:         center_refine = [(x2 + x1)/2, (y2 + y1)/2]")
print(f"   RESULT:          center_refine = [center_x, center_y]")

print(f"\n🔹 WAIT... That's actually CORRECT!")
print(f"   (x2 + x1)/2 = center_x")
print(f"   (y2 + y1)/2 = center_y")

print(f"\n🔹 THE REAL BUG MIGHT BE ELSEWHERE...")
print(f"   Let me check the crop calculation more carefully")

# Let's trace the actual crop calculation with exact values
print(f"\n🔹 DETAILED CROP CALCULATION:")

# Input projection: [511, 349, 761, 592, 0] 
w = 761 - 511  # 250
h = 592 - 349  # 243
center_x = int((511 + 760) / 2)  # 635 (760 = 761-1)
center_y = int((349 + 591) / 2)  # 470 (591 = 592-1)

print(f"   Projection: w={w}, h={h}")
print(f"   Center: ({center_x}, {center_y})")

# Crop calculation
crop_scale = 2.5
min_crop_size = 270
resize = crop_scale * max(w, h)  # 2.5 * 250 = 625
resize = max(resize, min_crop_size)  # 625 (no change)

xl_crop = center_x - resize/2 + 1  # 635 - 312.5 + 1 = 323.5
yt_crop = center_y - resize/2 + 1  # 470 - 312.5 + 1 = 158.5
xr_crop = xl_crop + resize - 1     # 323.5 + 625 - 1 = 947.5
yb_crop = yt_crop + resize - 1     # 158.5 + 625 - 1 = 782.5

print(f"   Crop region: xl={xl_crop}, yt={yt_crop}, xr={xr_crop}, yb={yb_crop}")
print(f"   Crop size: {xr_crop - xl_crop + 1} x {yb_crop - yt_crop + 1}")

# After int() conversion in crop function
xl_final = int(xl_crop)  # 323
xr_final = int(xr_crop + 1)  # int(948.5) = 948
yt_final = int(yt_crop)  # 158  
yb_final = int(yb_crop + 1)  # int(783.5) = 783

print(f"   Final crop: [{xl_final}, {xr_final}, {yt_final}, {yb_final}]")
print(f"   Final crop size: {xr_final - xl_final} x {yb_final - yt_final} = {xr_final - xl_final} x {yb_final - yt_final}")

print(f"\n🔹 ROUNDING ERROR ANALYSIS:")
print(f"   xl_crop: 323.5 → 323 (lost 0.5 pixels)")
print(f"   yt_crop: 158.5 → 158 (lost 0.5 pixels)")
print(f"   These rounding errors could accumulate to cause coordinate drift!")

print(f"\n🔹 POTENTIAL ROOT CAUSE:")
print(f"   The crop calculation uses center ± resize/2, but with +1 adjustments")
print(f"   that create fractional coordinates. When these get rounded down")
print(f"   during int() conversion, it introduces systematic offset errors.")