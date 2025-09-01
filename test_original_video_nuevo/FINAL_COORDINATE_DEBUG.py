#!/usr/bin/env python3
"""
Final comprehensive coordinate debugging to identify ALL issues
"""

print("=" * 80)
print("COMPREHENSIVE APOLLO TLR COORDINATE BUG ANALYSIS")
print("=" * 80)

# THE KEY INSIGHT: Let me trace what the detector SHOULD output vs what it DOES output

print("🔹 EXPECTED BEHAVIOR:")
print("   If the pipeline worked correctly, detections should be INSIDE or very close to projections")

input_proj = [511, 349, 761, 592, 0]
actual_det = [466, 260, 489, 327]

print(f"   Input projection: {input_proj[:4]}")
print(f"   Actual detection: {actual_det}")

# Check if detection is inside projection
det_inside_proj = (
    actual_det[0] >= input_proj[0] and  # x1 >= proj_x1
    actual_det[1] >= input_proj[1] and  # y1 >= proj_y1
    actual_det[2] <= input_proj[2] and  # x2 <= proj_x2
    actual_det[3] <= input_proj[3]      # y2 <= proj_y2
)

print(f"   Detection inside projection: {det_inside_proj}")
print(f"   Detection is OUTSIDE and OFFSET from projection!")

print(f"\n🔹 OFFSET ANALYSIS:")
offset_x1 = input_proj[0] - actual_det[0]  # 511 - 466 = 45
offset_y1 = input_proj[1] - actual_det[1]  # 349 - 260 = 89
offset_x2 = input_proj[2] - actual_det[2]  # 761 - 489 = 272
offset_y2 = input_proj[3] - actual_det[3]  # 592 - 327 = 265

print(f"   Top-left offset: ({offset_x1}, {offset_y1})")
print(f"   Bottom-right offset: ({offset_x2}, {offset_y2})")

# The bottom-right offset is much larger, suggesting the detection is much smaller
det_width = actual_det[2] - actual_det[0]   # 489 - 466 = 23
det_height = actual_det[3] - actual_det[1]  # 327 - 260 = 67
proj_width = input_proj[2] - input_proj[0]  # 761 - 511 = 250
proj_height = input_proj[3] - input_proj[1] # 592 - 349 = 243

print(f"\n🔹 SIZE COMPARISON:")
print(f"   Projection size: {proj_width} x {proj_height}")
print(f"   Detection size: {det_width} x {det_height}")
print(f"   Size ratio: {det_width/proj_width:.3f} x {det_height/proj_height:.3f}")

print(f"\n🔹 CRITICAL INSIGHT:")
print(f"   The detection is MUCH smaller than the projection!")
print(f"   This suggests the detector is finding the actual traffic light")
print(f"   WITHIN the projection region, which is correct behavior.")
print(f"   The 'offset' might not be a bug - it might be feature detection!")

print(f"\n🔹 HYPOTHESIS TEST:")
print(f"   IF the detector correctly found a small traffic light")
print(f"   at coordinates [143, 102, 166, 169] within the 270x270 crop,")
print(f"   and IF the crop offset is correctly [323, 158],")
print(f"   then the final coordinates should be:")

crop_det = [143, 102, 166, 169]  # Detector output in crop space
crop_offset = [323, 158]
scale_factor = 625 / 270

# Apply scaling and offset
final_x1 = crop_det[0] * scale_factor + crop_offset[0]
final_y1 = crop_det[1] * scale_factor + crop_offset[1]
final_x2 = crop_det[2] * scale_factor + crop_offset[0]
final_y2 = crop_det[3] * scale_factor + crop_offset[1]

print(f"   Calculated: [{final_x1:.0f}, {final_y1:.0f}, {final_x2:.0f}, {final_y2:.0f}]")
print(f"   Actual:     {actual_det}")
print(f"   Match: {abs(final_x1 - actual_det[0]) < 1 and abs(final_y1 - actual_det[1]) < 1}")

print(f"\n🔹 ALTERNATIVE HYPOTHESIS:")
print(f"   Maybe the coordinates are in different order than expected?")
print(f"   Let me check if there's a coordinate swap bug...")

# Check if coordinates might be swapped
print(f"\n🔹 COORDINATE SWAP CHECK:")
print(f"   Standard format: [x1, y1, x2, y2]")
print(f"   Possible swap:   [y1, x1, y2, x2]?")

# If coordinates were swapped:
swapped_det = [actual_det[1], actual_det[0], actual_det[3], actual_det[2]]  # [260, 466, 327, 489]
print(f"   If swapped: {swapped_det}")
print(f"   Still doesn't match projection bounds...")

print(f"\n🔹 NEED TO CHECK:")
print(f"   1. Is the restore_boxes_to_full_image() scaling step missing?")
print(f"   2. Are there coordinate system inconsistencies?")
print(f"   3. Is the crop calculation off by a systematic amount?")
print(f"   4. Are the detector coordinates in a different coordinate system?")

print(f"\n🔹 ACTION NEEDED:")
print(f"   Check if restore_boxes_to_full_image() needs to apply scaling")
print(f"   before adding the crop offset.")