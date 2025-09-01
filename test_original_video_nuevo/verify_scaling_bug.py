#!/usr/bin/env python3
"""
Verify the coordinate scaling bug theory
"""

print("=" * 80)
print("VERIFYING COORDINATE SCALING BUG")
print("=" * 80)

# Known values:
crop_offset_x = 323
crop_offset_y = 158
crop_size = 625
detector_input_size = 270
scale_factor = crop_size / detector_input_size  # 2.315

actual_detection = [466, 260, 489, 327]
expected_projection = [511, 349, 761, 592]

print(f"🔹 CURRENT PIPELINE (BUGGY):")
print(f"   1. Detector outputs coordinates in {detector_input_size}x{detector_input_size} space")
print(f"   2. restore_boxes_to_full_image() directly adds crop offset")
print(f"   3. Final detection: {actual_detection}")

print(f"\n🔹 CORRECT PIPELINE (FIXED):")
print(f"   1. Detector outputs coordinates in {detector_input_size}x{detector_input_size} space")
print(f"   2. Scale coordinates by {scale_factor:.3f} to {crop_size}x{crop_size} space")
print(f"   3. Then add crop offset to get full image coordinates")

# Reverse calculation to find detector output
detector_output_x1 = (actual_detection[0] - crop_offset_x) / scale_factor
detector_output_y1 = (actual_detection[1] - crop_offset_y) / scale_factor
detector_output_x2 = (actual_detection[2] - crop_offset_x) / scale_factor
detector_output_y2 = (actual_detection[3] - crop_offset_y) / scale_factor

print(f"\n🔹 REVERSE CALCULATION (what detector actually output):")
print(f"   Detector output in 270x270 space: [{detector_output_x1:.1f}, {detector_output_y1:.1f}, {detector_output_x2:.1f}, {detector_output_y2:.1f}]")

# Now calculate what the CORRECT coordinates should be
correct_x1 = detector_output_x1 * scale_factor + crop_offset_x
correct_y1 = detector_output_y1 * scale_factor + crop_offset_y
correct_x2 = detector_output_x2 * scale_factor + crop_offset_x  
correct_y2 = detector_output_y2 * scale_factor + crop_offset_y

print(f"\n🔹 WHAT THE CORRECT COORDINATES SHOULD BE:")
print(f"   With proper scaling: [{correct_x1:.1f}, {correct_y1:.1f}, {correct_x2:.1f}, {correct_y2:.1f}]")

print(f"\n🔹 COMPARISON:")
print(f"   Input projection:  [511, 349, 761, 592]")
print(f"   Current (buggy):   {actual_detection}")
print(f"   Should be (fixed): [{correct_x1:.0f}, {correct_y1:.0f}, {correct_x2:.0f}, {correct_y2:.0f}]")

print(f"\n🔹 THE BUG LOCATION:")
print(f"   The bug is in restore_boxes_to_full_image() function")
print(f"   It's missing the scaling step to convert detector coordinates")
print(f"   from {detector_input_size}x{detector_input_size} back to crop coordinates")

print(f"\n🔹 VERIFICATION:")
offset_x = expected_projection[0] - actual_detection[0]  # 511 - 466 = 45
offset_y = expected_projection[1] - actual_detection[1]  # 349 - 260 = 89
expected_offset_x = crop_offset_x * (1 - 1/scale_factor)  # Expected offset due to scaling
expected_offset_y = crop_offset_y * (1 - 1/scale_factor)  

print(f"   Observed offset: ({offset_x}, {offset_y})")
print(f"   Expected scaling-related offset: ({expected_offset_x:.1f}, {expected_offset_y:.1f})")

scale_missing = 1 / scale_factor  # What we're missing
print(f"\n🔹 SCALING FACTOR ANALYSIS:")
print(f"   Missing scale factor: {scale_missing:.3f}")
print(f"   This explains why coordinates are compressed by ~{scale_missing:.1%}")