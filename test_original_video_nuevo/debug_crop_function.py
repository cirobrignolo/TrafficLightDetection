#!/usr/bin/env python3
"""
Debug the crop function line by line to find the exact coordinate bug
"""

print("=" * 80)
print("DEBUGGING CROP FUNCTION - LINE BY LINE")
print("=" * 80)

# Simulate the exact crop function from utils.py lines 211-232
class ProjectionROI:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.xl = x
        self.yt = y
        self.xr = x + w - 1
        self.yb = y + h - 1
        self.center_x = int((self.xl + self.xr) / 2)
        self.center_y = int((self.yt + self.yb) / 2)

def crop_debug(image_shape, projection):
    width = image_shape[1]   # 1920
    height = image_shape[0]  # 1080
    crop_scale = 2.5
    min_crop_size = 270
    
    print(f"🔹 CROP FUNCTION INPUTS:")
    print(f"   image_shape = {image_shape}")
    print(f"   projection.w = {projection.w}")
    print(f"   projection.h = {projection.h}")
    print(f"   projection.center_x = {projection.center_x}")
    print(f"   projection.center_y = {projection.center_y}")
    
    resize = crop_scale * max(projection.w, projection.h)
    print(f"\n🔹 RESIZE CALCULATION:")
    print(f"   resize = {crop_scale} * max({projection.w}, {projection.h}) = {resize}")
    
    resize = max(resize, min_crop_size)
    print(f"   resize = max({resize}, {min_crop_size}) = {resize}")
    
    resize = min(resize, width)
    print(f"   resize = min({resize}, {width}) = {resize}")
    
    resize = min(resize, height)
    print(f"   resize = min({resize}, {height}) = {resize}")
    
    # HERE'S THE CRITICAL CALCULATION:
    xl = projection.center_x - resize/2 + 1
    print(f"\n🔹 CRITICAL BUG ANALYSIS:")
    print(f"   xl = center_x - resize/2 + 1")
    print(f"   xl = {projection.center_x} - {resize}/2 + 1")
    print(f"   xl = {projection.center_x} - {resize/2} + 1")
    print(f"   xl = {projection.center_x - resize/2 + 1}")
    
    xl = 0 if xl < 0 else xl
    
    yt = projection.center_y - resize/2 + 1
    print(f"   yt = center_y - resize/2 + 1") 
    print(f"   yt = {projection.center_y} - {resize}/2 + 1")
    print(f"   yt = {projection.center_y} - {resize/2} + 1")
    print(f"   yt = {projection.center_y - resize/2 + 1}")
    
    yt = 0 if yt < 0 else yt
    
    xr = xl + resize - 1
    yb = yt + resize - 1
    
    print(f"\n🔹 FINAL CROP COORDINATES:")
    print(f"   xl = {xl}")
    print(f"   yt = {yt}")
    print(f"   xr = {xr}")
    print(f"   yb = {yb}")
    
    # Boundary checks
    if xr >= width - 1:
        print(f"   Adjusting for width boundary: {xr} >= {width - 1}")
        xl -= xr - width + 1
        xr = width - 1
        
    if yb >= height - 1:
        print(f"   Adjusting for height boundary: {yb} >= {height - 1}")
        yt -= yb - height + 1
        yb = height - 1
    
    final_crop = [int(xl), int(xr + 1), int(yt), int(yb + 1)]
    print(f"\n🔹 RETURN VALUE:")
    print(f"   [int(xl), int(xr + 1), int(yt), int(yb + 1)]")
    print(f"   [{int(xl)}, {int(xr + 1)}, {int(yt)}, {int(yb + 1)}]")
    print(f"   = {final_crop}")
    
    return final_crop

# Test with our actual projection
image_shape = (1080, 1920, 3)
projection_box = [511, 349, 761, 592, 0]

# Convert to ProjectionROI
x = projection_box[0]  # 511
y = projection_box[1]  # 349
w = projection_box[2] - projection_box[0]  # 250
h = projection_box[3] - projection_box[1]  # 243

projection = ProjectionROI(x, y, w, h)

crop_result = crop_debug(image_shape, projection)

print(f"\n🔹 COORDINATE OFFSET ANALYSIS:")
print(f"   Original projection center: ({projection.center_x}, {projection.center_y})")
print(f"   Crop region: [{crop_result[0]}, {crop_result[1]}, {crop_result[2]}, {crop_result[3]}]")
print(f"   Crop offset: xl={crop_result[0]}, yt={crop_result[2]}")

print(f"\n🔹 VERIFICATION:")
print(f"   Expected detection after restore_boxes_to_full_image should be:")
print(f"   detection_in_crop + [xl_offset, yt_offset, xl_offset, yt_offset]")
print(f"   detection_in_crop + [{crop_result[0]}, {crop_result[2]}, {crop_result[0]}, {crop_result[2]}]")

print(f"\n🔹 REVERSE CALCULATION:")
print(f"   Actual output detection: [466, 260, 489, 327]")
print(f"   Crop offset: [{crop_result[0]}, {crop_result[2]}] = [323, 158]")
print(f"   Detection in crop coordinates: [466-323, 260-158, 489-323, 327-158]")
print(f"   Detection in crop coordinates: [143, 102, 166, 169]")
print(f"   Size in crop: {166-143} x {169-102} = 23 x 67 pixels")

print(f"\n🔹 SCALING ANALYSIS:")
crop_size = 625
target_size = 270
scale_factor = target_size / crop_size  # 270 / 625 = 0.432
print(f"   Crop size: {crop_size} x {crop_size}")
print(f"   Target detector input: {target_size} x {target_size}")
print(f"   Scale factor: {scale_factor:.3f}")

print(f"\n🔹 THE BUG IS LIKELY IN COORDINATE SCALING!")
print(f"   When the crop is resized from {crop_size}x{crop_size} to {target_size}x{target_size},")
print(f"   the detector outputs coordinates in {target_size}x{target_size} space.")
print(f"   These need to be scaled back to {crop_size}x{crop_size} space")
print(f"   BEFORE being restored to full image coordinates!")

print(f"\n🔹 MISSING SCALING STEP:")
print(f"   Detection coordinates need to be multiplied by:")
print(f"   scale_factor_inverse = {crop_size} / {target_size} = {crop_size/target_size:.3f}")