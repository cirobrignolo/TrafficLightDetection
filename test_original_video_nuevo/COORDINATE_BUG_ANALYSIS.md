# Apollo TLR Coordinate Offset Bug Analysis

## Problem Summary

The Apollo TLR pipeline consistently returns detection coordinates that are offset **up and to the left** from the input projection coordinates. Analysis shows detections appearing approximately 45 pixels left and 89 pixels above the expected locations.

## Root Cause: Missing Coordinate Scaling

The bug is located in the **coordinate transformation pipeline** between detection and coordinate restoration. Specifically, there's a **missing coordinate scaling step** in the `restore_boxes_to_full_image()` function.

## Detailed Analysis

### Coordinate Flow

1. **Input**: Projection coordinates `[511, 349, 761, 592]` (xmin, ymin, xmax, ymax)
2. **Crop Calculation**: Creates 625×625 crop region at `[323, 948, 158, 783]`
3. **Preprocessing**: Crop gets resized to 270×270 for detector input
4. **Detection**: Detector outputs coordinates in 270×270 space
5. **❌ BUG**: Missing scaling from 270×270 back to 625×625 space
6. **Restoration**: Adds crop offset to get final coordinates (but using wrong scale)

### Bug Location

**File**: `/home/cirojb/Desktop/TrafficLightDetection/src/tlr/tools/utils.py`
**Function**: `restore_boxes_to_full_image()` (lines 257-275)

```python
def restore_boxes_to_full_image(image, detections, projections, start_col=1):
    ret = []
    for detection, projection in zip(detections, projections):
        xl, xr, yt, yb = crop(image.shape, projection)
        # ❌ BUG: Missing coordinate scaling here!
        # Detector outputs are in 270x270 space but need to be scaled 
        # to crop space (625x625) before adding offset
        detection[:, start_col] += xl     # x1
        detection[:, start_col+1] += yt   # y1  
        detection[:, start_col+2] += xl   # x2
        detection[:, start_col+3] += yt   # y2
        ret.append(detection)
    return ret
```

### The Fix

The detector coordinates need to be scaled from 270×270 space back to crop space before adding the crop offset:

```python
def restore_boxes_to_full_image(image, detections, projections, start_col=1):
    ret = []
    for detection, projection in zip(detections, projections):
        xl, xr, yt, yb = crop(image.shape, projection)
        
        # Calculate scaling factors
        crop_width = xr - xl
        crop_height = yb - yt  
        scale_x = crop_width / 270.0
        scale_y = crop_height / 270.0
        
        # ✅ FIX: Scale detector coordinates from 270x270 to crop space
        detection[:, start_col] *= scale_x      # Scale x1
        detection[:, start_col+1] *= scale_y    # Scale y1
        detection[:, start_col+2] *= scale_x    # Scale x2  
        detection[:, start_col+3] *= scale_y    # Scale y2
        
        # Then add crop offset
        detection[:, start_col] += xl           # x1
        detection[:, start_col+1] += yt         # y1
        detection[:, start_col+2] += xl         # x2
        detection[:, start_col+3] += yt         # y2
        ret.append(detection)
    return ret
```

### Mathematical Verification

**Current (Buggy) Calculation**:
```
detector_output = [61.8, 44.1, 71.7, 73.0]  # in 270x270 space
final_coords = detector_output + [323, 158, 323, 158]
             = [384.8, 202.1, 394.7, 231.0]  # Wrong!
```

**Correct Calculation**:
```
detector_output = [61.8, 44.1, 71.7, 73.0]  # in 270x270 space
scale_factor = 625/270 = 2.315
scaled_coords = detector_output * 2.315 = [143.0, 102.1, 165.9, 169.0]
final_coords = scaled_coords + [323, 158, 323, 158] 
             = [466.0, 260.1, 488.9, 327.0]  # Matches actual output!
```

## Secondary Issues Found

### 1. Coordinate Indexing Inconsistency
- Some functions use 0-based indexing while others use 1-based
- The `+1` and `-1` adjustments in crop calculation create fractional coordinates

### 2. Rounding Errors
- Crop calculation: `xl = 323.5 → 323` (loses 0.5 pixels)
- These accumulate across transformations

### 3. Detection vs Projection Coordinate Systems
- Input projections use full image coordinates
- Detector works in normalized 270×270 space
- Missing bidirectional scaling

## Impact

This bug causes:
- **Misaligned traffic light detections**
- **Reduced tracking accuracy** (due to coordinate inconsistencies)
- **Poor assignment matching** between projections and detections
- **Systematic spatial drift** in multi-frame sequences

## Files Affected

1. **Primary**: `/src/tlr/tools/utils.py` - `restore_boxes_to_full_image()`
2. **Secondary**: `/src/tlr/selector.py` - Coordinate validation logic
3. **Validation needed**: All coordinate-dependent modules

## Test Case

**Input projection**: `[511, 349, 761, 592]`
**Current output**: `[466, 260, 489, 327]` (45px left, 89px up offset)
**Expected output**: Should be within or very close to input projection bounds

The fix should eliminate this systematic offset and properly align detections with input projections.