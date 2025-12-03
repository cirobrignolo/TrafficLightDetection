import math
import torch
from tlr.tools.utils import crop

def calc_2d_gaussian_score(p1, p2, sigma1, sigma2):
    return math.exp(-0.5 * ((p1[0] - p2[0]) * (p1[0] - p2[0]) / (sigma1 * sigma1) + (p1[1] - p2[1]) * (p1[1] - p2[1]) / (sigma2 * sigma2)))

def select_tls(ho, detections, projections, item_shape):
    """
    Apollo-style traffic light selection using Hungarian algorithm.

    detections shape is [n, 9]: [score, xmin, ymin, xmax, ymax, class_probs...]
    return [n, 2], the first col is the idx of the projection, the second col is the idx of the detection
    """
    costs = torch.zeros([len(projections), len(detections)])
    if torch.numel(costs) == 0:
        return torch.empty([0, 2])

    for row, projection in enumerate(projections):
        center_hd = [projection.center_x, projection.center_y]
        # Pre-compute crop ROI for this projection (Apollo does this)
        coors = crop(item_shape, projection)  # xmin, xmax, ymin, ymax

        for col, detection in enumerate(detections):
            gaussian_score = 100.0
            center_refine = [int((detection[3] + detection[1])/2), int((detection[4] + detection[2])/2)]
            distance_score = calc_2d_gaussian_score(center_hd, center_refine, gaussian_score, gaussian_score)

            max_score = 0.9
            detect_score = torch.max(detection[5:])
            detection_score = max_score if detect_score > max_score else detect_score

            distance_weight = 0.7
            detection_weight = 1 - distance_weight
            costs[row, col] = detection_weight * detection_score + distance_weight * distance_score

            # APOLLO FIX: Validate detection is inside crop ROI BEFORE Hungarian
            # Apollo's logic from select.cc:76-83
            det_box = detection[1:5]  # xmin, ymin, xmax, ymax
            # Check if detection is outside crop_roi → set score to 0
            if coors[0] > det_box[0] or \
                coors[1] < det_box[2] or \
                coors[2] > det_box[1] or \
                coors[3] < det_box[3]:
                costs[row, col] = 0.0

    assignments = ho.maximize(costs.detach().numpy())

    # Simplified post-processing (validation already done in cost matrix)
    final_assignment1s = []
    final_assignment2s = []

    for assignment in assignments:
        proj_idx, det_idx = assignment[0], assignment[1]

        # Check for duplicates and out-of-bounds
        if proj_idx in final_assignment1s or det_idx in final_assignment2s:
            continue
        if proj_idx >= len(projections) or det_idx >= len(detections):
            continue

        final_assignment1s.append(proj_idx)
        final_assignment2s.append(det_idx)

    if not final_assignment1s:
        return torch.empty([0, 2])

    return torch.stack([torch.tensor(final_assignment1s), torch.tensor(final_assignment2s)]).transpose(1, 0)