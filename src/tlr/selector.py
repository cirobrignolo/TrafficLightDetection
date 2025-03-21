import math
import torch
from tlr.tools.utils import crop

def calc_2d_gaussian_score(p1, p2, sigma1, sigma2):
    return math.exp(-0.5 * ((p1[0] - p2[0]) * (p1[0] - p2[0]) / (sigma1 * sigma1) + (p1[1] - p2[1]) * (p1[1] - p2[1]) / (sigma2 * sigma2)))

def select_tls(ho, detections, projections, item_shape):
    """
    detections shape is [n, 9]
    return [n, 2], the first col is the idx of the ground truth, the second col is the idx of the valid detections
    """
    costs = torch.zeros([len(projections), len(detections)])
    if torch.numel(costs) == 0:
        return torch.empty([0, 2])
    for row, projection in enumerate(projections):
        center_hd = [projection.center_x, projection.center_y]
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
    assignments = ho.maximize(costs.detach().numpy())
    final_assignment1s = []
    final_assignment2s = []
    # check if the detection is inside the crop
    for assignment in assignments:
        # check if there is any double-assignment
        if assignment[0] in final_assignment1s or assignment[1] in final_assignment2s:
            continue
        # check if the assignment[0] is out-of-index of the projections
        # check if the assignment[1] is out-of-index of the detections
        if assignment[0] >= len(projections) or assignment[1] >= len(detections):
            continue
        # get the crop 
        coors = crop(item_shape, projections[assignment[0]]) # xmin, xmax, ymin, ymax
        # get the detection
        detection = detections[assignment[1]] # xmin, ymin, xmax, ymax
        # check if the detection is inside the crop
        if coors[0] > detection[1] or \
            coors[1] < detection[3] or \
            coors[2] > detection[2] or \
            coors[3] < detection[4]:
            continue
        final_assignment1s.append(assignment[0])
        final_assignment2s.append(assignment[1])
    return torch.stack([torch.tensor(final_assignment1s), torch.tensor(final_assignment2s)]).transpose(1, 0)