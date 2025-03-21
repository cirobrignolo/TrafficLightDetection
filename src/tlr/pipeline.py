import torch
import torch.nn as nn
from tlr.detector import TLDetector
from tlr.recognizer import Recognizer
from tlr.hungarian_optimizer import HungarianOptimizer
from tlr.tools.utils import preprocess4det, preprocess4rec, restore_boxes_to_full_image, nms, boxes2projections
from tlr.selector import select_tls
import json
import os

class Pipeline(nn.Module):
    """
    This class will be responsible for detecting and recognizing a single ROI.
    """
    def __init__(self, detector, classifiers, ho, means_det, means_rec, device=None):
        super().__init__()
        self.detector = detector
        self.classifiers = classifiers
        self.means_det = means_det
        self.means_rec = means_rec
        self.ho = ho
        self.device = device
    def detect(self, image, boxes):
        """bboxes should be a list of list, each sub-list is like [xmin, ymin, xmax, ymax]"""
        detected_boxes = []
        projections = boxes2projections(boxes)
        for projection in projections:
            input = preprocess4det(image, projection, self.means_det)
            bboxes = self.detector(input.unsqueeze(0).permute(0, 3, 1, 2))
            detected_boxes.append(bboxes)
        detections = restore_boxes_to_full_image(image, detected_boxes, projections)
        detections = torch.vstack(detections).reshape(-1, 9)
        idxs = nms(detections[:, 1:5], 0.6)
        detections = detections[idxs]
        return detections
    def recognize(self, img, detections, tl_types):
        recognitions = []
        for detection, tl_type in zip(detections, tl_types):
            det_box = detection[1:5].type(torch.long)
            recognizer, shape = self.classifiers[tl_type-1]
            input = preprocess4rec(img, det_box, shape, self.means_rec)
            output = recognizer(input.permute(2, 0, 1).unsqueeze(0))
            assert output.shape[0] == 1
            recognitions.append(output[0])
        return torch.vstack(recognitions).reshape(-1, 4)
    def forward(self, img, boxes):
        """img should not substract the means, if there's a perturbation, the perturbation should be added to the img
        return valid_detections, recognitions, assignments, invalid_detections
        """
        if len(boxes) == 0:
            return torch.empty((0, 9), device=self.device), \
                torch.empty((0, 4), device=self.device), \
                torch.empty((0, 2), device=self.device), \
                torch.empty((0, 9), device=self.device)
        detections = self.detect(img, boxes)
        tl_types = torch.argmax(detections[:, 5:], dim=1)
        valid_inds = tl_types != 0
        valid_detections = detections[valid_inds]
        invalid_detections = detections[~valid_inds]
        assignments = select_tls(self.ho, valid_detections, boxes2projections(boxes), img.shape).to(self.device)
        # Baidu Apollo only recognize the selected TLs, we recognize all valid detections.
        if len(valid_detections) != 0:
            recognitions = self.recognize(img, valid_detections, tl_types[valid_inds])
        else:
            recognitions = torch.empty((0, 4), device=self.device)
        return valid_detections, recognitions, assignments, invalid_detections

def load_pipeline(device=None):
    DIR = os.path.dirname(__file__)
    print(f'Loaded the TL pipeline. Device is {device}.')
    means_det = torch.Tensor([102.9801, 115.9465, 122.7717]).to(device)
    means_rec = torch.Tensor([69.06, 66.58, 66.56]).to(device)

    with open(f'{DIR}/confs/bbox_reg_param.json', 'r') as f:
        bbox_reg_param = json.load(f)
    with open(f'{DIR}/confs/detection_output_ssd_param.json', 'r') as f:
        detection_output_ssd_param = json.load(f)
    with open(f'{DIR}/confs/dfmb_psroi_pooling_param.json', 'r') as f:
        dfmb_psroi_pooling_param = json.load(f)
    with open(f'{DIR}/confs/rcnn_bbox_reg_param.json', 'r') as f:
        rcnn_bbox_reg_param = json.load(f)
    with open(f'{DIR}/confs/rcnn_detection_output_ssd_param.json', 'r') as f:
        rcnn_detection_output_ssd_param = json.load(f)
    im_info = [270, 270]

    detector = TLDetector(bbox_reg_param, detection_output_ssd_param, dfmb_psroi_pooling_param, rcnn_bbox_reg_param, rcnn_detection_output_ssd_param, im_info, device=device)
    detector.load_state_dict(torch.load(f'{DIR}/weights/tl.torch'))
    detector = detector.to(device)
    detector.eval();

    quad_pool_params = {'kernel_size': (4, 4), 'stride': (4, 4)}
    hori_pool_params = {'kernel_size': (2, 6), 'stride': (2, 6)}
    vert_pool_params = {'kernel_size': (6, 2), 'stride': (6, 2)}
    quad_recognizer = Recognizer(quad_pool_params)
    hori_recognizer = Recognizer(hori_pool_params)
    vert_recognizer = Recognizer(vert_pool_params)

    quad_recognizer.load_state_dict(torch.load(f'{DIR}/weights/quad.torch'))
    quad_recognizer = quad_recognizer.to(device)
    quad_recognizer.eval();

    hori_recognizer.load_state_dict(torch.load(f'{DIR}/weights/hori.torch'))
    hori_recognizer = hori_recognizer.to(device)
    hori_recognizer.eval();

    vert_recognizer.load_state_dict(torch.load(f'{DIR}/weights/vert.torch'))
    vert_recognizer = vert_recognizer.to(device)
    vert_recognizer.eval();
    classifiers = [(vert_recognizer, (96, 32, 3)), (quad_recognizer, (64, 64, 3)), (hori_recognizer, (32, 96, 3))]

    ho = HungarianOptimizer()
    pipeline = Pipeline(detector, classifiers, ho, means_det, means_rec, device=device)
    return pipeline