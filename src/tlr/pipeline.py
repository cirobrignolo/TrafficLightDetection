import torch
import torch.nn as nn
from tlr.detector import TLDetector
from tlr.recognizer import Recognizer
from tlr.hungarian_optimizer import HungarianOptimizer
from tlr.tools.utils import preprocess4det, preprocess4rec, restore_boxes_to_full_image, nms, boxes2projections
from tlr.selector import select_tls
from tlr.tracking import TrafficLightTracker, REVISE_TIME_S, BLINK_THRESHOLD_S, HYSTERETIC_THRESHOLD_COUNT
import json
import os

class Pipeline(nn.Module):
    """
    This class will be responsible for detecting and recognizing a single ROI.
    """
    def __init__(self, detector, classifiers, ho, means_det, means_rec, device=None, tracker=None):
        super().__init__()
        self.detector = detector
        self.classifiers = classifiers
        self.means_det = means_det
        self.means_rec = means_rec
        self.ho = ho
        self.device = device
        self.tracker = tracker

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
        idxs = nms(detections[:, 1:5], 0.7)
        detections = detections[idxs]
        return detections

    def recognize(self, img, detections, tl_types):
        recognitions = []
        for detection, tl_type in zip(detections, tl_types):
            det_box = detection[1:5].type(torch.long)
            recognizer, shape = self.classifiers[tl_type-1]
            input = preprocess4rec(img, det_box, shape, self.means_rec)
            output = recognizer(input.permute(2, 0, 1).unsqueeze(0))
            # APOLLO'S CONFIDENCE THRESHOLD
            max_prob = torch.max(output[0])
            if max_prob < 0.5:  # Apollo's threshold
                # Force to BLACK (index 0)
                forced_output = torch.zeros_like(output[0])
                forced_output[0] = 1.0  # BLACK = 100% confidence
                recognitions.append(forced_output)
            else:
                recognitions.append(output[0])
        return torch.vstack(recognitions).reshape(-1, 4) if recognitions else torch.empty((0, 4), device=self.device)

    def forward(self, img, boxes, frame_ts=None):
        """img should not substract the means, if there's a perturbation, the perturbation should be added to the img
        return valid_detections, recognitions, assignments, invalid_detections
        """
        """
        :param img: Tensor [C,H,W]
        :param boxes: lista de [x1,y1,x2,y2,id]
        :param frame_ts: timestamp en segundos (float) para el tracker
        :returns:
            valid_detections (Tensor n×9),
            recognitions      (Tensor n×4),
            assignments       (Tensor m×2),
            invalid_detections(Tensor k×9),
            revised_states    (dict proj_id → (color, blink))  # si tracker no es None
        """
        # 1) Early exit si no hay cajas
        if len(boxes) == 0:
            empty9 = torch.empty((0, 9), device=self.device)
            empty4 = torch.empty((0, 4), device=self.device)
            empty2 = torch.empty((0, 2), device=self.device)
            revised = {} if self.tracker else None
            return empty9, empty4, empty2, empty9, revised

        # 2) Detección
        detections = self.detect(img, boxes)

        # 3) Filtrado por tipo y asignación
        tl_types = torch.argmax(detections[:, 5:], dim=1)
        valid_mask = tl_types != 0
        valid_detections = detections[valid_mask]
        invalid_detections = detections[~valid_mask]
        assignments = select_tls(self.ho, valid_detections, boxes2projections(boxes), img.shape).to(self.device)

        # 4) Reconocimiento
        # Baidu Apollo only recognize the selected TLs, we recognize all valid detections.
        if len(valid_detections) != 0:
            recognitions = self.recognize(img, valid_detections, tl_types[valid_mask])
        else:
            recognitions = torch.empty((0, 4), device=self.device)

        # 5) TRACKING / REVISION TEMPORAL
        revised = None
        if self.tracker:
            if frame_ts is None:
                raise ValueError("Para usar tracking debes pasar frame_ts")
            # assignments es tensor m×2; recognitions es tensor n×4
            # convertimos a listas de Python para el tracker
            assigns_list = assignments.cpu().tolist()
            recs_list    = recognitions.cpu().tolist()
            revised = self.tracker.track(frame_ts, assigns_list, recs_list)

        return valid_detections, recognitions, assignments, invalid_detections, revised

        #return valid_detections, recognitions, assignments, invalid_detections

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
    detector.load_state_dict(torch.load(f'{DIR}/weights/tl.torch', weights_only=False))
    detector = detector.to(device)
    detector.eval();

    quad_pool_params = {'kernel_size': (4, 4), 'stride': (4, 4)}
    hori_pool_params = {'kernel_size': (2, 6), 'stride': (2, 6)}
    vert_pool_params = {'kernel_size': (6, 2), 'stride': (6, 2)}
    quad_recognizer = Recognizer(quad_pool_params)
    hori_recognizer = Recognizer(hori_pool_params)
    vert_recognizer = Recognizer(vert_pool_params)

    quad_recognizer.load_state_dict(torch.load(f'{DIR}/weights/quad.torch', weights_only=False))
    quad_recognizer = quad_recognizer.to(device)
    quad_recognizer.eval();

    hori_recognizer.load_state_dict(torch.load(f'{DIR}/weights/hori.torch', weights_only=False))
    hori_recognizer = hori_recognizer.to(device)
    hori_recognizer.eval();

    vert_recognizer.load_state_dict(torch.load(f'{DIR}/weights/vert.torch', weights_only=False))
    vert_recognizer = vert_recognizer.to(device)
    vert_recognizer.eval();
    classifiers = [(vert_recognizer, (96, 32, 3)), (quad_recognizer, (64, 64, 3)), (hori_recognizer, (32, 96, 3))]

    ho = HungarianOptimizer()

    tracker = TrafficLightTracker(
        revise_time_s=REVISE_TIME_S,
        blink_threshold_s=BLINK_THRESHOLD_S,
        hysteretic_threshold=HYSTERETIC_THRESHOLD_COUNT
    )
    pipeline = Pipeline(detector, classifiers, ho, means_det, means_rec, device=device, tracker=tracker)

    return pipeline