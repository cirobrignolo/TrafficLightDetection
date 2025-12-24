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

        # APOLLO FIX: Sort by score BEFORE NMS (like Apollo does in detection.cc:381-390)
        # NOTA: Este detector no tiene detect_score en [:, 0] (siempre es 0)
        # Usamos el mismo criterio que selector.py: max de [bg, vert, quad, hori]
        scores = torch.max(detections[:, 5:9], dim=1).values  # Max score de clasificación
        sorted_indices = torch.argsort(scores, descending=True)  # Sort descending (highest first)
        detections_sorted = detections[sorted_indices]

        # Apply NMS on sorted detections
        # APOLLO FIX: Use threshold 0.6 like Apollo (detection.h:87: iou_thresh = 0.6)
        idxs = nms(detections_sorted[:, 1:5], 0.6)
        detections = detections_sorted[idxs]

        # Validación de tamaño de detecciones (Apollo-style)
        # Apollo rechaza detecciones muy grandes (>300px) o muy chicas (<5px)
        # Esto filtra falsos positivos (edificios rojos, ruido)
        if len(detections) > 0:
            MIN_SIZE = 5
            MAX_SIZE = 300
            MIN_ASPECT = 0.5
            MAX_ASPECT = 8.0

            valid_mask = torch.ones(len(detections), dtype=torch.bool, device=detections.device)

            for i, det in enumerate(detections):
                w = det[3] - det[1]  # xmax - xmin
                h = det[4] - det[2]  # ymax - ymin

                # Validar tamaño
                if w < MIN_SIZE or h < MIN_SIZE or w > MAX_SIZE or h > MAX_SIZE:
                    valid_mask[i] = False
                    continue

                # Validar aspect ratio (evita detecciones muy elongadas)
                aspect = h / w if w > 0 else 0
                if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
                    valid_mask[i] = False

            detections = detections[valid_mask]

        return detections

    def recognize(self, img, detections, tl_types):
        """
        Recognition with EXACT Apollo Prob2Color logic
        Apollo status_map: {BLACK=0, RED=1, YELLOW=2, GREEN=3}
        """
        recognitions = []
        # status_names = ['BLACK', 'RED', 'YELLOW', 'GREEN']  # For debugging
        
        for detection, tl_type in zip(detections, tl_types):
            det_box = detection[1:5].type(torch.long)
            recognizer, shape = self.classifiers[tl_type-1]
            input = preprocess4rec(img, det_box, shape, self.means_rec)
            
            # Apollo preprocessing: subtract means and apply scale
            input_scaled = input.permute(2, 0, 1).unsqueeze(0)  # NCHW format
            # Apollo uses scale=0.01 after mean subtraction
            input_scaled = input_scaled * 0.01
            
            # Get raw probabilities from model
            output_probs = recognizer(input_scaled)[0]  # [4]
            
            # Apollo's EXACT Prob2Color logic
            max_prob, max_idx = torch.max(output_probs, dim=0)
            threshold = 0.5  # Apollo's classify_threshold
            
            # Apollo's decision: if max_prob > threshold use max_idx, else force BLACK (0)
            if max_prob > threshold:
                color_id = max_idx.item()
            else:
                color_id = 0  # Force to BLACK like Apollo does
                
            # Create one-hot result (Apollo style)
            result = torch.zeros_like(output_probs)
            result[color_id] = 1.0
            
            # Apollo-style logging (uncomment for debug)
            # print(f"Light status recognized as {status_names[color_id]}")
            # print(f"Color Prob: {output_probs.tolist()}")
            # print(f"Max prob: {max_prob:.4f}, Threshold: {threshold}")
            
            recognitions.append(result)
            
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
            revised_states    (dict signal_id → (color, blink))  # si tracker no es None
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

        # 2.1) Filtro de confidence (Apollo-style adaptado)
        # Este detector no tiene detect_score en [:, 0], usamos max de scores de clasificación
        # Mismo criterio que selector.py y el ordenamiento en detect()
        MIN_CONFIDENCE = 0.3
        if len(detections) > 0:
            # Calcular score como máximo de [bg, vert, quad, hori]
            confidence_scores = torch.max(detections[:, 5:9], dim=1).values
            confidence_mask = confidence_scores >= MIN_CONFIDENCE
            detections = detections[confidence_mask]

        # 3) Filtrado por tipo y asignación
        if len(detections) > 0:
            tl_types = torch.argmax(detections[:, 5:], dim=1)
            valid_mask = tl_types != 0
            valid_detections = detections[valid_mask]
            invalid_detections = detections[~valid_mask]
        else:
            tl_types = torch.empty(0, dtype=torch.long, device=self.device)
            valid_detections = torch.empty((0, 9), device=self.device)
            invalid_detections = torch.empty((0, 9), device=self.device)

        # Calcular projections UNA sola vez (reutilizar en selector y tracker)
        projections = boxes2projections(boxes)
        assignments = select_tls(self.ho, valid_detections, projections, img.shape).to(self.device)

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

            # NUEVO: Pasar projections para que tracker acceda a signal_id
            revised = self.tracker.track(frame_ts, assigns_list, recs_list, projections)

        return valid_detections, recognitions, assignments, invalid_detections, revised

        #return valid_detections, recognitions, assignments, invalid_detections

def load_pipeline(device=None):
    DIR = os.path.dirname(__file__)
    print(f'Loaded the TL pipeline. Device is {device}.')
    means_det = torch.Tensor([102.9801, 115.9465, 122.7717]).to(device)
    # Apollo recognition.pb.txt: mean RGB = (69.06, 66.58, 66.56)
    # Pero cv2.imread() devuelve BGR, entonces invertimos el orden:
    means_rec = torch.Tensor([66.56, 66.58, 69.06]).to(device)  # BGR order

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