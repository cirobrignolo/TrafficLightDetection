import torch
import torch.nn as nn
from tlr.tools.utils import nms

class RCNNProposal(nn.Module):
    def __init__(self, bbox_reg_param, detection_output_ssd_param, device=None):
        super(RCNNProposal, self).__init__()
        self.device = device
        self.bbox_mean = torch.tensor(bbox_reg_param['bbox_mean'], device=self.device)
        self.bbox_std = torch.tensor(bbox_reg_param['bbox_std'], device=self.device)
        self.num_class = detection_output_ssd_param['num_class']
        self.rpn_proposal_output_score = detection_output_ssd_param['rpn_proposal_output_score']
        self.regress_agnostic = detection_output_ssd_param['regress_agnostic']
        self.min_size_h = detection_output_ssd_param['min_size_h']
        self.min_size_w = detection_output_ssd_param['min_size_w']
        self.min_size_mode = detection_output_ssd_param['min_size_mode']
        self.threshold_objectness = detection_output_ssd_param['threshold_objectness']
        self.thresholds = detection_output_ssd_param['thresholds']
        self.refine_out_of_map_bbox = detection_output_ssd_param['refine_out_of_map_bbox']
        self.nms_param = detection_output_ssd_param['nms_param']

    def bbox_transform_inv_rcnn(self, boxes, deltas):
        if len(boxes.shape) == 4:
            boxes = boxes.squeeze(2)
            boxes = boxes.squeeze(2)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1)

        dx = deltas[:, :, 0]
        dy = deltas[:, :, 1]
        dw = deltas[:, :, 2]
        dh = deltas[:, :, 3]

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros(deltas.shape, dtype=deltas.dtype, device=self.device)
        # x1
        pred_boxes[:, :, 0] = pred_ctr_x - 0.5 * (pred_w - 1)
        # y1
        pred_boxes[:, :, 1] = pred_ctr_y - 0.5 * (pred_h - 1)
        # x2
        pred_boxes[:, :, 2] = pred_ctr_x + 0.5 * (pred_w - 1)
        # y2
        pred_boxes[:, :, 3] = pred_ctr_y + 0.5 * (pred_h - 1)
        return pred_boxes

    def clip_boxes(self, boxes, height, width):
        """
        Clip boxes to image boundaries.
        """
        clone = boxes.clone()
        boxes[:, 0] = torch.clamp(clone[:, 0], 0, width - 1)
        boxes[:, 1] = torch.clamp(clone[:, 1], 0, height - 1)
        boxes[:, 2] = torch.clamp(clone[:, 2], 0, width - 1)
        boxes[:, 3] = torch.clamp(clone[:, 3], 0, height - 1)
        return boxes

    def forward(self, cls_score_softmax, bbox_pred, rois, im_info):
        origin_height = im_info[0]
        origin_width = im_info[1]
        # normalize the rois
        bbox_pred = (bbox_pred.reshape(-1, 4) * self.bbox_std + self.bbox_mean).reshape(-1, self.num_class + 1, 4)

        # slice_rois
        sliced_rois = rois[:, 1:]
        
        # decode bbox
        decoded_bbox_pred = self.bbox_transform_inv_rcnn(sliced_rois, bbox_pred)

        if self.refine_out_of_map_bbox:
            decoded_bbox_pred = self.clip_boxes(decoded_bbox_pred, origin_height, origin_width)
        
        # filter by objectness
        # bbox_pred dims: [num_box, num_class+1, 4],
        # scores dims: [num_box, num_class+1],
        indices = 1 - cls_score_softmax[:,0] >= self.threshold_objectness
        cls_score_softmax = cls_score_softmax[indices]
        decoded_bbox_pred = decoded_bbox_pred[indices]

        maxes, argmaxes = torch.max(cls_score_softmax[:,1:], 1) # => max: n, argmax: n
        argmaxes += 1
        indices = maxes > self.thresholds[0]
        argmaxes = argmaxes[indices]
        maxes = maxes[indices]

        # simplified this step. In theory, 3 classes can have different threshold, but in the model definition, they are the same. 
        # So the simplification should not have any affects to the results
        cls_score_softmax = cls_score_softmax[indices]
        decoded_bbox_pred = decoded_bbox_pred[indices] # => (n, 4, 4)
        decoded_bbox_pred = decoded_bbox_pred[torch.arange(decoded_bbox_pred.shape[0]), argmaxes] # => (n, 4)

        w = decoded_bbox_pred[:, 2] - decoded_bbox_pred[:, 0] + 1
        h = decoded_bbox_pred[:, 3] - decoded_bbox_pred[:, 1] + 1

        if self.min_size_mode == "HEIGHT_OR_WIDTH":
            keep = (w >= self.min_size_w) + (h >= self.min_size_h)
        elif self.min_size_mode == "HEIGHT_AND_WIDTH":
            keep = (w >= self.min_size_w) * (h >= self.min_size_h)
        decoded_bbox_pred = decoded_bbox_pred[keep]
        cls_score_softmax = cls_score_softmax[keep]

        # keep max N candidates
        num_keep = min(maxes[keep].shape[0], self.nms_param['max_candidate_n'])
        top_indices = torch.topk(maxes[keep], num_keep).indices
        pre_nms_bbox = decoded_bbox_pred[top_indices]
        pre_nms_all_probs = cls_score_softmax[top_indices]

        argmaxes = torch.argmax(pre_nms_all_probs[:,1:], 1) + 1

        nms_indices = nms(pre_nms_bbox, self.nms_param['overlap_ratio'])
        boxes = pre_nms_bbox[nms_indices][:self.nms_param['top_n']]
        scores = pre_nms_all_probs[nms_indices][:self.nms_param['top_n']]
        return torch.hstack([torch.zeros((boxes.shape[0], 1), device=self.device), boxes, scores])
