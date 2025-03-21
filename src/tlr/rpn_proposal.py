import torch
import torch.nn as nn
import torch.nn.functional as F
from tlr.tools.utils import nms
# detection
class RPNProposalSSD(nn.Module):
    def __init__(self, bbox_reg_param, detection_output_ssd_param, device=None):
        super(RPNProposalSSD, self).__init__()
        self.device = device
        # read box params
        self.bbox_mean = torch.tensor(bbox_reg_param['bbox_mean'], device=self.device)
        self.bbox_std = torch.tensor(bbox_reg_param['bbox_std'], device=self.device)
        # read detection param
        self.anchor_stride = detection_output_ssd_param['heat_map_a']
        self.gen_anchor_param = detection_output_ssd_param['gen_anchor_param']
        self.num_anchor_per_point = len(self.gen_anchor_param['anchor_widths'])
        self.min_size_mode = detection_output_ssd_param['min_size_mode']
        self.min_size_h = detection_output_ssd_param['min_size_h']
        self.min_size_w = detection_output_ssd_param['min_size_w']
        self.threshold_objectness = detection_output_ssd_param['threshold_objectness']
        self.nms_param = detection_output_ssd_param['nms_param']
        self.refine_out_of_map_bbox = detection_output_ssd_param['refine_out_of_map_bbox']

    def generate_anchors(self):
        """
        anchor is represented by 4 pts (xmin, ymin, xmax, ymax)
        """
        anchor_widths = torch.tensor(self.gen_anchor_param['anchor_widths'], device=self.device)
        anchor_heights = torch.tensor(self.gen_anchor_param['anchor_heights'], device=self.device)
        xmins = - 0.5 * (anchor_widths - 1)
        xmaxs = + 0.5 * (anchor_widths - 1)
        ymins = - 0.5 * (anchor_heights - 1)
        ymaxs = + 0.5 * (anchor_heights - 1)
        anchors = torch.vstack((xmins, ymins, xmaxs, ymaxs)).transpose(1, 0)
        return anchors

    def bbox_transform_inv(self, boxes, deltas):
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1)

        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes = torch.zeros(deltas.shape, dtype=deltas.dtype, device=self.device)
        # x1
        pred_boxes[:, 0] = pred_ctr_x - 0.5 * (pred_w - 1)
        # y1
        pred_boxes[:, 1] = pred_ctr_y - 0.5 * (pred_h - 1)
        # x2
        pred_boxes[:, 2] = pred_ctr_x + 0.5 * (pred_w - 1)
        # y2
        pred_boxes[:, 3] = pred_ctr_y + 0.5 * (pred_h - 1)
        return pred_boxes

    def clip_boxes(self, boxes, height, width):
        """
        Clip boxes to image boundaries.
        """
        clone = boxes.clone()
        boxes[:, 0::4] = torch.clamp(clone[:, 0::4], 0, width - 1)
        boxes[:, 1::4] = torch.clamp(clone[:, 1::4], 0, height - 1)
        boxes[:, 2::4] = torch.clamp(clone[:, 2::4], 0, width - 1)
        boxes[:, 3::4] = torch.clamp(clone[:, 3::4], 0, height - 1)
        return boxes

    def filter_boxes(self, proposals, scores, num_box, num_class, filter_class, min_size_mode, min_size_h, min_size_w, threshold_score):
        # filter cases whose scores are below the threshold
        keep = scores[:, filter_class] > threshold_score
        proposals = proposals[keep]
        scores = scores[keep]

        # filter out cases whose widths and heights are lower than the min_size_w/h
        ws = proposals[:, 2] - proposals[:, 0] + 1
        hs = proposals[:, 3] - proposals[:, 1] + 1
        assert torch.all(ws >= 0) and torch.all(hs >= 0)

        if min_size_mode == 'HEIGHT_AND_WIDTH':
            keep = ws >= min_size_w
            keep *= hs >= min_size_h # get the && of boolean
        elif min_size_mode == 'HEIGHT_OR_WIDTH':
            keep = ws >= min_size_w
            keep += hs >= min_size_h # get the || of boolean
        else:
            raise
        return proposals[keep], scores[keep]

    def forward(self, rpn_cls_prob_reshape, rpn_bbox_pred, im_info):
        """
        rpn_cls_prob_reshape has a shape of [N, 2 * num_anchor_per_point, W, H]
        rpn_bbox_pred        has a shape of [N, 4 * num_anchor_per_point, W, H]
        im_info (origin_width, origin_height, )
        part of the implementation refers to https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/proposal_layer.py#L47
        """
        assert rpn_cls_prob_reshape.shape[0] == 1 # only support batch=1
        origin_height = im_info[0]
        origin_width  = im_info[1]
        height = rpn_cls_prob_reshape.shape[-2]
        width  = rpn_cls_prob_reshape.shape[-1]
        num_anchor = self.num_anchor_per_point * height * width
        anchor_size = num_anchor * 4

        # Enumerate all shifts
        shift_x = torch.arange(width, device=self.device) * self.anchor_stride
        shift_y = torch.arange(height, device=self.device) * self.anchor_stride
        shift_x, shift_y = torch.meshgrid(shift_x, shift_y, indexing='ij')
        shifts = torch.vstack((shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel())).transpose(1, 0)
        
        anchors = self.generate_anchors()
        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        anchors = anchors.unsqueeze(0) + shifts.unsqueeze(1)
        anchors = anchors.reshape((-1, 4))

        # reshape the predicted regressors to (W * H * num_anchor_per_point, 4)
        rpn_bbox_pred = rpn_bbox_pred.reshape(self.num_anchor_per_point, 4, 34, 34).permute(2, 3, 0, 1).reshape(-1, 4)
        rpn_bbox_pred = rpn_bbox_pred * self.bbox_std # multiply the 4 std with each row (i.e., the 4 regressors)
        rpn_bbox_pred = rpn_bbox_pred + self.bbox_mean # add the 4 mean to each row (i.e., the 4 regressors)

        # Convert anchors into proposals via bbox transformations
        # print(anchors.shape, rpn_bbox_pred.shape)
        proposals = self.bbox_transform_inv(anchors, rpn_bbox_pred)
        # clip boxes, i.e. refine proposals which are out of map
        if self.refine_out_of_map_bbox:
            proposals = self.clip_boxes(proposals, origin_height, origin_width)

        # reshape scores
        scores = rpn_cls_prob_reshape.reshape(2, self.num_anchor_per_point, -1).permute(2, 1, 0).reshape(-1, 2)

        proposals, scores = self.filter_boxes(proposals, scores, num_anchor, 2, 1, self.min_size_mode, self.min_size_h, self.min_size_w, self.threshold_objectness)

        # keep max N candidates
        top_indices = torch.topk(scores[:, 1], min(scores.shape[0], self.nms_param['max_candidate_n'])).indices
        proposals = proposals[top_indices]

        # apply NMS
        nms_indices = nms(proposals, self.nms_param['overlap_ratio'])
        proposals = proposals[nms_indices][:self.nms_param['top_n']]
        proposals = torch.hstack([torch.zeros((proposals.shape[0], 1), device=self.device), proposals])
        return proposals
