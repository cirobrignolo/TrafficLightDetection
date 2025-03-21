import torch.nn as nn
import torch.nn.functional as F
from tlr.rpn_proposal import RPNProposalSSD
from tlr.dfmb_roi_align import DFMBPSROIAlign
from tlr.faster_rcnn import RCNNProposal
from tlr.feature_net import FeatureNet

class TLDetector(nn.Module):
    """
    The entire network for traffic light detection.
    """
    def __init__(self, bbox_reg_param, detection_output_ssd_param, dfmb_psroi_pooling_param, rcnn_bbox_reg_param, rcnn_detection_output_ssd_param, im_info, device=None):
        super().__init__()
        self.device = device
        self.feature_net = FeatureNet().to(device)
        self.proposal = RPNProposalSSD(bbox_reg_param, detection_output_ssd_param, device=self.device)
        self.psroi_rois = DFMBPSROIAlign(dfmb_psroi_pooling_param, device=self.device)
        self.inner_rois = nn.Linear(in_features=10 * 7 * 7, out_features=2048, bias=True, device=self.device)
        self.cls_score = nn.Linear(in_features=2048, out_features=4, bias=True, device=self.device)
        self.bbox_pred = nn.Linear(in_features=2048, out_features=16, bias=True, device=self.device)
        self.rcnn_proposal = RCNNProposal(rcnn_bbox_reg_param, rcnn_detection_output_ssd_param, device=self.device)
        self.im_info = im_info

    def forward(self, x):
        rpn_cls_prob_reshape, rpn_bbox_pred, ft_add_left_right = self.feature_net(x)
        # print(rpn_cls_prob_reshape.shape, rpn_bbox_pred.shape, ft_add_left_right.shape)
        rois = self.proposal(rpn_cls_prob_reshape, rpn_bbox_pred, self.im_info)
        # print(rois.shape)
        psroi_rois = self.psroi_rois(ft_add_left_right, rois)
        # print(psroi_rois.shape)
        inner_rois = F.relu(self.inner_rois(psroi_rois.reshape(-1, 490)))
        # print(inner_rois.shape)
        cls_score = self.cls_score(inner_rois)
        bbox_pred = self.bbox_pred(inner_rois)
        cls_score_softmax = F.softmax(cls_score, dim=1)
        # print(cls_score.shape, cls_score_softmax.shape, bbox_pred.shape)
        bboxes = self.rcnn_proposal(cls_score_softmax, bbox_pred, rois, self.im_info)
        # print(bboxes.shape)

        return bboxes

# torch.Size([1, 30, 34, 34]) torch.Size([1, 60, 34, 34]) torch.Size([1, 490, 34, 34])
# torch.Size([52, 5])
# torch.Size([52, 10, 49])
# torch.Size([52, 2048])
# torch.Size([52, 4]) torch.Size([52, 4]) torch.Size([52, 16])
# torch.Size([3, 9])