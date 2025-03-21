import torch
import torch.nn as nn
import torch.nn.functional as F

class DFMBPSROIAlign(nn.Module):
    def __init__(self, dfmb_psroi_pooling_param, device=None):
        super(DFMBPSROIAlign, self).__init__()
        self.device = device
        self.pooled_height = dfmb_psroi_pooling_param['pooled_height']
        self.pooled_width = dfmb_psroi_pooling_param['pooled_width']
        self.anchor_stride = dfmb_psroi_pooling_param['heat_map_a']
        self.sample_per_part = dfmb_psroi_pooling_param['sample_per_part']

        self.channels = 10
        self.width = 34
        self.height = 34

    def forward(self, ft_add_left_right, rois):
        """
        compute the ROI area on the feature map 
        refers to https://github.com/ApolloAuto/apollo/blob/v7.0.0/modules/perception/inference/tensorrt/plugins/dfmb_psroi_align_plugin.cu 
        and https://erdem.pl/2020/02/understanding-region-of-interest-part-2-ro-i-align
        """
        if len(rois.shape) == 4:
            rois = rois.squeeze(2)
            rois = rois.squeeze(2)
        ft_add_left_right = ft_add_left_right[0].reshape(self.channels, self.pooled_height, self.pooled_width, self.height, self.width)
        
        # ROI positions. In the original code, it has calculations with pad_w/h and heat_map_b. 
        # Not sure what they are, there values should be 0 by my analysis, so just ignore them.
        roi_start_w = rois[:,1] / self.anchor_stride
        roi_start_h = rois[:,2] / self.anchor_stride
        roi_end_w   = rois[:,3] / self.anchor_stride
        roi_end_h   = rois[:,4] / self.anchor_stride
        
        roi_height = roi_end_h - roi_start_h
        roi_width  = roi_end_w - roi_start_w
        roi_height = torch.threshold(roi_height, 0.1, 0.1)
        roi_width  = torch.threshold(roi_width,  0.1, 0.1)

        bin_size_h = roi_height / self.pooled_height
        bin_size_w = roi_width  / self.pooled_width
        sub_bin_size_h = bin_size_h / self.sample_per_part
        sub_bin_size_w = bin_size_w / self.sample_per_part

        grid = torch.meshgrid(torch.arange(self.pooled_height, device=self.device), torch.arange(self.pooled_width, device=self.device), indexing='ij')
        phs = grid[0].reshape(-1, 1) # => 49, 1
        pws = grid[1].reshape(-1, 1) # => 49, 1

        hstart = torch.floor(roi_start_h + phs * bin_size_h) # => 49, n
        wstart = torch.floor(roi_start_w + pws * bin_size_w) # => 49, n

        sum_ = torch.zeros((self.channels, self.pooled_height * self.pooled_width, rois.shape[0]), device=self.device)
        count = torch.zeros((self.pooled_height * self.pooled_width, rois.shape[0]), device=self.device)

        for ih in range(self.sample_per_part):
            for iw in range(self.sample_per_part):
                # w and h are the samples
                w = wstart + (iw + 0.5) * sub_bin_size_w
                h = hstart + (ih + 0.5) * sub_bin_size_h

                keep = (w > -1) * (w < self.width) * (h > -1) * (h < self.height)

                # bilinear interpolation
                x1 = torch.floor(w).to(torch.long)
                x2 = torch.ceil(w).to(torch.long)
                y1 = torch.floor(h).to(torch.long)
                y2 = torch.ceil(h).to(torch.long)
                x1valid = (x1 >= 0) * (x1 < self.width)
                x2valid = (x2 >= 0) * (x2 < self.width)
                y1valid = (y1 >= 0) * (y1 < self.height)
                y2valid = (y2 >= 0) * (y2 < self.height)

                x1 = torch.clamp(x1, 0, 33)
                x2 = torch.clamp(x2, 0, 33)
                y1 = torch.clamp(y1, 0, 33)
                y2 = torch.clamp(y2, 0, 33)

                dist_x = w - x1 # => 49, n
                dist_y = h - y1 # => 49, n

                value11 = ft_add_left_right[:, phs, pws, y1, x1] * x1valid * y1valid # => 10, 49, n
                value12 = ft_add_left_right[:, phs, pws, y2, x1] * x1valid * y2valid # => 10, 49, n
                value21 = ft_add_left_right[:, phs, pws, y1, x2] * x2valid * y1valid # => 10, 49, n
                value22 = ft_add_left_right[:, phs, pws, y2, x2] * x2valid * y2valid # => 10, 49, n

                value = (1 - dist_x) * (1 - dist_y) * value11 \
                        + (1 - dist_x) * dist_y * value12 \
                        + dist_x * (1 - dist_y) * value21 \
                        + dist_x * dist_y * value22 # => 10, 49, n
                sum_[:, keep] += value[:, keep]
                count[keep] += 1
        nonzero = count > 0
        sum_[:, nonzero] /= count[nonzero]
        return sum_.permute(2, 0, 1)