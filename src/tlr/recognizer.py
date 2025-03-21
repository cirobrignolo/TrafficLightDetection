import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

class ConvBNScale4Rec(nn.Module):
    """
    This is a very common sub-structure in apollo's network: Convolution -> BatchNorm -> Scale.
    Note that there are inconsistencies between the Caffe's BatchNorm and standard BatchNorm. 
    Caffe's BatchNorm ->  Scale is similar to BatchNorm in PyTorch https://github.com/BVLC/caffe/blob/master/include/caffe/layers/batch_norm_layer.hpp#L28.
    Besides, Caffe's BatchNorm has an extra parameter called moving_average_fraction. The solution to handle this is in https://stackoverflow.com/questions/55644109/how-to-convert-batchnorm-weight-of-caffe-to-pytorch-bathnorm. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias)
        self.bn   = nn.BatchNorm2d(num_features=out_channels, affine=False)
        self.gamma = nn.Parameter(torch.FloatTensor(out_channels))
        self.beta  = nn.Parameter(torch.FloatTensor(out_channels))
    
    def forward(self, input):
        return self.bn(self.conv(input)) * self.gamma[None, :, None, None]# + self.beta[None, :, None, None]

class FNBNScale(nn.Module):
    """
    This is a very common sub-structure in apollo's network: Convolution -> BatchNorm -> Scale.
    Note that there are inconsistencies between the Caffe's BatchNorm and standard BatchNorm. 
    Caffe's BatchNorm ->  Scale is similar to BatchNorm in PyTorch https://github.com/BVLC/caffe/blob/master/include/caffe/layers/batch_norm_layer.hpp#L28.
    Besides, Caffe's BatchNorm has an extra parameter called moving_average_fraction. The solution to handle this is in https://stackoverflow.com/questions/55644109/how-to-convert-batchnorm-weight-of-caffe-to-pytorch-bathnorm. 
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.fn = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.bn   = nn.BatchNorm2d(num_features=out_features, affine=False)
        self.gamma = nn.Parameter(torch.FloatTensor(out_features))
        # self.beta  = nn.Parameter(torch.FloatTensor(out_features))
    
    def forward(self, input):
        fn = self.fn(input.reshape(-1, self.fn.out_features))
        bn = self.bn(fn[:, :, None, None])
        return bn * self.gamma[None, :, None, None]
class Recognizer(nn.Module):
    def __init__(self, pool5_params):
        super().__init__()
        self.conv1 = ConvBNScale4Rec(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1, dilation=1, bias=True)
        self.conv2 = ConvBNScale4Rec(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1, dilation=1, bias=True)
        self.conv3 = ConvBNScale4Rec(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1, dilation=1, bias=True)
        self.conv4 = ConvBNScale4Rec(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1, dilation=1, bias=True)
        self.conv5 = ConvBNScale4Rec(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1, dilation=1, bias=True)
        self.pool5 = nn.AvgPool2d(kernel_size=pool5_params['kernel_size'], stride=pool5_params['stride'])
        self.ft = FNBNScale(in_features=128, out_features=128, bias=True)
        self.logits = nn.Linear(in_features=128, out_features=4, bias=True)

    def forward(self, x):
        conv1 = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=3, stride=2, padding=1)
        conv2 = F.max_pool2d(F.relu(self.conv2(conv1)), kernel_size=3, stride=2, padding=1)
        conv3 = F.max_pool2d(F.relu(self.conv3(conv2)), kernel_size=3, stride=2, padding=1)
        conv4 = F.max_pool2d(F.relu(self.conv4(conv3)), kernel_size=3, stride=2, padding=1)
        conv5 = self.pool5(F.relu(self.conv5(conv4)))
        ft = F.relu(self.ft(conv5))
        logits = self.logits(ft.reshape(-1, self.logits.in_features))
        prob = F.softmax(logits, dim=1)

        return prob
