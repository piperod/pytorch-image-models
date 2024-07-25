
"""
An implementation of HMAX:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy as sp
import time

from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs



def get_gabor(l_size, la, si, n_ori, aspect_ratio):
    """generate the gabor filters

    Args
    ----
        l_size: float
            gabor sizes
        la: float
            lambda
        si: float
            sigma
        n_ori: type integer
            number of orientations
        aspect_ratio: type float
            gabor aspect ratio

    Returns
    -------
        gabor: type nparray
            gabor filter

    """

    gs = l_size

    # TODO: inverse the axes in the begining so I don't need to do swap them back
    # thetas for all gabor orientations
    th = np.array(range(n_ori)) * np.pi / n_ori + np.pi / 2.
    th = th[sp.newaxis, sp.newaxis, :]
    hgs = (gs - 1) / 2.
    yy, xx = sp.mgrid[-hgs: hgs + 1, -hgs: hgs + 1]
    xx = xx[:, :, sp.newaxis]
    yy = yy[:, :, sp.newaxis]

    # x = xx * np.cos(th) - yy * np.sin(th)
    # y = xx * np.sin(th) + yy * np.cos(th)
    x = xx * np.cos(th) + yy * np.sin(th)
    y = - xx * np.sin(th) + yy * np.cos(th)

    filt = np.exp(-(x ** 2 + (aspect_ratio * y) ** 2) / (2 * si ** 2)) * np.cos(2 * np.pi * x / la)
    filt[np.sqrt(x ** 2 + y ** 2) > gs / 2.] = 0

    # gabor normalization (following cns hmaxgray package)
    for ori in range(n_ori):
        filt[:, :, ori] -= filt[:, :, ori].mean()
        filt_norm = fastnorm(filt[:, :, ori])
        if filt_norm != 0: filt[:, :, ori] /= filt_norm

    filt_c = np.array(filt, dtype='float32').swapaxes(0, 2).swapaxes(1, 2)

    filt_c = torch.Tensor(filt_c)
    filt_c = filt_c.view(n_ori, 1, gs, gs)
    # filt_c = filt_c.repeat((1, 3, 1, 1))

    return filt_c


def fastnorm(in_arr):
    arr_norm = np.dot(in_arr.ravel(), in_arr.ravel()).sum() ** (1. / 2.)
    return arr_norm

def fastnorm_tensor(in_arr):
    arr_norm = torch.dot(in_arr.ravel(), in_arr.ravel()).sum() ** (1. / 2.)
    return arr_norm


def get_sp_kernel_sizes_C(scales, num_scales_pooled, scale_stride):
    '''
    Recursive function to find the right relative kernel sizes for the spatial pooling performed in a C layer.
    The right relative kernel size is the average of the scales that will be pooled. E.g, if scale 7 and 9 will be
    pooled, the kernel size for the spatial pool is 8 x 8

    Parameters
    ----------
    scales
    num_scales_pooled
    scale_stride

    Returns
    -------
    list of sp_kernel_size

    '''

    if len(scales) < num_scales_pooled:
        return []
    else:
        average = int(sum(scales[0:num_scales_pooled]) / len(scales[0:num_scales_pooled]))
        return [average] + get_sp_kernel_sizes_C(scales[scale_stride::], num_scales_pooled, scale_stride)


def pad_to_size(a, size):
    current_size = (a.shape[-2], a.shape[-1])
    total_pad_h = size[0] - current_size[0]
    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top

    total_pad_w = size[1] - current_size[1]
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left

    a = nn.functional.pad(a, (pad_left, pad_right, pad_top, pad_bottom))

    return a

def check_for_nans(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaNs found in {name}")

#########################################################################################################
class HMAX(nn.Module):
    def __init__(self,
                #  args,
                 in_chans=3,
                 s1_channels_out=96,
                 s1_scale=15,
                 s1_stride=1,
                 s2b_kernel_size=[4,8,12,16],
                 s2b_channels_out=128,
                 s2_channels_out=128,
                 s2_kernel_size=3,
                 s2_stride=1,
                 s3_channels_out=256,
                 s3_kernel_size=3,
                 s3_stride=1,
                 hidden_dim=1024,
                 num_classes=1000,
                 drop_rate=0.5,
                 drop_path_rate=0.5,
                 bypass_only=False,
                 ):
        
        self.in_chans=in_chans
        self.s1_channels_out=s1_channels_out
        self.s1_scale=s1_scale
        self.s1_stride=s1_stride
        self.s2b_kernel_size=s2b_kernel_size
        self.s2b_channels_out= s2b_channels_out
        self.s2_channels_out = s2_channels_out
        self.s2_kernel_size = s2_kernel_size
        self.s2_stride=s2_stride
        self.s3_channels_out=s3_channels_out
        self.s3_kernel_size=s3_kernel_size
        self.s3_stride=s3_stride
        self.num_classes=num_classes
        self.drop_rate=drop_rate
        self.drop_path_rate=drop_path_rate
        self.hidden_dim=hidden_dim

        super(HMAX, self).__init__()
#########################################################################################################

        self.conv1 = nn.Conv2d(in_chans, self.s1_channels_out, kernel_size=self.s1_scale, stride=self.s1_stride, padding='valid')
        self.batchnorm1 = nn.Sequential(nn.BatchNorm2d(self.s1_channels_out, 1e-3),
                                    nn.ReLU(True),
                                    )

        self.s2b_seqs = nn.ModuleList()
        for size in self.s2b_kernel_size:
            self.s2b_seqs.append(nn.Sequential(
                nn.Conv2d(self.s1_channels_out, self.s2b_channels_out, kernel_size=size, stride=1, padding=size//2),
                nn.BatchNorm2d(self.s2b_channels_out, 1e-3),
                nn.ReLU(True)
            ))        
        
        self.s2_seq = nn.Sequential(nn.Conv2d(self.s1_channels_out, self.s2_channels_out, kernel_size=self.s2_kernel_size, stride=self.s2_stride),
                                                nn.BatchNorm2d(self.s2_channels_out, 1e-3),
                                                nn.ReLU(True)
                                                )

        self.s3_seq = nn.Sequential(nn.Conv2d(self.s2_channels_out, self.s3_channels_out, kernel_size=self.s3_kernel_size, stride=self.s3_stride),
                                                nn.BatchNorm2d(self.s3_channels_out, 1e-3),
                                                nn.ReLU(True)
                                                )

        self.classifier = nn.Sequential(
                                        # nn.Dropout(0.5),
                                        nn.Linear(self.get_s4_in_channels(), self.hidden_dim),  # fc1
                                        # nn.Dropout(0.2),
                                        nn.Linear(self.hidden_dim, 1024),  # fc2
                                        nn.Linear(1024, self.num_classes)  # fc3
                                        )


    def get_s4_in_channels(self):
        
        s1_out_size = ((224 - self.s1_scale) // self.s1_stride) + 1
        c1_out_size = ((s1_out_size - 14) // 1) + 1
        s2b_out_size = (c1_out_size + 1) ## this is currently true because padding + stride 1
        c2b_out_size = ((s2b_out_size - 12) // 6) + 1
        s2_out_size =  ((c1_out_size - self.s2_kernel_size) // self.s2_stride) + 1
        c2_out_size = ((s2_out_size - 12) // 6) + 1
        s2_out_size =  ((c2_out_size - self.s3_kernel_size) // self.s3_stride) + 1
        c3_out_size = ((s2_out_size - 3) // 1) + 1

        c2b_out = len(self.s2b_kernel_size) * self.s2b_channels_out * c2b_out_size * c2b_out_size
        c3_out = self.s3_channels_out * c3_out_size * c3_out_size
        s4_in = c2b_out + c3_out

        return s4_in

    def forward(self, x):
        start = time.time()
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.max_pool2d(x, kernel_size=14, stride=1)

        bypass = torch.cat([seq(x) for seq in self.s2b_seqs], dim=1)
        bypass = F.max_pool2d(bypass, kernel_size=12, stride=6)

        x = self.s2_seq(x)
        x = F.max_pool2d(x, kernel_size=12, stride=6)

        x = self.s3_seq(x)
        x = F.max_pool2d(x, 3, 1)

        x = torch.flatten(x, start_dim=1)
        bypass = torch.flatten(bypass, start_dim=1)

        x = torch.cat([bypass, x], dim=1)
        del bypass

        x = self.classifier(x)

        return x    

#########################################################################################################
class HMAX_bypass(nn.Module):
    def __init__(self,
                #  args,
                 in_chans=3,
                 s1_channels_out=96,
                 s1_scale=15,
                 s1_stride=1,
                 s2b_kernel_size=[4,8,12,16],
                 s2b_channels_out=128,
                 hidden_dim=1024,
                 num_classes=1000,
                 drop_rate=0.5,
                 drop_path_rate=0.5,
                 bypass_only=False,
                 ):
        
        self.in_chans=in_chans
        self.s1_channels_out=s1_channels_out
        self.s1_scale=s1_scale
        self.s1_stride=s1_stride
        self.s2b_kernel_size=s2b_kernel_size
        self.s2b_channels_out= s2b_channels_out
        self.num_classes=num_classes
        self.drop_rate=drop_rate
        self.drop_path_rate=drop_path_rate
        self.hidden_dim=hidden_dim

        super(HMAX_bypass, self).__init__()
#########################################################################################################

        self.conv1 = nn.Conv2d(in_chans, self.s1_channels_out, kernel_size=self.s1_scale, stride=self.s1_stride, padding='valid')
        self.batchnorm1 = nn.Sequential(nn.BatchNorm2d(self.s1_channels_out, 1e-3),
                                    nn.ReLU(True),
                                    )

        self.s2b_seqs = nn.ModuleList()
        for size in self.s2b_kernel_size:
            self.s2b_seqs.append(nn.Sequential(
                nn.Conv2d(self.s1_channels_out, self.s2b_channels_out, kernel_size=size, stride=1, padding=size//2),
                nn.BatchNorm2d(self.s2b_channels_out, 1e-3),
                nn.ReLU(True)
            ))        

        self.classifier = nn.Sequential(
                                        # nn.Dropout(0.5),
                                        nn.Linear(self.get_s4_in_channels(), self.hidden_dim),  # fc1
                                        # nn.Dropout(0.2),
                                        nn.Linear(self.hidden_dim, 1024),  # fc2
                                        nn.Linear(1024, self.num_classes)  # fc3
                                        )


    def get_s4_in_channels(self):
        
        s1_out_size = ((224 - self.s1_scale) // self.s1_stride) + 1
        c1_out_size = ((s1_out_size - 14) // 1) + 1
        s2b_out_size = (c1_out_size + 1) ## this is currently true because padding + stride 1
        c2b_out_size = ((s2b_out_size - 12) // 6) + 1

        c2b_out = len(self.s2b_kernel_size) * self.s2b_channels_out * c2b_out_size * c2b_out_size
        s4_in = c2b_out

        return s4_in

    def forward(self, x):
        start = time.time()
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.max_pool2d(x, kernel_size=14, stride=1)

        x = torch.cat([seq(x) for seq in self.s2b_seqs], dim=1)
        x = F.max_pool2d(x, kernel_size=12, stride=6)

        x = self.classifier(x)

        return x    

def checkpoint_filter_fn(state_dict, model: nn.Module):
    out_dict = {}
    for k, v in state_dict.items():
        out_dict[k] = v
    return out_dict


def _create_HMAX(variant, pretrained=False, **kwargs):
    """
    Constructs an HMAX model
    """
    model_kwargs = dict(
        **kwargs,
    )
    return build_model_with_cfg(
        HMAX,
        variant,
        pretrained,
        **model_kwargs,
    )

def _create_HMAX_bypass(variant, pretrained=False, **kwargs):
    return HMAX_bypass(**kwargs)

@register_model
def hmax_full(pretrained=False, **kwargs) -> HMAX:
    """ HMAX """
    model = _create_HMAX('hmax_full', pretrained=pretrained, **kwargs)
    return model

@register_model
def hmax_bypass(pretrained=False, **kwargs) -> HMAX:
    """ HMAX """
    model = _create_HMAX_bypass('hmax_bypass', pretrained=pretrained, **kwargs)
    return model