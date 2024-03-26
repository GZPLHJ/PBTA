# Ultralytics YOLO 🚀, GPL-3.0 license
"""
Common modules
"""

import math

import torch
import torch.nn as nn

from ultralytics.yolo.utils.tal import dist2bbox, make_anchors


import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function, Variable
from torch.nn import Module, parameter

from queue import Queue
import warnings
# try:
#     from queue import Queue
# except ImportError:
    # from Queue import Queue

from torch.nn.modules.batchnorm import _BatchNorm
from functools import partial


from timm.models.layers import DropPath, trunc_normal_



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p




#  1*1 3*3 1*1
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        expansion = 4
        c = out_channels // expansion

        self.conv1 = nn.Conv2d(in_channels, c, kernel_size=1, stride=1, padding=0, bias=False)  # [64, 256, 1, 1]
        self.bn1 = norm_layer(c)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(c, c, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(c)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(c, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(out_channels)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.residual_bn = norm_layer(out_channels)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, return_x_2=False):#默认值是return_x_2=True。若为True那输出的是两个值。输入：[6, 32, 640, 640]，输出：分别是[6, 16, 640, 640]，[6, 64, 640, 640]
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) #if x_t_r is None else self.conv2(x + x_t_r)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions. Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)




# region LVCBlock
class LVCBlock(nn.Module):
    def __init__(self, c1, c2, num_codes, channel_ratio=0.25, base_channel=64):
        super(LVCBlock, self).__init__()
        self.c2 = c2
        self.num_codes = num_codes
        num_codes = 64

        self.conv_1 = ConvBlock(in_channels=c1, out_channels=c2, res_conv=True, stride=1)
        #return_x_2=True时 输入：[6, 32, 640, 640]，输出：分别是[6, 16, 640, 640]，[6, 64, 640, 640]
        #return_x_2=False时 输入：[6, 32, 640, 640]，输出：分别是[6, 64, 640, 640]
        self.LVC = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            Encoding(in_channels=c1, num_codes=num_codes),
            nn.BatchNorm1d(num_codes),
            nn.ReLU(inplace=True),
            Mean(dim=1))
        self.fc = nn.Sequential(nn.Linear(c1, c1), nn.Sigmoid())
    def forward(self, x):
        # x = self.conv_1(x, return_x_2=False)
        x = self.conv_1(x)
        en = self.LVC(x)
        gam = self.fc(en)
        b, in_channels, _, _ = x.size()
        y = gam.view(b, in_channels, 1, 1)
        x = F.relu(x + x * y,inplace=False)
        return x

# endregion

# region L-MLPBlock
# LightMLPBlock
class LightMLPBlock(nn.Module):
    '''
    x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.dw(self.norm1(x)))注意MLP要求此特征图输入输出变化得一样才行，这里
    注意MLP要求此特征图输入输出变化得一样才行，这里
    '''
    def __init__(self, c1, c2, ksize=1, stride=1, act="silu",
                 mlp_ratio=4., drop=0., act_layer=nn.GELU,
                 use_layer_scale=True, layer_scale_init_value=1e-5, drop_path=0.,
                 norm_layer=GroupNorm):  # act_layer=nn.GELU,
        super().__init__()
        self.dw = DWConv_m_a(c1, c2, k=1, s=1, act="silu")
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.out_channels = c2

        self.norm1 = norm_layer(c1)
        self.norm2 = norm_layer(c1)

        mlp_hidden_dim = int(c1 * mlp_ratio)
        self.mlp = Mlp(in_features=c1, hidden_features=mlp_hidden_dim, act_layer=nn.GELU,
                       drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((c2)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((c2)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.dw(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.dw(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# endregion

# region EVC Block
# EVCBlock
class EVCBlock(nn.Module):
    def __init__(self, c1, c2, channel_ratio=4, base_channel=16):
        super().__init__()
        expansion = 2
        ch = c2 * expansion
        # Stem stage: get the feature maps by conv block (copied form resnet.py) 进入conformer框架之前的处理
        # !!!!!!此处理输出的图片通道尺寸均不变
        self.conv1 = nn.Conv2d(c1, c1, kernel_size=7, stride=1, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(c1)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 1 / 4 [56, 56]

        # LVC
        self.lvc = LVCBlock(c1=c1, c2=c2, num_codes=64)  # c1值暂时未定
        # LightMLPBlock
        self.l_MLP = LightMLPBlock(c1, c2, ksize=1, stride=1, act="silu", act_layer=nn.GELU, mlp_ratio=4., drop=0.,
                                   use_layer_scale=True, layer_scale_init_value=1e-5, drop_path=0.,
                                   norm_layer=GroupNorm)
        self.cnv1 = nn.Conv2d(ch, c2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 预处理 !!!!!!此处理输出的图片通道尺寸均不变
        x1 = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        # LVCBlock
        x_lvc = self.lvc(x1)
        # LightMLPBlock
        x_lmlp = self.l_MLP(x1)
        # concat concat叠加后由
        x = torch.cat((x_lvc, x_lmlp), dim=1)
        x = self.cnv1(x)
        return x
# endregion


#region 2.5 深度可分离卷积-act可选择
def DWConv_m_a(c1, c2, k=1, s=1, act = "silu"):
    # Depthwise convolution
    return Conv_m_a(c1, c2, k, s, g=math.gcd(c1, c2), act=act)
#endregion

#region 3.5普通卷积 conv+bn -act可选择
class Conv_m_a(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act="silu"):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_m_a, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
"""用于Model类的fuse函数
融合conv+bn 加速推理 一般用于测试/验证阶段
"""

#endregion

#region actfunction选择
def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module
#endregion


# LVC
class Encoding(nn.Module):
    def __init__(self, in_channels, num_codes):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.in_channels, self.num_codes = in_channels, num_codes
        num_codes = 64
        std = 1. / ((num_codes * in_channels)**0.5)
        # [num_codes, channels] 经过axivr初始化的二维张量
        self.codewords = nn.Parameter(
            torch.empty(num_codes, in_channels, dtype=torch.float).uniform_(-std, std), requires_grad=True)
        # [num_codes] 经过axivr初始化的一维张量
        self.scale = nn.Parameter(torch.empty(num_codes, dtype=torch.float).uniform_(-1, 0), requires_grad=True)

    @staticmethod
    def scaled_l2(x, codewords, scale):##输入 x为(32,50176,3)和codewords（num_codes, in_channels）构成的初始化二位张量
        num_codes, in_channels = codewords.size()
        b = x.size(0)
        expanded_x = x.unsqueeze(2).expand((b, x.size(1), num_codes, in_channels))#(32,50176,3)-（unsqueeze）->(32,50176,1,3)-(expand)->(32,50176,num_codes=64,3) 其中64*3
        # 注意其中expand()函数在第三维度进行了扩展，扩展的方式，就是把最后的第四维度的那3个数进行复制，复制64（num_codes）次

        # ---处理codebook (num_code, c1)
        reshaped_codewords = codewords.view((1, 1, num_codes, in_channels))#(64,3)-(view)->(1,1,64,3)

        # 把scale从1, num_code变成   batch, c2, N, num_codes
        reshaped_scale = scale.view((1, 1, num_codes))  # (num_codes=64)-(view)->(1,1,64)

        # ---计算rik = z1 - d  # b, N, num_codes
        scaled_l2_norm = reshaped_scale * (expanded_x - reshaped_codewords).pow(2).sum(dim=3)
        #[(32,50176,64,3)-(1,1,64,3)]->(32,50176,64,3)每一组（3个像素点组成的组合）像素点与码本相减
        #(32,50176,64,3)-（pow(2)）-》(32,50176,64)
        #(32,50176,64)*(1,1,64)-》(32,50176,64)
        return scaled_l2_norm

    @staticmethod
    def aggregate(assignment_weights, x, codewords):#assignment_weights:(32,50176,64)  x:(32,3,224,224)   codewords:(num_codes=64, in_channels=3):(64,3)
        num_codes, in_channels = codewords.size()

        # ---处理codebook
        reshaped_codewords = codewords.view((1, 1, num_codes, in_channels))#(64,3)->(1,1,64,3)
        b = x.size(0)

        # ---处理特征向量x b, c1, N
        expanded_x = x.unsqueeze(2).expand((b, x.size(1), num_codes, in_channels))#(32,50176,3)-（unsqueeze）->(32,50176,1,3)-(expand)->(32,50176,num_codes=64,3) 其中64*3

        #变换rei  b, N, num_codes,-
        assignment_weights = assignment_weights.unsqueeze(3)  # b, N, num_codes,(32,50176,64)->(32,50176,64,1)

        # ---开始计算eik,必须在Rei计算完之后
        encoded_feat = (assignment_weights * (expanded_x - reshaped_codewords)).sum(1)
        #{(32,50176,64,1)*[(32,50176,64,3)-(1,1,64,3)]}.sum(1)-----》{(32,50176,64,1)*(32,50176,64,3)}.sum(1)-----》(32,50176,64,3).sum(1)----》(32,64,3)
        #注意在某维度求和就是对于这个维度的比如x个块当中的同一位置处求和最终生成一个块。而这个块是此维度下属维度的数组成的。比如(32,50176,64,3).sum(1)就是在第二维度进行求和，把（64，3）这个块也就是三四维的数据组成的块，在50176个（64，3）这样的块的相同位置处的数据进行求和输出一个最终的值，最后50176的这个维度也就被压缩没了生成了新的(32,64,3)尺寸的数组
        return encoded_feat#(32,64,3)

    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == self.in_channels
        b, in_channels, w, h = x.size()

        # [batch_size, height x width, channels]
        x = x.view(b, self.in_channels, -1).transpose(1, 2).contiguous()#(32,3,224,224)-(view)->(32,3,50176)-(transpose)->(32,50176,3)-(contiguous连续化)->(32,50176,3)  这样操作可以理解为，将一张图片中RGB图像的所有像素信息按照3个一组进行排列，然后后续通过自己创造的码本找到像素点之间的信息。

        # assignment_weights: [batch_size, channels, num_codes]
        assignment_weights = F.softmax(self.scaled_l2(x, self.codewords, self.scale), dim=2)#输入 x为(32,50176,3)和codewords（num_codes, in_channels）构成的初始化二位张量，以及scale这个一维张量

        # aggregate
        encoded_feat = self.aggregate(assignment_weights, x, self.codewords)#(32,64,3)
        return encoded_feat

