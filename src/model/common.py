import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                     padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
        self,
        rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040),
        rgb_std=(1.0, 1.0, 1.0),
        sign=-1,
    ):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
        self,
        conv,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=False,
        bn=True,
        act=nn.ReLU(True),
    ):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
        self,
        conv,
        n_feats,
        kernel_size,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class ResFractalConvBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1,
                 padding=1, pad_type='zero'):
        super().__init__()
        if pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'reflect':
            # [!] the paper used reflect padding - just for data augmentation?
            self.pad = nn.ReflectionPad2d(padding)
        else:
            raise ValueError(pad_type)

        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride,
                              padding=0, bias=True)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        out = F.relu_(out)

        return out


class ResFractalBlock(nn.Module):
    def __init__(self, n_columns, C_in, C_out, pad_type='zero'):
        """ Fractal block
        Args:
            - n_columns: # of columns
            - C_in: channel_in
            - C_out: channel_out
            - p_ldrop: local droppath prob
            - p_dropout: dropout prob
            - pad_type: padding type of conv
            - doubling: if True, doubling by 1x1 conv in front of the block.
            - dropout_pos: the position of dropout
                - CDBR (default): conv-dropout-BN-relu
                - CBRD: conv-BN-relu-dropout
                - FD: fractal_block-dropout
        """
        super().__init__()

        self.n_columns = n_columns

        self.columns = nn.ModuleList([nn.ModuleList()
                                      for _ in range(n_columns)])
        self.max_depth = 2 ** (n_columns-1)

        dist = self.max_depth
        self.count = np.zeros([self.max_depth], dtype=np.int)
        for col in self.columns:
            for i in range(self.max_depth):
                if (i+1) % dist == 0:
                    first_block = (i+1 == dist)  # first block in this column
                    if first_block:
                        cur_C_in = C_in
                    else:
                        cur_C_in = C_out

                    module = ResFractalConvBlock(cur_C_in, C_out,
                                                 pad_type=pad_type)
                    self.count[i] += 1
                else:
                    module = None

                col.append(module)

            dist //= 2

        joins = []
        assert self.n_columns in (2, 3, 4), "only 2, 3, 4 columns are supported"
        if self.n_columns == 2:
            joins = [2]
        elif self.n_columns == 3:
            joins = [2, 3]
        elif self.n_columns == 4:
            joins = [2, 3, 2, 4]

        self.joinTimes = len(joins)

        # self.weights = nn.ModuleList()
        self.weights = []
        for i in joins:
            self.weights.append(nn.Parameter((torch.ones(i)).cuda()))

        self.joinConv = nn.ModuleList()
        for i in joins:
            self.joinConv.append(default_conv(i * C_out, C_out, 1, bias=False))

    def mean_join(self, outs, cnt=0):
        """
        Args:
            - outs: the outputs to join
            - global_cols: global drop path columns
        """
        outs = torch.stack(outs)  # [n_cols, B, C, H, W]
        outs = outs.mean(dim=0)  # no drop
        return outs

    def conv_join(self, outs, cnt=0):
        """
        Args:
            - outs: the outputs to join
            - global_cols: global drop path columns
        """
        # out = torch.stack(outs)  # [n_cols, B, C, H, W]
        # out = out.mean(dim=0)  # no drop

        outs = torch.cat(outs, dim=1)  # [n_cols, B, C, H, W]
        return self.joinConv[cnt](outs)

    def weighted_join(self, outs, cnt=0):
        """
        Args:
            - outs: the outputs to join
            - global_cols: global drop path columns
        """
        # out = torch.stack(outs)  # [n_cols, B, C, H, W]
        # out = out.mean(dim=0)  # no drop

        outs = torch.stack(outs)  # [n_cols, B, C, H, W]
        weights = torch.nn.functional.softmax(self.weights[cnt])

        out = torch.mul(outs[0], weights[0])
        for i in range(1, len(outs)):
            out += torch.mul(outs[i], weights[i])
        return out

    def forward(self, x):
        outs = [x] * self.n_columns
        cnt = 0
        for i in range(self.max_depth):
            st = self.n_columns - self.count[i]
            cur_outs = []  # outs of current depth

            for c in range(st, self.n_columns):
                cur_in = outs[c]  # current input
                cur_module = self.columns[c][i]  # current module
                cur_outs.append(cur_module(cur_in))

            if len(cur_outs) == 1:
                joined = cur_outs[0]
            else:
                joined = self.conv_join(cur_outs, cnt)
                # joined = self.weighted_join(cur_outs, cnt)
                # joined = self.mean_join(cur_outs, cnt)
                # joined = self.join(cur_outs, cnt)
                cnt = (cnt + 1) % self.joinTimes
            for c in range(st, self.n_columns):
                outs[c] = joined

        """ add resdual """
        # since the after the loop, each elem in outs is the same,
        # can process the last elem only
        outs[-1] += x

        return outs[-1]


class MSRBlock(nn.Module):
    def __init__(self, conv=default_conv, n_feats=64):
        super().__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = conv(n_feats, n_feats, kernel_size_1)
        self.conv_3_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1)
        self.conv_5_1 = conv(n_feats, n_feats, kernel_size_2)
        self.conv_5_2 = conv(n_feats * 2, n_feats * 2, kernel_size_2)
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x
        return output


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
