from model import common
import torch
import torch.nn as nn


def make_model(args):
    return MSFFRN(args)


class MSFFRN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super().__init__()

        # n_feats = 64
        n_feats = args.n_feats
        n_blocks = args.n_resblocks
        kernel_size = 3
        scale = args.scale[0]

        self.n_blocks = n_blocks

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(common.ResFractalBlock(args.n_columns,
                                                       n_feats, n_feats))

        # MSFFRN tail module
        modules_tail = [
            nn.Conv2d(n_feats * (self.n_blocks + 1), n_feats, 1, padding=0,
                      stride=1),
            nn.Conv2d(n_feats, n_feats, kernel_size,padding=(kernel_size-1)//2, stride=1),
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(n_feats,  args.n_colors, kernel_size,padding=(kernel_size-1)//2, stride=1)
        ]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = x

        MSFFRB_out = []
        for i in range(self.n_blocks):
            x = self.body[i](x)
            MSFFRB_out.append(x)
        MSFFRB_out.append(res)

        res = torch.cat(MSFFRB_out, 1)
        x = self.tail(res)
        x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError(
                            'While copying the parameter named {}, '
                            'whose dimensions in the model are {} and '
                            'whose dimensions in the checkpoint are {}.'.format(
                                name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError(
                        'unexpected key "{}" in state_dict'.format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError(
                    'missing keys in state_dict: "{}"'.format(missing))