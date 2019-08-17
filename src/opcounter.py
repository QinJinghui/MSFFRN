import torch

import data
import loss
import model
import utility
from option import args
from trainer import Trainer
from thop import profile

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if args.data_test == "video":
    from videotester import VideoTester
    model = model.Model(args, checkpoint)
    t = VideoTester(args, model, checkpoint)
    t.test()
else:
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        input = torch.randn(1, 3, 224, 224)
        flops, params = profile(model, inputs=(input, ))
        # flops, params = profile(model, input_size=(1, 3, 224,224))
        print("flops\tparams: %d\t%d" % (int(flops), int(params)))
