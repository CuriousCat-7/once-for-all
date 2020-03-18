import json
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision


class RunManager():
    def __init__(self, path, net, run_config: RunConfig, init=True, measure_latency=None, no_gpu=False, mix_prec=None):
        self.path = path
        self.net = net
        self.run_config = run_config
        self.mix_prec = mix_prec

        os.makedirs(self.path, exist_ok=True)

        # move network to GPU if available
        if torch.cuda.is_available() and (not no_gpu):
            self.device = torch.device('cuda:0')
            self.net = self.net.to(self.device)
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
