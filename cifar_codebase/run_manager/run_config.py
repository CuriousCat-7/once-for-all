import json
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision


class RunConfig() :

    def __init__(self, init_lr, no_decay_keys, opt_param):
        # optimizer
        self.init_lr = init_lr
        self.no_decay_keys = no_decay_keys
        self.opt_param = opt_param

        #  dataset
        pass

    def build_scheduler(self, optimizer):
        pass

    def build_optimizer(self, net_params):
        if self.no_decay_keys is not None:
            assert isinstance(net_params, list) and len(net_params) == 2
            net_params = [
                {'params': net_params[0], 'weight_decay': self.weight_decay},
                {'params': net_params[1], 'weight_decay': 0},
            ]
        else:
            net_params = [{'params': net_params, 'weight_decay': self.weight_decay}]

        if self.opt_type == 'sgd':
            opt_param = {} if self.opt_param is None else self.opt_param
            momentum, nesterov = opt_param.get('momentum', 0.9), opt_param.get('nesterov', True)
            optimizer = torch.optim.SGD(net_params, self.init_lr, momentum=momentum, nesterov=nesterov)
        elif self.opt_type == 'adam':
            optimizer = torch.optim.Adam(net_params, self.init_lr)
        else:
            raise NotImplementedError
        return optimizer


