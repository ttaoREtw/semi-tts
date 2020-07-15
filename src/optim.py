import torch
import numpy as np

class Optimizer():
    def __init__(self, parameters, optimizer, lr, lr_scheduler, tf_start=1, tf_end=1, tf_step=1,
                 recon_init_weight=1.0, recon_decay=0.0, **kwargs):
        
        # Setup teacher forcing scheduler
        self.tf_rate = lambda step: max(tf_end, tf_start-(tf_start-tf_end)*step/tf_step)
        self.recon_sch = recon_init_weight!=1.0
        self.recon_rate = lambda step: max(1.0, recon_init_weight-(recon_init_weight-1.0)/max(recon_decay,1.0))

        # Setup torch optimizer
        self.tf_type = tf_end!=1
        self.opt_type = optimizer
        self.sch_type = lr_scheduler
        opt = getattr(torch.optim,optimizer)
        if lr_scheduler == 'warmup':
            warmup_step = 4000.0
            init_lr = lr
            self.lr_scheduler = lambda step: init_lr * warmup_step **0.5 * np.minimum((step+1)*warmup_step**-1.5,(step+1)**-0.5 )
            self.opt = opt(parameters,lr=1.0)
        elif lr_scheduler == 'decay':
            warmup_step = 1000.0
            init_lr = lr
            self.lr_scheduler = lambda step: init_lr * warmup_step **0.5 * np.minimum((step+1)*warmup_step**-1.5,(step+1)**-0.5 )
            self.opt = opt(parameters,lr=1.0)
        else:
            self.lr_scheduler = None
            self.opt = opt(parameters,lr=lr)

    def get_opt_state_dict(self):
        return self.opt.state_dict()

    def load_opt_state_dict(self, state_dict):
        self.opt.load_state_dict(state_dict)

    def pre_step(self, step):
        if self.lr_scheduler is not None:
            cur_lr = self.lr_scheduler(step)
            for param_group in self.opt.param_groups:
                param_group['lr'] = cur_lr
        self.opt.zero_grad()
        return self.tf_rate(step)

    def step(self):
        self.opt.step()
    
    def recon_rate(self,step):
        return self.recon_rate(step)

    def create_msg(self):
        return ['Optim.spec.| Algo. = {}\t| Lr/sampling/rec.loss scheduler = {}/{}/{}'\
                   .format(self.opt_type, self.sch_type, self.tf_type, self.recon_sch)]



