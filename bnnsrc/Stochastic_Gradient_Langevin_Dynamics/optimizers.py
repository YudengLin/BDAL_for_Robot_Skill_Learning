from torch.optim.optimizer import Optimizer, required

import torch

class SGLD(Optimizer):
    """
    RMSprop preconditioned SGLD using pytorch rmsprop implementation.
    """

    def __init__(self, model, params, lr=required, norm_sigma=0, alpha=0.99, eps=1e-8, centered=False, addnoise=True, Exp=None):

        weight_decay = 0# 1 / (norm_sigma ** 2)
        self.model = model
        self.Exp = Exp
        self.lr = 1e-3
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, weight_decay=weight_decay, alpha=alpha, eps=eps, centered=centered, addnoise=addnoise)
        super(SGLD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('centered', False)

    def step(self,closure=None, *, grad_scaler=None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:

            weight_decay = group['weight_decay']

            for name, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                d_p = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']
                state['step'] += 1

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                square_avg.mul_(alpha).addcmul_(d_p, d_p, value=1 - alpha, )

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(d_p,alpha=1 - alpha)
                    avg = torch.addcmul(square_avg, grad_avg, grad_avg,value=-1).sqrt().add_(group['eps'])

                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['addnoise']:
                  # used program model
                    err_margin = 0.
                    no_verify = 0
                    _, layer_id, array, _ = name.split('.', 3)
                    with torch.no_grad():
                        layer = self.model.layers[int(layer_id)]
                        scale = layer.scale
                        delta = - self.lr * 0.5 * d_p.div_(avg)*scale
                        target = p.data + delta
                        p_adjust = group['lr']
                        layer.program_conductance(err_margin, delta, target, no_verify, p_adjust,
                                                  Exp=self.Exp)

                else:
                    p.data.addcdiv_(-self.lr, 0.5 * d_p, avg)


        return loss
