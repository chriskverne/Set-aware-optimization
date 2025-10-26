import torch
from torch.optim.optimizer import Optimizer

class AdamRoot(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 gamma=0.95, ema_c=0.9, inner_steps=1,
                 alpha_max=1.0, dir_norm_clip=10.0, denom_floor=1e-12,
                 residual_clip=10.0, success_tol=0.99):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        gamma=gamma, ema_c=ema_c, inner_steps=inner_steps,
                        alpha_max=alpha_max, dir_norm_clip=dir_norm_clip,
                        denom_floor=denom_floor, residual_clip=residual_clip,
                        success_tol=success_tol)
        super().__init__(params, defaults)
        self.state.setdefault('c', None)

    @torch.no_grad()
    def _adam_direction(self, group):
        d_norm2 = 0.0
        for p in group['params']:
            if p.grad is None: 
                continue
            state = self.state.setdefault(p, {})
            if len(state) == 0:
                state['step'] = 0
                state['m'] = torch.zeros_like(p)
                state['v'] = torch.zeros_like(p)
            m, v = state['m'], state['v']
            beta1, beta2 = group['betas']
            state['step'] += 1
            m.mul_(beta1).add_(p.grad, alpha=1-beta1)
            v.mul_(beta2).addcmul_(p.grad, p.grad, value=1-beta2)
            # bias correction
            m_hat = m / (1 - beta1**state['step'])
            v_hat = v / (1 - beta2**state['step'])
            d = m_hat / (v_hat.sqrt() + group['eps'])
            state['d'] = d
            d_norm2 += float(d.pow(2).sum().item())
        return d_norm2

    def step(self, closure):
        """
        closure() must:
          - zero_grad
          - forward
          - backward()
          - return loss (tensor)
        """
        assert callable(closure), "AdamRoot requires a closure that does forward+backward"

        # compute current loss and grads WITH autograd enabled
        loss = closure()
        loss_val = float(loss.detach().item())

        for group in self.param_groups:
            # init/maintain setpoint c (global across groups here)
            c_prev = self.state.get('c')
            if c_prev is None:
                c = loss_val * group['gamma']
            else:
                c_target = loss_val * group['gamma']
                c = group['ema_c'] * c_prev + (1 - group['ema_c']) * c_target

            # Build Adam direction (no_grad inside helper)
            d_norm2 = self._adam_direction(group)
            d_norm = (d_norm2**0.5) if d_norm2 > 0 else 0.0

            # directional derivative g·d
            gdot = 0.0
            for p in group['params']:
                if p.grad is None: 
                    continue
                d = self.state[p]['d']
                gdot += float((p.grad * d).sum().item())

            # Polyak-style root step along d
            r = loss_val - c
            r = max(min(r, group['residual_clip']), -group['residual_clip'])
            denom = max(abs(gdot), group['denom_floor'])
            alpha = r / denom
            alpha = max(min(alpha, group['alpha_max']), -group['alpha_max'])

            scale = group['lr'] * (1.0 if d_norm == 0.0 else min(1.0, group['dir_norm_clip']/d_norm))

            # parameter update w/o grad tracking
            with torch.no_grad():
                for p in group['params']:
                    if p.grad is None: 
                        continue
                    d = self.state[p]['d']
                    p.add_(d, alpha=-(alpha * scale))

            # success check: recompute loss and grads
            new_loss = closure()
            new_loss_val = float(new_loss.detach().item())
            if new_loss_val <= group['success_tol'] * loss_val:
                self.state['c'] = c
            else:
                # don't over-tighten
                if c_prev is None:
                    self.state['c'] = c
                else:
                    self.state['c'] = c_prev

        return loss
