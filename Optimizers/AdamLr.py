import torch

class RootStep:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, gamma=0.95):
        self.params = list(params)
        self.opt = torch.optim.Adam(self.params, lr=lr, betas=(beta1, beta2), eps=eps)
        self.gamma = gamma
        self.c = None
        self._ema_abs_res = 0.0  # optional for diagnostics / adaptive caps

    @torch.no_grad()
    def _init_c(self, residual):
        # residual is a Python float here
        self.c = residual * self.gamma

    def step(self, loss_scalar, update_c="decay"):
        """
        loss_scalar: a scalar tensor f(theta) (e.g., MSE, or any scalar you want to hit c)
        """
        if self.c is None:
            self._init_c(float(loss_scalar.detach()))

        # Root loss: g = 1/2 (f - c)^2
        g = 0.5 * (loss_scalar - loss_scalar.new_tensor(self.c)).pow(2)

        self.opt.zero_grad(set_to_none=True)
        g.backward()                      # ∇g = (f - c) ∇f
        # (optional) clip for stability on early iterations
        torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)

        self.opt.step()

        # Update target c
        with torch.no_grad():
            f_val = float(loss_scalar.detach())
            self._ema_abs_res = 0.95 * self._ema_abs_res + 0.05 * abs(f_val - self.c)
            if update_c == "decay":
                self.c = self.gamma * f_val
            elif update_c == "hold":
                pass
            # If you really want "bisection-like" behavior, maintain [a,b] bounds on f
            # and move c to the midpoint — but do that on the scalar f, not tensors,
            # and keep it consistent across steps.
