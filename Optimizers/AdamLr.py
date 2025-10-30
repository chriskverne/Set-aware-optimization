import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

class RootStep:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, gamma=0.95):
        self.params = list(params)
        self.opt = torch.optim.Adam(self.params, lr=lr, betas=(beta1, beta2), eps=eps)
        self.gamma = gamma
        self.c = None
        self._ema_abs_res = 0.0 
        self.pid_controller = PIDController(Kp=0.1, Ki=0.001, Kd=0.05)

    @torch.no_grad()
    def _init_c(self, residual):
        # residual is a Python float here
        self.c = residual * self.gamma

    @torch.no_grad()
    def update_root(self, f_val):
        self.c = self.gamma * f_val

    
    @torch.no_grad()
    def update_root_pid(self, f_val):
        error = f_val
        adjustment = self.pid_controller.step(error)
        self.c -= adjustment
        self.c = max(0.0, min(self.c, f_val * 0.99))

    def step(self, loss_scalar, update_c="xxx"):
        if self.c is None:
            self._init_c(loss_scalar)

        # Root loss: g = 1/2 (f - c)^2
        r = torch.where(loss_scalar > self.c, loss_scalar - self.c, loss_scalar.detach() - self.c)
        if loss_scalar > self.c:
            g = 0.5 * r.pow(2)
        else:
            g = 0.5 * r.pow(2)

        self.opt.zero_grad(set_to_none=True)
        g.backward()                      # ∇g = (f - c) ∇f
        self.opt.step()

        # Update target c
        f_val = float(loss_scalar.detach())
        if update_c == 'decay':
            self.update_root(f_val)
        else:
            self.update_root_pid(f_val)



class PIDController:
    """A simple PID controller."""
    def __init__(self, Kp=0.1, Ki=0.001, Kd=0.05):
        self.Kp = Kp  # Proportional gain
        self.Ki = Ki  # Integral gain
        self.Kd = Kd  # Derivative gain
        
        # State variables
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, error):
        """Calculates the control signal."""
        # Integral term (accumulates past errors)
        self.integral += error
        
        # Derivative term (estimates future trend)
        derivative = error - self.prev_error
        
        # Update the state for the next step
        self.prev_error = error
        
        # Calculate the PID output signal
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative