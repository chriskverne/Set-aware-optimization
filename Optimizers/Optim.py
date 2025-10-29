"""
Newton (your original) converges fast near the root but can explode when gradients are tiny or Hessian effects are wild; damping could help.
Secant (α-wise) is a great compromise: cheap, 1-D, often stable, and doesn’t require exact line search.
Bisection/Brent (α-wise) are rock-solid if you can bracket.
Param-wise secant is the simplest gradient-free swap, but it’s the most brittle in high-D because it treats each coordinate as a separate 1-D problem tied to a single scalar residual.

Generative Model Inversion: You have a trained GAN generator $G(\mathbf{z})$ that turns random noise $\mathbf{z}$ into an image. If you want to find the specific noise vector $\mathbf{z}$ that creates a particular target image $\mathbf{y}$, you solve $G(\mathbf{z}) - \mathbf{y} = \mathbf{0}$. This is exactly your problem!
Physics-Informed Neural Networks (PINNs): In PINNs, a network must obey a physical law, often expressed as a differential equation like $\text{PDE}(\mathbf{f}) = 0$. The training involves finding weights $\mathbf{w}$ that satisfy this equation at many points, which is a root-finding task.
Control Theory & Robotics: Finding the exact sequence of motor commands (the network's output) to place a robotic arm in a specific target position.
"""

import torch
from torch.optim.optimizer import Optimizer

# Needs to override __init__(), step(), zero_grad()
class RootFinding(Optimizer):
    def __init__(self, params, gamma=0.95, inner_steps=3):
        # Defaults Stores hyperparameters
        defaults = dict(gamma=gamma, inner_steps=inner_steps, eps=1e-8)
        # Gamma represents c decrease,
        # inner_steps represents how many times we solve the equation f = c
        super(RootFinding,self).__init__(params, defaults)
        # Represents current root we try to solve f = c for (This changes throguh optimization)
        self.c =  None 
        # self.a = None Explore later for setting root
        # self.b = 0 Explore later for setting root

    def update_root(self, curr_loss):
        # We can find more sophisticated methods for this approach
        self.c = curr_loss * self.defaults['gamma']

    def bisection_root_update(self, curr_loss):
        self.c = [self.a + self.b]/2

    def iterate_inner_loop(self, model, loss_fn, inputs, targets, mode='secant'):
        # Set inital target if not set
        if self.c is None:
            curr_loss = loss_fn(model(inputs), targets)
            self.update_root(curr_loss)
        
        if mode == 'secant':
            for _ in range(self.defaults['inner_steps']):
                # Evaluate current loss (no gradients needed)
                curr_loss = loss_fn(model(inputs), targets)
                # Perform a secant update
                self.secant_step(curr_loss)

        elif mode == 'polyak':
            for _ in range(self.defaults['inner_steps']):
                # Get current loss
                curr_loss = loss_fn(model(inputs), targets)
                # compute grads
                self.zero_grad()
                curr_loss.backward()
                # Do single step
                self.polyak_step(curr_loss)
        
        elif mode == 'sgd':
            for _ in range(self.defaults['inner_steps']):
                # Get current loss
                curr_loss = loss_fn(model(inputs), targets)
                # compute grads
                self.zero_grad()
                curr_loss.backward()
                # Do single step
                self.sgd_root_step(curr_loss, lr=0.01)

        # Update target c
        self.update_root(loss_fn(model(inputs), targets))

    def polyak_step(self, curr_loss):
        # Apply newton's step:
        #  x - (f-c)g/||g||^2
        
        with torch.no_grad():
            # 1) Compute sum (g^2)
            g2 = 0
            for group in self.param_groups:
                for p in group['params']:
                    g2 += p.grad.pow(2).sum()

            # 2) update params
            for group in self.param_groups:
                for p in group['params']:
                    p.add_(-(curr_loss-self.c)*p.grad/(g2+self.defaults['eps']))
    
    def secant_step(self, curr_loss, pilot_scale=1e-2):
        with torch.no_grad():
            # residual (keep self.c as a float)
            r_n = float(curr_loss.detach().item()) - float(self.c)

            # snapshot x_n BEFORE any update
            params_n_snapshot = []
            for g in self.param_groups:
                for p in g['params']:
                    params_n_snapshot.append(p.data.clone())

            prev = self.state.get('secant_prev', None)

            # --- Case 1: no previous point for this c -> create one (pilot move) ---
            if prev is None:
                # save (x0, r0) so next call has a valid pair
                self.state['secant_prev'] = {'params': params_n_snapshot, 'residual': r_n}
                # tiny random pilot: x1 = x0 - ε * u  (purely gradient-free)
                for g in self.param_groups:
                    for p in g['params']:
                        mag = p.data.norm().item() or 1.0
                        u = torch.randn_like(p.data)
                        u = u / (u.norm() + 1e-12)
                        p.add_(-pilot_scale * mag * u)
                return

            # --- Case 2: standard secant chord step (only the 2 checks you want) ---
            denom = r_n - prev['residual']              # (1) compute denom
            if abs(denom) >= self.defaults['eps']:      # (2) if denom not too small, update
                step_scale = r_n / denom
                i = 0
                for g in self.param_groups:
                    for p in g['params']:
                        diff = p.data - prev['params'][i]
                        p.add_(-diff * step_scale)
                        i += 1

            # (3) save the pre-update snapshot as the new "prev"
            self.state['secant_prev'] = {'params': params_n_snapshot, 'residual': r_n}

    def sgd_root_step(self, curr_loss, lr=0.01):
        # want to find where f(x) = c
        # define g(x) = 0.5[f(x) - c]^2
        # gradient g(x) = [f(x) - c] gradient f(x)
        # x(t+1) = x(t) - eta[f(x)-c] gradient f(x)
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    p.add_(-lr*(curr_loss-self.c)*p.grad)
            