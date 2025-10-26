"""
Newton (your original) converges fast near the root but can explode when gradients are tiny or Hessian effects are wild; damping could help.
Secant (α-wise) is a great compromise: cheap, 1-D, often stable, and doesn’t require exact line search.
Bisection/Brent (α-wise) are rock-solid if you can bracket.
Param-wise secant is the simplest gradient-free swap, but it’s the most brittle in high-D because it treats each coordinate as a separate 1-D problem tied to a single scalar residual.
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

    def update_root(self, curr_loss):
        # We can find more sophisticated methods for this approach
        self.c = curr_loss * self.defaults['gamma']

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
    
    def secant_step(self, curr_loss, pilot_scale=1e-3):
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


    # def secant_step(self,curr_loss,pilot_scale=1e-3):
    #     # Uses x(t+1) = x(t) = f/f' however it tries to approximate f' using 2 FPs
    #     # f' = f(t+1) - f(t)/ x(t+1) - x(t)
    #     # x0 and x1 start at random values where ideally there is a root between them somewhere

    #     # For multivariable functions:
    #     # r(x) = loss(x) - c.
    #     # Uses: x_{n+1} = x_n - r_n * (x_n - x_{n-1}) / (r_n - r_{n-1})

    #     with torch.no_grad():
    #         r_n = curr_loss - self.c

    #         # Get previous params / loss
    #         prev = self.state.get('secant_prev')

    #         # Store current params as previous for next step
    #         curr_snapshot = []
    #         for group in self.param_groups:
    #             for p in group['params']:
    #                 curr_snapshot.append(p.data.clone())

    #         # In case of first call just sore x_n and r_n
    #         if prev is None:
    #             self.state['secant_prev'] = {'params': curr_snapshot, 'residual': r_n.clone()}
    #             for g in self.param_groups:
    #                 for p in g['params']:
    #                     # keep step size roughly in units of the parameter magnitude
    #                     mag = p.data.norm().item()
    #                     if mag == 0.0:
    #                         mag = 1.0
    #                     u = torch.randn_like(p.data)
    #                     u = u / (u.norm() + 1e-12)  # unit direction
    #                     p.add_(-pilot_scale * mag * u)
    #             return
            
    #         # r_n - - r_{n-1} (Consider checking ig this is too small)
    #         denom = r_n - prev['residual']

    #         # Uses: x_{n+1} = x_n - r_n * (x_n - x_{n-1}) / (r_n - r_{n-1})
    #         i = 0
    #         step_scale = r_n / denom
    #         for group in self.param_groups:
    #             for p in group['params']:
    #                 #x_n - x_{n-1}
    #                 diff = p.data - prev['params'][i]
    #                 # print(r_n, diff, denom)
    #                 p.add_(-diff * step_scale)
    #                 i+= 1

    #         # Refresh cache for next call (store *post-step* current as "prev" for n+1)
    #         updated_snapshot = []
    #         for group in self.param_groups:
    #             for p in group['params']:
    #                 updated_snapshot.append(p.data.clone())
    #         self.state['secant_prev'] = {'params': updated_snapshot, 'residual': r_n.clone()}
