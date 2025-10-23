import torch
from torch.optim.optimizer import Optimizer

# Needs to override __init__(), step(), zero_grad()
class RootFinding(Optimizer):
    def __init__(self, params, gamma=0.95, inner_steps=3):
        # Defaults Stores hyperparameters
        defaults = dict(gamma=gamma, inner_steps=inner_steps)
        # Gamma represents c decrease,
        # inner_steps represents how many times we solve the equation f = c
        super(RootFinding,self).__init__(params, defaults)
        # Represents current root we try to solve f = c for (This changes throguh optimization)
        self.c =  None 

    def update_root(self, curr_loss):
        # We can find more sophisticated methods for this approach
        self.c = curr_loss * self.defaults['gamma']

    def iterate_inner_loop(self, model, loss_fn, inputs, targets):
        # Set inital target if not set
        if self.c is None:
            curr_loss = loss_fn(model(inputs), targets)
            self.update_root(curr_loss)
        
        for i in range(self.defaults['inner_steps']):
            # Get current loss
            curr_loss = loss_fn(model(inputs), targets)
            # compute grads
            self.zero_grad()
            curr_loss.backward()
            # Do single step
            self.step(curr_loss)
        
        # Update target c
        self.update_root(loss_fn(model(inputs), targets))

    def step(self, curr_loss):
        # param_groups is a set of dictionaires where each dict has some params and some hyperparams for those params
        for group in self.param_groups:
            # Iterate over each parameter in the parameter group
            for parameter in group['params']:
                grad = parameter.grad

                #  Simple step x(t+1) = x(t) - (f - c)/f'
                parameter.minus_((curr_loss - self.c)/grad)

