import torch
from torch.optim.optimizer import Optimizer

class LSRTO(Optimizer):
    """
    Implements the Level-Set Regularization for Task-Oriented Optimization algorithm.
    """
    def __init__(self, params, gamma=0.9, eps_grad=1e-12):
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"Invalid gamma: {gamma}")
        
        defaults = dict(gamma=gamma, eps_grad=eps_grad)
        super(LSRTO, self).__init__(params, defaults)
        
        # State to hold the target loss 'c'
        self.state['c'] = None

    def update_target(self, current_loss: float):
        """
        Updates the target loss 'c' based on the current loss.
        This mimics the "outer loop" of the original algorithm.
        """
        gamma = self.defaults['gamma']
        self.state['c'] = gamma * current_loss

    @torch.no_grad()
    def step(self, current_loss: float):
        """
        Performs a single optimization step.

        Args:
            current_loss (float): The current loss value, required to compute the step size.
        """
        if self.state.get('c') is None:
            # First time step is called, initialize the target loss 'c'
            self.update_target(current_loss)
            
        c = self.state['c']
        eps_grad = self.defaults['eps_grad']
        
        # --- 1) Compute the squared L2 norm of the gradient: ||g||^2 ---
        grad_norm_sq = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad_norm_sq += torch.sum(p.grad ** 2).item()
        
        grad_norm_sq += eps_grad
        
        # --- 2) Compute the Newton step size: (f - c) / ||g||^2 ---
        step_size = (current_loss - c) / grad_norm_sq
        
        # --- 3) Update the parameters: theta <- theta - step_size * grad ---
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.add_(p.grad, alpha=-step_size)



import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Using the same data and model from your original code
torch.manual_seed(7)
n = 240
X = torch.empty(n, 1).uniform_(-3.0, 3.0)
true_fn = lambda x: torch.sin(1.5 * x) + 0.3 * x
y = true_fn(X) + 0.10 * torch.randn_like(X)

class TinyMLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, 1)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

def half_mse(pred, target):
    return 0.5 * torch.mean((pred - target) ** 2)

# --- NEW Training Loop ---
model = TinyMLP(hidden=128)
loss_fn = half_mse

# Hyperparameters for the training logic
outer_levels = 100
inner_max = 1

# Initialize our custom optimizer
optimizer = LSRTO(model.parameters(), gamma=0.9)

train_losses = []
targets = []

# This is the "outer loop" for setting new targets
for outer_step in range(outer_levels):
    
    # 1. Calculate current loss to set the new target
    with torch.no_grad():
        current_loss = loss_fn(model(X), y).item()
    
    # 2. Update the optimizer's internal target 'c'
    optimizer.update_target(current_loss)
    targets.extend([optimizer.state['c']] * inner_max)

    # This is the "inner loop" for stepping towards the target
    for inner_step in range(inner_max):
        # A. Clear old gradients
        optimizer.zero_grad()
        
        # B. Forward pass to get the loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # C. Compute gradients
        loss.backward()
        
        # D. Perform the update step, passing the current loss
        optimizer.step(loss.item())
        
        train_losses.append(loss.item())

# --- Plotting the results ---
print("Final training loss:", train_losses[-1] if train_losses else None)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss (f)")
plt.plot(targets, linestyle='--', label="Target Loss (c)")
plt.xlabel("Iteration (Inner Steps)")
plt.ylabel("0.5 * MSE")
plt.title("LSRTO Training with Optimizer Class")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()