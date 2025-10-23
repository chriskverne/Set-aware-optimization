import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(7)

# 1) Data (1D -> 1D regression)
n = 240
X = torch.empty(n, 1).uniform_(-3.0, 3.0)
true_fn = lambda x: torch.sin(1.5 * x) + 0.3 * x
y = true_fn(X) + 0.10 * torch.randn_like(X)

# 2) Tiny MLP
class TinyMLP(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, 1)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

def half_mse(pred, target):
    return 0.5 * torch.mean((pred - target) ** 2)

# 3) Train Model with f(x) = c method
def train_lsrto(model, X, y, gamma=0.9, outer_levels=12, inner_max=18, eps_grad=1e-12, tol=1e-6):
    train_losses, targets = [], []
    with torch.no_grad():
        cur_loss = float(half_mse(model(X), y).item())
    c = gamma * cur_loss  # Decide new target a bit smaller than current target

    # How many times to set a new target for c
    for _ in range(outer_levels):

        # Specific Method for finding f(x) =  c and how many iterations to do so.
        for _ in range(inner_max):
            # Forward pass to get pred and loss
            pred = model(X)
            f = half_mse(pred, y) 
            # Clear old gradients
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            # Compute new gradients
            f.backward()

            # Compute ||grad||^2
            g2 = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    g2 += float((p.grad ** 2).sum().item())
            g2 = g2 + eps_grad

            # Newtons method step = (f - c) / ||grad||^2
            step = (float(f.item()) - c) / g2

            # theta <- theta - step * grad
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-step)

            with torch.no_grad():
                cur_loss = float(half_mse(model(X), y).item())
            train_losses.append(cur_loss)
            targets.append(c)

            if abs(cur_loss - c) < tol:
                break

        # new target level based on current loss
        with torch.no_grad():
            cur_loss = float(half_mse(model(X), y).item())
        c = gamma * cur_loss

    return train_losses, targets, model

# Run
model = TinyMLP(hidden=128)
train_losses, targets, model = train_lsrto(model, X, y, gamma=0.9, outer_levels=50, inner_max=1)

# Print final loss and make a simple training-loss curve
print("Final training loss:", train_losses[-1] if train_losses else None)

plt.figure()
plt.plot(train_losses, label="training loss")
plt.xlabel("iteration")
plt.ylabel("0.5 * MSE")
plt.title("LSRTO training loss")
plt.legend()
plt.show()
