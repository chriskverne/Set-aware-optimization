import torch
import torch.nn as nn

# ------------------------------
# Data
# ------------------------------
def f(x):
    return torch.sin(x)

x = torch.linspace(-10, 10, 1000).unsqueeze(1)
y = f(x)

# ------------------------------
# Model
# ------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# ------------------------------
# Optimizer (your AdamRoot)
# ------------------------------
from Optimizers.AdamRoot import AdamRoot

model = MLP()
loss_fn = nn.MSELoss()
optimizer = AdamRoot(model.parameters(), lr=0.001, gamma=0.8, ema_c=0.9, dir_norm_clip=100000, alpha_max=10)

# ------------------------------
# Training loop
# ------------------------------
for epoch in range(10000):

    def closure():
        optimizer.zero_grad(set_to_none=True)
        preds = model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")
