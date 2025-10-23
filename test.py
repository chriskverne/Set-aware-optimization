import torch
import torch.nn as nn
import torch.optim as optim

# # Create some sample dataset
# def f(x):
#     return torch.sin(10*x) + torch.cos(3*x)

# Create some sample dataset
def f(x):
    return torch.sin(x)

n_samples = 1000
x = torch.linspace(-10, 10, n_samples) #[1000, 1]
x = x.unsqueeze(1) # [1000, 1] Adds another postion at last index 1
y = f(x)

# Sample Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # simple 1d input
        self.input = nn.Linear(1, 128)
        self.hidden1 = nn.Linear(128, 64)
        self.hidden2 = nn.Linear(64,64)
        self.output = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.tanh(self.input(x))
        x = self.tanh(self.hidden1(x))
        x = self.tanh(self.hidden2(x))
        return self.output(x)

# # Iterate over epochs
# model = MLP()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# loss_fn = nn.MSELoss()
# n_epochs = 1000
# for epoch in range(n_epochs):
#     pred = model(x)
#     loss = loss_fn(pred, y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     print(f'Epoch: {epoch}, Loss: {loss}')

# Custom Rootfinding optimizer
from Optimizers.Optim import RootFinding

model = MLP()
optimizer = RootFinding(model.parameters(), gamma=0.90, inner_steps=3)
n_epochs = 100
loss_fn = nn.MSELoss()
for epoch in range(n_epochs):
    preds = model(x)
    loss = loss_fn(preds, y)

    optimizer.iterate_inner_loop(model, loss_fn, x, y)
    print(f'Epoch: {epoch}, Loss: {loss}')


