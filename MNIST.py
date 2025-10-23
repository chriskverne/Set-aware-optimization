import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 0. LSRTO Optimizer and MNIST_MLP Class (Unchanged) ---
# (Including them here for a complete, runnable script)
class LSRTO(Optimizer):
    def __init__(self, params, gamma=0.9, eps_grad=1e-12):
        defaults = dict(gamma=gamma, eps_grad=eps_grad)
        super(LSRTO, self).__init__(params, defaults)
    
    # We will now call this for every mini-batch
    def update_target(self, current_loss: float):
        gamma = self.defaults['gamma']
        self.state['c'] = gamma * current_loss

    @torch.no_grad()
    def step(self, current_loss: float):
        # We will set 'c' at every step now, so no need for the initial check
        c = self.state['c']
        eps_grad = self.defaults['eps_grad']
        # Compute L2 norm of gradient vector
        grad_norm_sq = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad_norm_sq += torch.sum(p.grad ** 2).item()
        grad_norm_sq += eps_grad
        # Newton's method step
        step_size = (current_loss - c) / grad_norm_sq
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.add_(p.grad, alpha=-step_size)

class MNIST_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# --- 1. Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 256
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transform),
    batch_size=1000)

# --- 2. Model, Loss, and Optimizer ---
model = MNIST_MLP().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = LSRTO(model.parameters(), gamma=0.9)

# --- 3. The FULLY STOCHASTIC Training Loop ---
epochs = 5
batch_losses = []
print("\nStarting training with Fully Stochastic LSRTO...")

for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        
        # Update the target 'c' based on this specific mini-batch's loss
        optimizer.update_target(loss.item())
        
        # Perform the step
        optimizer.step(loss.item())
        
        batch_losses.append(loss.item())

# --- 4. Final Evaluation ---
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        correct += (pred.argmax(1) == y_batch).sum().item()
        total += y_batch.size(0)

print(f"\nTraining Finished!")
print(f"Final Test Accuracy: {100 * correct / total:.2f}%")

# --- 5. Visualize the (likely extreme) Instability ---
plt.figure(figsize=(12, 6))
plt.plot(batch_losses)
plt.title("Mini-Batch Loss during Fully Stochastic LSRTO Training")
plt.xlabel("Training Step (Batch)")
plt.ylabel("Cross-Entropy Loss")
plt.ylim(0, 5) # Cap the y-axis to prevent spikes from ruining the view
plt.grid(True)
plt.show()