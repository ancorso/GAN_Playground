import torch
from torch.nn.utils import spectral_norm
import math

x = torch.linspace(-math.pi, math.pi, 10)
y = torch.sin(x)
y

p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)
model = torch.nn.Sequential(
    SpectralNorm(torch.nn.Linear(3, 1), n_power_iterations = 100),
    torch.nn.Flatten(0, 1)
)

import numpy as np
print(np.array(model[0].weight.data))
print(np.array(model[0].bias.data))

loss_fn = torch.nn.MSELoss(reduction='mean')
learning_rate = 1e-6

y_pred = model(xx)
y_pred
loss = loss_fn(y_pred, y)
loss
model.zero_grad()

loss.backward()

for param in model.parameters():
    print(param)
    print(param.grad)
model[0].weight

model[0].compute_weight()

# Update the weights using gradient descent. Each parameter is a Tensor, so
# we can access its gradients like we did before.
with torch.no_grad():
    for param in model.parameters():
        param -= learning_rate * param.grad

