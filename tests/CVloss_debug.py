import torch
from anemoi.training.losses.spectral import SpectralCRPSLoss, SpectralCVLoss
import math

torch.manual_seed(0)
ny, nx = 969, 849
batch_size = 1
n_steps = 1
n_members = 2
n_variables = 1

# Shape: batch, time, ensemble, flattened grid, variable
prediction = torch.randn(
    batch_size, n_steps, n_members, ny * nx, n_variables,
    requires_grad=True,
)
target = torch.randn(
    batch_size, n_steps, 1, ny * nx, n_variables,
)

bin_weights = torch.randn(30)

loss_fn = SpectralCRPSLoss(x_dim=nx, y_dim=ny)
loss_cv = SpectralCVLoss(x_dim=nx, y_dim=ny, min_freq=math.log10(1/ny), max_freq=math.log10(0.5), bin_weights=bin_weights)

loss = loss_cv(prediction, target)

print("Loss:", loss.item())

loss.backward()
print("Gradient shape:", prediction.grad.shape)
print("Finite gradients:", torch.isfinite(prediction.grad).all().item())