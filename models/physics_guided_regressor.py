"""
Physics-Guided Neural Network (PGNN) for Gas Turbine Derating.
Enforces thermodynamic monotonicity: dP/dT_amb <= 0 (ambient temperature penalty constraint).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsGuidedMLP(nn.Module):
    def __init__(self, in_features: int = 6, hidden_dim: int = 64):
        super(PhysicsGuidedMLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.gelu(self.fc1(x))
        h2 = F.gelu(self.fc2(h1))
        h3 = F.gelu(self.fc3(h2))
        return self.out(h3)


def physics_loss_fn(
    model: nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    t_ambient_col_idx: int = 0,
    lambda_mono: float = 0.5,
) -> tuple:
    """
    Computes data MSE loss + thermodynamic monotonicity violation penalty:
    Monotonicity constraint: dP/dT_ambient <= 0 (power must decrease with rising ambient heat).
    """
    x_var = x.clone().detach().requires_grad_(True)
    y_pred = model(x_var)

    loss_data = F.mse_loss(y_pred, y_true)

    # Compute derivative dP / dT_amb using autograd
    grad_outputs = torch.ones_like(y_pred)
    grad_x = torch.autograd.grad(y_pred, x_var, grad_outputs=grad_outputs, create_graph=True)[0]
    dP_dTamb = grad_x[:, t_ambient_col_idx]

    # Penalty for positive derivative (violation of thermodynamic derating)
    violation = torch.clamp(dP_dTamb, min=0.0)
    loss_mono = torch.mean(violation ** 2)

    total_loss = loss_data + lambda_mono * loss_mono
    return total_loss, loss_data, loss_mono
