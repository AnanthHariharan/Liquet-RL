import torch
from torch import nn


class LTCCell(nn.Module):
    """Liquid Time-Constant recurrent cell."""

    def __init__(self, input_size: int, hidden_size: int, dt: float = 0.02):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt

        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        # Time constants are positive. We store them in log-space for stability.
        self.log_tau = nn.Parameter(torch.zeros(hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.constant_(self.weight_hh, 0.0)
        nn.init.constant_(self.bias, 0.0)
        nn.init.constant_(self.log_tau, 1.0)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Forward pass of the LTC cell.

        Args:
            x: input tensor of shape ``(batch, input_size)``
            h: previous hidden state of shape ``(batch, hidden_size)``
        Returns:
            Next hidden state tensor of shape ``(batch, hidden_size)``
        """
        tau = torch.relu(self.log_tau) + 1e-3  # ensure positivity
        x_dot = -h / tau + x @ self.weight_ih.T + h @ self.weight_hh.T + self.bias
        h_next = h + self.dt * x_dot
        return h_next
