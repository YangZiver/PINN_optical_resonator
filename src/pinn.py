# pinn.py
import numpy as np
import torch
from torch import nn
from torch import Tensor
import parameters
import utils
class Pinn(nn.Module):
    def __init__(
        self,
        input_dim: int = parameters.input_dim,
        output_dim: int = parameters.output_dim,
        hidden_dim: int = parameters.hidden_dim,
        num_hidden: int = parameters.num_hidden,
        round_num: int = 1,
        use_fourier: bool = True,
        fourier_dim: int = parameters.fourier_dim,
        fourier_scale: float = parameters.fourier_scale
    ):
        """
        define neural networks
        parameters:
        input_dim, dimension of input layers
        output_dim, dimension of output layers
        hidden_dim, dimension of hidden layers
        num_hidden, number of hidden neurals
        round_num, rounds in resonator
        use_fourier, if use fourier feature embedding
        """
        super().__init__()
        self.use_fourier = use_fourier
        self.fourier_dim = fourier_dim
        self.fourier_scale = fourier_scale
        if use_fourier:
            # Generate a fixed random Fourier feature matrix B of shape 
            # (input_dim, fourier_dim//2)
            B = torch.randn(input_dim, fourier_dim // 2) * fourier_scale
            self.register_buffer('B', B)
            # input dimension: fourier_dim
            self.inp = nn.Linear(fourier_dim, hidden_dim)
        else:
            self.inp = nn.Linear(input_dim, hidden_dim)
        self.hid = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden - 1)])
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.tanh
        self.round_num = round_num
        self.device = utils.setting_device()
        self.to(self.device)
        # SIREN initialize
        # for i, m in enumerate(self.modules()):
        #     if isinstance(m, nn.Linear):
        #         if i == 0:
        #             nn.init.uniform_(m.weight, -1 / m.in_features, 1 / m.in_features)
        #         else:
        #             nn.init.uniform_(m.weight, -np.sqrt(6 / m.in_features), np.sqrt(6 / m.in_features))
        #         nn.init.zeros_(m.bias)
        # xiavier initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,
                z_s: Tensor,
                T_s: Tensor) -> tuple[Tensor, Tensor]:
        """
        input: z, T
        output: u, v
        """
        x = torch.cat([z_s, T_s], dim=1)
        if self.use_fourier:
            x_proj = 2 * np.pi * x @ self.B  # (N, fourier_dim//2)
            x_fourier = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # (N, fourier_dim)
            h = self.act(self.inp(x_fourier))
        else:
            h = self.act(self.inp(x))
        for L in self.hid:
            h = self.act(L(h))
        o = self.out(h)
        u = o[:, 0:1]
        v = o[:, 1:2]
        return u, v



