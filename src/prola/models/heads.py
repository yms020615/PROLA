import torch
from torch import nn
from torch.nn import functional as F

from .blocks import DepthwiseSwiGLUStack2D, DepthwiseSwiGLUStack3D


class SpatialCorrectionHead(nn.Module):
    def __init__(self, surface_channels=3, upper_channels=65, hidden_surface=32, hidden_upper=512):
        super().__init__()
        self.surface_net = DepthwiseSwiGLUStack3D(
            3 * surface_channels, hidden_surface, surface_channels, (1, 3, 3), (0, 1, 1)
        )
        self.upper_net = DepthwiseSwiGLUStack3D(
            3 * upper_channels, hidden_upper, upper_channels, (1, 3, 3), (0, 1, 1)
        )

    def forward(self, context_surface, context_upper, endpoint_surface, endpoint_upper):
        surface = torch.cat((endpoint_surface, context_surface[:, 0], context_surface[:, 1]), dim=1)
        upper = torch.cat(
            (
                endpoint_upper.flatten(1, 2),
                context_upper[:, 0].flatten(1, 2),
                context_upper[:, 1].flatten(1, 2),
            ),
            dim=1,
        )
        ds = self.surface_net(surface.unsqueeze(2)).squeeze(2)
        du = self.upper_net(upper.unsqueeze(2)).squeeze(2).reshape_as(endpoint_upper)
        return ds, du


class TemporalCorrectionHead(nn.Module):
    def __init__(
        self,
        surface_channels=3,
        upper_channels=65,
        hidden_surface=32,
        hidden_upper=512,
        num_leads=6,
    ):
        super().__init__()
        self.surface_basis = DepthwiseSwiGLUStack2D(
            5 * surface_channels, hidden_surface, 4 * surface_channels, 3, 1
        )
        self.upper_basis = DepthwiseSwiGLUStack2D(
            5 * upper_channels, hidden_upper, 12 * upper_channels, 3, 1
        )
        self.surface_gate = nn.Sequential(
            nn.Linear(5 * surface_channels + 16, 64), nn.SiLU(), nn.Linear(64, 4)
        )
        self.upper_gate = nn.Sequential(
            nn.Linear(5 * upper_channels + 16, 256), nn.SiLU(), nn.Linear(256, 12)
        )
        self.projector_lead_embedding = nn.Embedding(max(num_leads, 8), 16)
        nn.init.normal_(self.projector_lead_embedding.weight, std=0.02)
        for gate in (self.surface_gate, self.upper_gate):
            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)

    def forward(
        self,
        current_surface,
        current_upper,
        endpoint_surface,
        endpoint_upper,
        scaffold_surface,
        scaffold_upper,
    ):
        batch, leads, variables, height, width = scaffold_surface.shape
        surface = torch.cat(
            (
                current_surface,
                endpoint_surface,
                scaffold_surface[:, 0],
                endpoint_surface - current_surface,
                scaffold_surface[:, 0] - current_surface,
            ),
            dim=1,
        )
        current = current_upper.flatten(1, 2)
        endpoint = endpoint_upper.flatten(1, 2)
        first = scaffold_upper[:, 0].flatten(1, 2)
        upper = torch.cat((current, endpoint, first, endpoint - current, first - current), dim=1)
        basis_s = self.surface_basis(surface).reshape(batch, 4, variables, height, width)
        basis_u = self.upper_basis(upper).reshape(batch, 12, *scaffold_upper.shape[2:])
        embedding = self.projector_lead_embedding(torch.arange(1, leads + 1, device=surface.device))
        embedding = embedding.unsqueeze(0).expand(batch, -1, -1)
        pooled_s = F.adaptive_avg_pool2d(surface, 1).flatten(1).unsqueeze(1).expand(-1, leads, -1)
        pooled_u = F.adaptive_avg_pool2d(upper, 1).flatten(1).unsqueeze(1).expand(-1, leads, -1)
        gate_s = self.surface_gate(torch.cat((pooled_s, embedding), dim=-1)).softmax(-1)
        gate_u = self.upper_gate(torch.cat((pooled_u, embedding), dim=-1)).softmax(-1)
        return torch.einsum("btk,bkvhw->btvhw", gate_s, basis_s), torch.einsum(
            "btk,bkvchw->btvchw", gate_u, basis_u
        )
