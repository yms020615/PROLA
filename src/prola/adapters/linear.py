import torch
from torch import nn
from torch.nn import functional as F


class PROLALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        principal_rank=4,
        orthogonal_rank=4,
        alpha=1.0,
        svd_iterations=16,
        initialize=True,
    ):
        super().__init__()
        if min(principal_rank, orthogonal_rank) < 1:
            raise ValueError("Both branch ranks must be positive.")
        if max(principal_rank, orthogonal_rank) > min(linear.weight.shape):
            raise ValueError("Ranks cannot exceed the matrix dimensions.")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.principal_rank = principal_rank
        self.orthogonal_rank = orthogonal_rank
        self.principal_scale = alpha / principal_rank
        self.orthogonal_scale = alpha / orthogonal_rank
        self.svd_iterations = svd_iterations
        self.weight = linear.weight.requires_grad_(False)
        self.bias = linear.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)
        options = dict(device=linear.weight.device, dtype=torch.float32)
        self.principal_a = nn.Parameter(torch.zeros(principal_rank, self.in_features, **options))
        self.principal_b = nn.Parameter(torch.zeros(self.out_features, principal_rank, **options))
        self.orthogonal_a = nn.Parameter(torch.empty(orthogonal_rank, self.in_features, **options))
        self.orthogonal_b = nn.Parameter(torch.zeros(self.out_features, orthogonal_rank, **options))
        nn.init.normal_(self.orthogonal_a, std=0.02)
        self.register_buffer("u", torch.zeros(self.out_features, principal_rank, **options))
        self.register_buffer("v", torch.zeros(self.in_features, principal_rank, **options))
        self.register_buffer("initial_a", torch.zeros_like(self.principal_a))
        self.register_buffer("initial_b", torch.zeros_like(self.principal_b))
        self.register_buffer("initialized", torch.tensor(False, device=linear.weight.device))
        if initialize:
            self.initialize()

    @torch.no_grad()
    def initialize(self):
        if self.initialized:
            return
        with torch.autocast(device_type=self.weight.device.type, enabled=False):
            weight = self.weight.float()
            if self.svd_iterations > 0:
                u, values, v = torch.svd_lowrank(
                    weight,
                    q=max(self.principal_rank, self.orthogonal_rank),
                    niter=self.svd_iterations,
                )
            else:
                u, values, vh = torch.linalg.svd(weight, full_matrices=False)
                v = vh.T
            r = self.principal_rank
            root = (values[:r].clamp_min(0) / self.principal_scale).sqrt()
            self.u.copy_(u[:, :r])
            self.v.copy_(v[:, :r])
            self.principal_b.copy_(u[:, :r] * root)
            self.principal_a.copy_(root[:, None] * v[:, :r].T)
            self.initial_a.copy_(self.principal_a)
            self.initial_b.copy_(self.principal_b)
            residual = self.principal_scale * (self.principal_b @ self.principal_a)
            self.weight.sub_(residual.to(self.weight.dtype))
            self.initialized.fill_(True)

    @torch.no_grad()
    def residualize_loaded_base(self):
        if not self.initialized:
            raise RuntimeError("The adapter checkpoint is not initialized.")
        with torch.autocast(device_type=self.weight.device.type, enabled=False):
            self.weight.sub_(
                (self.principal_scale * (self.initial_b @ self.initial_a)).to(self.weight.dtype)
            )

    def projected_factors(self):
        with torch.autocast(device_type=self.weight.device.type, enabled=False):
            b = self.orthogonal_b.float()
            a = self.orthogonal_a.float()
            u, v = self.u.float(), self.v.float()
            return b - u @ (u.T @ b), a - (a @ v) @ v.T

    def forward(self, x):
        if not self.initialized:
            raise RuntimeError("Initialize PROLA or load its checkpoint before calling forward.")
        base = F.linear(x, self.weight, self.bias)
        b, a = self.projected_factors()
        inputs = x.to(self.principal_a.dtype)
        principal = F.linear(F.linear(inputs, self.principal_a), self.principal_b)
        orthogonal = F.linear(F.linear(inputs, a), b)
        return base + (self.principal_scale * principal + self.orthogonal_scale * orthogonal).to(
            base.dtype
        )
