import torch

from .linear import PROLALinear


class PROLAStep:
    def __init__(self, model, optimizer, strength=1.0, eps=1e-6):
        self.model = model
        self.optimizer = optimizer
        self.strength = strength
        self.eps = eps

    @torch.no_grad()
    def multipliers(self):
        entries = []
        for layer in self.model.modules():
            if not isinstance(layer, PROLALinear):
                continue
            factors = [
                (layer.principal_a, layer.principal_b),
                (layer.orthogonal_a, layer.orthogonal_b),
            ]
            if any(a.grad is None or b.grad is None for a, b in factors):
                continue
            scores = [a.grad.float().norm(dim=1) + b.grad.float().norm(dim=0) for a, b in factors]
            means = torch.stack([s.mean() for s in scores])
            rho = (means + self.eps) / (means.sum() + 2 * self.eps)
            rank = sum(s.numel() for s in scores)
            for (a, b), score, mass in zip(factors, scores, rho):
                pi = (score + self.eps) / (score.sum() + score.numel() * self.eps)
                scale = 1 + self.strength * (rank * mass * pi - 1)
                entries.extend([(a, scale[:, None]), (b, scale[None, :])])
        return entries

    @torch.no_grad()
    def step(self, max_grad_norm=None):
        parameters = [
            p
            for group in self.optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not parameters:
            raise RuntimeError("No gradients are available for this optimizer step.")
        finite = torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all()
        if not finite:
            raise FloatingPointError("Nonfinite gradients; optimizer step was not applied.")
        entries = self.multipliers()
        if not entries:
            raise RuntimeError("No PROLA gradients were found.")
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
        before = [(p, p.detach().clone(), scale) for p, scale in entries]
        self.optimizer.step()
        for parameter, old, scale in before:
            parameter.copy_(old + (parameter - old) * scale.to(parameter))
