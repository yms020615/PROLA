import io

import torch
from torch import nn

from prola import PROLALinear, PROLAStep


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    original = nn.Linear(16, 12)
    x = torch.randn(8, 16)
    expected = original(x).detach()
    model = PROLALinear(original, principal_rank=4, orthogonal_rank=4, alpha=1, svd_iterations=0)
    torch.testing.assert_close(model(x), expected, atol=2e-6, rtol=2e-6)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    step = PROLAStep(model, optimizer)
    target = torch.randn(8, 12)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - target).square().mean()
        loss.backward()
        step.step(max_grad_norm=1.0)
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = PROLALinear(nn.Linear(16, 12), initialize=False)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored(x), model(x))
    print(
        f"Initial function preserved; scaling={model.principal_scale:g}; loss={loss.item():.4f}; checkpoint restored."
    )


if __name__ == "__main__":
    main()
