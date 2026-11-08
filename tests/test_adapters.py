import copy
import io
import unittest

import torch
from torch import nn

from prola import PROLALinear, PROLAStep, inject_prola
from prola.training import adaptation_state, load_adaptation


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)

    def test_initial_function_and_nonmutating_forward(self):
        base = nn.Linear(12, 10)
        inputs = torch.randn(3, 12)
        expected = base(inputs).detach()
        model = PROLALinear(base, svd_iterations=0)
        self.assertEqual(model.principal_scale, 0.25)
        self.assertEqual(model.orthogonal_scale, 0.25)
        torch.testing.assert_close(model(inputs), expected)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        (model(inputs).sum() + model(inputs).square().mean()).backward()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_orthogonal_complement(self):
        layer = PROLALinear(nn.Linear(12, 10), svd_iterations=0)
        with torch.no_grad():
            layer.orthogonal_b.normal_()
        b, a = layer.projected_factors()
        delta = b @ a
        self.assertLess(float((layer.u.T @ delta).norm()), 1e-5)
        self.assertLess(float((delta @ layer.v).norm()), 1e-5)

    def test_scaled_increments_preserve_adam_moments(self):
        model = PROLALinear(nn.Linear(12, 10), svd_iterations=0)
        reference = copy.deepcopy(model)
        for p, q in zip(model.parameters(), reference.parameters()):
            if p.requires_grad:
                p.grad = torch.randn_like(p)
                q.grad = p.grad.clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3, weight_decay=0.05)
        scaler = PROLAStep(model, optimizer)
        entries = scaler.multipliers()
        mean = torch.cat([entries[0][1].flatten(), entries[2][1].flatten()]).mean()
        torch.testing.assert_close(mean, torch.ones_like(mean))
        before = {id(p): p.clone() for p, _ in entries}
        reference_optimizer.step()
        scaler.step()
        refs = dict(zip((id(p) for p in model.parameters()), reference.parameters()))
        for p, multiplier in entries:
            expected = before[id(p)] + multiplier * (refs[id(p)] - before[id(p)])
            torch.testing.assert_close(p, expected)
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    optimizer.state[p][key], reference_optimizer.state[refs[id(p)]][key]
                )

    def test_nonfinite_step_does_not_modify_parameters(self):
        model = PROLALinear(nn.Linear(12, 10))
        model(torch.randn(2, 12)).sum().backward()
        model.principal_a.grad[0, 0] = float("nan")
        before = model.principal_a.clone()
        optimizer = torch.optim.AdamW(model.parameters())
        with self.assertRaises(FloatingPointError):
            PROLAStep(model, optimizer).step()
        torch.testing.assert_close(model.principal_a, before)
        self.assertFalse(optimizer.state)

    def test_full_checkpoint_roundtrip(self):
        model = PROLALinear(nn.Linear(12, 10))
        x = torch.randn(3, 12)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        other = PROLALinear(nn.Linear(12, 10), initialize=False)
        other.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(model(x), other(x))

    def test_adapter_checkpoint_and_single_residualization(self):
        source = nn.Sequential(nn.Linear(12, 10))
        original = copy.deepcopy(source)
        inject_prola(source, target_modules=("0",), svd_iterations=0)
        x = torch.randn(2, 12)
        source(x).sum().backward()
        PROLAStep(source, torch.optim.AdamW(source.parameters())).step()
        state = adaptation_state(source)
        self.assertNotIn("0.weight", state)
        inject_prola(original, target_modules=("0",), initialize=False)
        load_adaptation(original, state)
        torch.testing.assert_close(original(x), source(x))
        with self.assertRaises(ValueError):
            load_adaptation(original, state)


if __name__ == "__main__":
    unittest.main()
