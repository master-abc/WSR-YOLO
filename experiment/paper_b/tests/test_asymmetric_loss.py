from __future__ import annotations

import unittest

import torch

from algorithm.asymmetric_loss import AsymmetricFocalBCE


class AsymmetricFocalBCETest(unittest.TestCase):
    def test_preserves_positive_bce_when_positive_gamma_is_zero(self) -> None:
        loss = AsymmetricFocalBCE(gamma_negative=2.0, gamma_positive=0.0)
        logits = torch.tensor([[0.0, 2.0]])
        targets = torch.ones_like(logits)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        torch.testing.assert_close(loss(logits, targets), expected)

    def test_focuses_negative_loss_on_hard_false_alarms(self) -> None:
        loss = AsymmetricFocalBCE(
            gamma_negative=2.0, gamma_positive=0.0, negative_weight=2.0
        )
        logits = torch.tensor([[-6.0, 2.0]])
        targets = torch.zeros_like(logits)
        values = loss(logits, targets)
        self.assertGreater(float(values[0, 1]), float(values[0, 0]) * 1000.0)

    def test_rejects_invalid_configuration_and_shape(self) -> None:
        with self.assertRaises(ValueError):
            AsymmetricFocalBCE(gamma_negative=-1.0)
        loss = AsymmetricFocalBCE()
        with self.assertRaises(ValueError):
            loss(torch.zeros(1, 2), torch.zeros(1, 3))


if __name__ == "__main__":
    unittest.main()
