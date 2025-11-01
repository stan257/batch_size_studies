from unittest.mock import Mock

import pytest

from batch_size_studies.definitions import LossType
from batch_size_studies.training_utils import reverse_eta_adjustment_theoretical


class TestReverseEtaAdjustmentTheoretical:
    """Tests for the `reverse_eta_adjustment_theoretical` utility function."""

    @pytest.mark.parametrize(
        "exp_config, base_value, expected_result",
        [
            # Case 1: Non-MLP experiment.
            # adj_factor=1.0 (mocked).
            # theoretical_divisor=2 (since loss is not MSE).
            (
                {"adj_factor": 1.0, "loss_type": LossType.XENT},
                20.0,
                10.0,  # (20.0 / 1.0) / 2.0
            ),
            # Case 2: MLP in SP with SGD.
            # adj_factor = 2.0 (mocked from gamma**(2/L)).
            # theoretical_divisor=2 (non-MSE).
            (
                {"adj_factor": 2.0, "loss_type": LossType.XENT},
                20.0,
                5.0,  # (20.0 / 2.0) / 2.0
            ),
            # Case 3: MLP in muP with Adam.
            # adj_factor = 32.0 (mocked from gamma**(1/L) * sqrt(N)).
            # theoretical_divisor=2/num_outputs = 2/10 = 0.2 (MSE with classification).
            (
                {"adj_factor": 32.0, "loss_type": LossType.MSE, "num_outputs": 10},
                64.0,
                10.0,  # (64.0 / 32.0) / (2.0 / 10.0)
            ),
            # Case 4: Non-MLP with MSE loss (regression).
            # adj_factor=1.0 (mocked).
            # theoretical_divisor=2/1=2 (MSE, num_outputs defaults to 1).
            (
                {"adj_factor": 1.0, "loss_type": LossType.MSE},
                20.0,
                10.0,  # (20.0 / 1.0) / 2.0
            ),
        ],
    )
    def test_theoretical_adjustment_scenarios(self, exp_config, base_value, expected_result):
        # 1. Setup: Create a mock experiment.
        # Use a spec to ensure that accessing an unset attribute raises an AttributeError,
        # which allows getattr's default value to be used correctly.
        spec_attrs = ["get_adjusted_eta", "loss_type"]
        if "num_outputs" in exp_config:
            spec_attrs.append("num_outputs")

        mock_exp = Mock(spec=spec_attrs)
        mock_exp.get_adjusted_eta.return_value = exp_config["adj_factor"]
        mock_exp.loss_type = exp_config["loss_type"]
        if "num_outputs" in exp_config:
            mock_exp.num_outputs = exp_config["num_outputs"]

        # Define a simple function to be wrapped, which returns a constant value
        base_func = lambda b: base_value

        # 2. Action: Apply the theoretical adjustment
        adjusted_func = reverse_eta_adjustment_theoretical(base_func, mock_exp)
        # The batch_size argument doesn't affect the outcome in this test
        result = adjusted_func(batch_size=32)

        # 3. Assert: Check if the result matches the expected calculation
        assert result == pytest.approx(expected_result)
