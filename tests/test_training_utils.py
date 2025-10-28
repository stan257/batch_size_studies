from unittest.mock import Mock

import pytest

from batch_size_studies.definitions import LossType, OptimizerType, Parameterization
from batch_size_studies.experiments import LinearStudentExperiment, MLPStudentExperiment
from batch_size_studies.training_utils import reverse_eta_adjustment_theoretical


class TestReverseEtaAdjustmentTheoretical:
    """Tests for the `reverse_eta_adjustment_theoretical` utility function."""

    @pytest.mark.parametrize(
        "exp_config, base_value, expected_result",
        [
            # Case 1: Non-MLP experiment.
            # adj_factor=1.0 (since it's not an MLPStudentExperiment).
            # theoretical_divisor=2 (since loss is not MSE).
            (
                {"spec": LinearStudentExperiment, "loss_type": LossType.XENT},
                20.0,
                10.0,  # (20.0 / 1.0) / 2.0
            ),
            # Case 2: MLP in SP with SGD.
            # adj_factor = gamma**(2/L) * 1 = 2.0**(2/2) = 2.0.
            # theoretical_divisor=2 (non-MSE).
            (
                {
                    "spec": MLPStudentExperiment,
                    "parameterization": Parameterization.SP,
                    "optimizer": OptimizerType.SGD,
                    "gamma": 2.0,
                    "L": 2,
                    "N": 128,
                    "loss_type": LossType.XENT,
                },
                20.0,
                5.0,  # (20.0 / 2.0) / 2.0
            ),
            # Case 3: MLP in muP with Adam.
            # adj_factor = gamma**(1/L) * sqrt(N) = 4.0**(1/1) * sqrt(64) = 4 * 8 = 32.0.
            # theoretical_divisor=2/num_outputs = 2/10 = 0.2 (MSE with classification).
            (
                {
                    "spec": MLPStudentExperiment,
                    "parameterization": Parameterization.MUP,
                    "optimizer": OptimizerType.ADAM,
                    "gamma": 4.0,
                    "L": 1,
                    "N": 64,
                    "loss_type": LossType.MSE,
                    "num_outputs": 10,
                },
                64.0,
                10.0,  # (64.0 / 32.0) / (2.0 / 10.0)
            ),
            # Case 4: Non-MLP with MSE loss (regression).
            # adj_factor=1.0 (non-MLP).
            # theoretical_divisor=2/1=2 (MSE, num_outputs defaults to 1).
            (
                {"spec": LinearStudentExperiment, "loss_type": LossType.MSE},
                20.0,
                10.0,  # (20.0 / 1.0) / 2.0
            ),
        ],
    )
    def test_theoretical_adjustment_scenarios(self, exp_config, base_value, expected_result):
        """Tests that theoretical adjustments are applied correctly across different experiment types."""
        # 1. Setup: Create a mock experiment from the configuration
        mock_exp = Mock(spec=exp_config["spec"])
        for key, value in exp_config.items():
            if key != "spec":
                setattr(mock_exp, key, value)

        # Define a simple function to be wrapped, which returns a constant value
        base_func = lambda b: base_value

        # 2. Action: Apply the theoretical adjustment
        adjusted_func = reverse_eta_adjustment_theoretical(base_func, mock_exp)
        # The batch_size argument doesn't affect the outcome in this test
        result = adjusted_func(batch_size=32)

        # 3. Assert: Check if the result matches the expected calculation
        assert result == pytest.approx(expected_result)
