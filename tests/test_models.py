import jax.random as jr
import pytest

from batch_size_studies.definitions import Parameterization
from batch_size_studies.models import MLP


def test_mlp_initialization_and_validation():
    """Tests that the MLP class initializes correctly and validates its scale."""
    # Should succeed
    mlp_sp = MLP(parameterization=Parameterization.SP, gamma=1.0)
    assert mlp_sp.parameterization == Parameterization.SP

    mlp_mup = MLP(parameterization=Parameterization.MUP, gamma=0.95)
    assert mlp_mup.parameterization == Parameterization.MUP

    # Should fail with an invalid type for parameterization
    with pytest.raises(
        TypeError,
        match="parameterization must be a member of the Parameterization enum, but got type str.",
    ):
        # This call correctly triggers the error by passing an invalid string.
        MLP(parameterization="invalid_scale", gamma=1.0)


def test_init_params_returns_correct_shapes():
    """Tests that the init_params method generates weights with the correct shapes."""
    mlp = MLP(parameterization=Parameterization.SP, gamma=1.0)
    widths = [128, 256, 64, 1]
    params = mlp.init_params(init_key=0, widths=widths)

    assert len(params) == 3
    assert params[0].shape == (128, 256)
    assert params[1].shape == (256, 64)
    assert params[2].shape == (64, 1)


def test_sp_and_mup_forward_pass_are_different():
    """
    Tests that SP and muP models produce different outputs for the same inputs
    and parameters, due to the different output layer scaling.
    """
    widths = [10, 20, 1]
    mlp_sp = MLP(parameterization=Parameterization.SP, gamma=1.0)
    mlp_mup = MLP(parameterization=Parameterization.MUP, gamma=1.0)

    params = mlp_sp.init_params(init_key=42, widths=widths)
    x = jr.normal(jr.key(1), (1, 10))

    output_sp = mlp_sp(params, x)
    output_mup = mlp_mup(params, x)

    assert output_sp.shape == (1, 1)
    assert output_mup.shape == (1, 1)
    assert output_sp != output_mup
