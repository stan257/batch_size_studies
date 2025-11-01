import jax.numpy as jnp
import jax.random as jr
import pytest

from batch_size_studies.definitions import Parameterization
from batch_size_studies.models import MLP, LinearModel


def test_mlp_initialization_and_validation():
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


class TestLinearModel:
    @pytest.fixture
    def linear_model(self):
        return LinearModel()

    def test_init_params_returns_correct_shapes_and_zeros(self, linear_model):
        input_dim, output_dim = 10, 3
        widths = [input_dim, output_dim]
        W = linear_model.init_params(init_key=0, widths=widths)

        assert isinstance(W, jnp.ndarray)
        assert W.shape == (input_dim, output_dim)
        assert jnp.all(W == 0)

    def test_init_params_raises_error_for_invalid_widths(self, linear_model):
        with pytest.raises(ValueError, match="expects `widths` to be a list of length 2"):
            linear_model.init_params(init_key=0, widths=[10, 5, 3])  # More than 2

        with pytest.raises(ValueError, match="expects `widths` to be a list of length 2"):
            linear_model.init_params(init_key=0, widths=[10])  # Less than 2

    def test_forward_pass_computes_correctly(self, linear_model):
        input_dim, output_dim = 5, 2
        key = jr.PRNGKey(42)
        x = jr.normal(key, (input_dim,))

        # Use non-zero params for a more robust check
        W = jnp.ones((input_dim, output_dim))
        params = W

        output = linear_model(params, x)

        expected_output = jnp.dot(x, W)
        assert output.shape == (output_dim,)
        assert jnp.allclose(output, expected_output)

    def test_forward_pass_with_batch_input(self, linear_model):
        batch_size, input_dim, output_dim = 4, 5, 2
        key = jr.PRNGKey(42)
        x_batch = jr.normal(key, (batch_size, input_dim))

        W = jnp.ones((input_dim, output_dim))
        params = W

        output = linear_model(params, x_batch)

        expected_output = jnp.dot(x_batch, W)
        assert output.shape == (batch_size, output_dim)
        assert jnp.allclose(output, expected_output)

    def test_forward_pass_raises_error_for_invalid_params(self, linear_model):
        x = jnp.ones((5,))
        # `params` should be an array, not a list containing an array
        # JAX error messages can vary, so we use a regex to match either form.
        with pytest.raises(TypeError, match="(requires ndarray or scalar arguments|Error interpreting argument)"):
            linear_model([jnp.zeros((5, 1))], x)

        # `params` should be an array, not a string
        with pytest.raises(TypeError, match="(requires ndarray or scalar arguments|Error interpreting argument)"):
            linear_model("not_an_array", x)
