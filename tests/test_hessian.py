import jax
import jax.numpy as jnp
import numpy as np
import pytest

from batch_size_studies.hessian import JaxHessian

# --- Test Setup: A simple quadratic problem where the Hessian is known ---


@pytest.fixture
def quadratic_problem():
    """
    Sets up a test problem with a known quadratic loss: L(w) = 1/2 * w^T @ A @ w.
    For this loss, the Hessian is exactly the matrix A, which allows for direct
    comparison of eigenvalues, trace, and other properties.
    """
    # 1. Define the known Hessian matrix 'A'
    dim = 10
    key = jax.random.PRNGKey(42)
    # Create a random symmetric matrix for the Hessian
    A_random = jax.random.normal(key, (dim, dim))
    A = (A_random + A_random.T) / 2
    true_eigenvalues, _ = np.linalg.eigh(A)

    # 2. Define a dummy model and initialize parameters with zeros
    # The model simply returns its parameters. This allows the loss function
    # to be defined directly on the parameters, as required for this test.
    model = lambda params, inputs: params
    params = jnp.zeros((dim,))

    # 3. Define the loss function that implements the quadratic form
    def loss_fn(w_vec, _):  # Second arg is a dummy target
        # Loss = 0.5 * w^T * A * w
        return 0.5 * jnp.dot(w_vec, jnp.dot(A, w_vec))

    # 4. Create a mock data loader
    # The data itself doesn't matter for this loss function, but the
    # hessian class needs a data loader to iterate over.
    dummy_inputs = jnp.ones((1, dim))
    dummy_targets = jnp.ones(1)
    mock_data_loader = [(dummy_inputs, dummy_targets)]

    return {
        "model": model,
        "loss_fn": loss_fn,
        "params": params,
        "data_loader": mock_data_loader,
        "true_hessian": A,
        "true_eigenvalues": sorted(true_eigenvalues, reverse=True),  # Largest to smallest
        "true_trace": np.trace(A),
    }


# --- Tests for JaxHessian methods ---


def test_hvp(quadratic_problem):
    """
    Tests the Hessian-Vector-Product directly against the known matrix A.
    H @ v should be equal to A @ v.
    """
    hessian_calc = JaxHessian(
        model=quadratic_problem["model"],
        loss_fn=quadratic_problem["loss_fn"],
        data_loader=quadratic_problem["data_loader"],
    )
    params = quadratic_problem["params"]
    A = quadratic_problem["true_hessian"]
    key = jax.random.PRNGKey(0)

    # Create a random vector 'v' with the same pytree structure as params
    v_vec = jax.tree_util.tree_map(lambda x: jax.random.normal(key, x.shape), params)

    # Calculate HVP using our class
    hvp_vec = hessian_calc._hvp_full_dataset(params, v_vec)

    # Calculate the true product A @ v
    true_hvp = jnp.dot(A, v_vec)

    np.testing.assert_allclose(hvp_vec, true_hvp, rtol=1e-5)


def test_eigenvalues(quadratic_problem):
    """
    Tests that the power iteration method finds the correct top eigenvalues.
    """
    hessian_calc = JaxHessian(
        model=quadratic_problem["model"],
        loss_fn=quadratic_problem["loss_fn"],
        data_loader=quadratic_problem["data_loader"],
    )
    params = quadratic_problem["params"]
    true_eigenvalues_sorted_by_value = quadratic_problem["true_eigenvalues"]
    key = jax.random.PRNGKey(1)

    # Power iteration finds eigenvalues with the largest magnitude (absolute value).
    # The test fixture provides eigenvalues sorted by value, so we must re-sort
    # them by magnitude to create the correct ground truth for this test.
    true_top_eigenvalues_by_mag = sorted(true_eigenvalues_sorted_by_value, key=abs, reverse=True)

    # Find the top 3 eigenvalues
    top_n = 3
    estimated_eigenvalues, _ = hessian_calc.eigenvalues(params, key, top_n=top_n, max_iter=500)

    np.testing.assert_allclose(estimated_eigenvalues, true_top_eigenvalues_by_mag[:top_n], atol=1e-1)


def test_trace(quadratic_problem):
    """
    Tests that Hutchinson's method correctly estimates the trace.
    """
    hessian_calc = JaxHessian(
        model=quadratic_problem["model"],
        loss_fn=quadratic_problem["loss_fn"],
        data_loader=quadratic_problem["data_loader"],
    )
    params = quadratic_problem["params"]
    true_trace = quadratic_problem["true_trace"]
    key = jax.random.PRNGKey(2)

    estimated_trace, _ = hessian_calc.trace(params, key, max_iter=500)

    # The estimate is stochastic, so we use a larger tolerance.
    np.testing.assert_allclose(estimated_trace, true_trace, rtol=0.1)


def test_density(quadratic_problem):
    """
    Tests the eigenvalue density estimation from Stochastic Lanczos Quadrature.
    """
    hessian_calc = JaxHessian(
        model=quadratic_problem["model"],
        loss_fn=quadratic_problem["loss_fn"],
        data_loader=quadratic_problem["data_loader"],
    )
    params = quadratic_problem["params"]
    true_eigenvalues = quadratic_problem["true_eigenvalues"]
    key = jax.random.PRNGKey(3)

    ritz_values_list, _ = hessian_calc.density(params, key, num_iterations=10, num_runs=50)
    all_ritz_values = np.concatenate(ritz_values_list)

    # Check that the range of Ritz values (the estimated eigenvalues)
    # is contained within the range of the true eigenvalues.
    min_true_eig, max_true_eig = min(true_eigenvalues), max(true_eigenvalues)
    min_ritz, max_ritz = np.min(all_ritz_values), np.max(all_ritz_values)

    # Add a small tolerance for floating point issues
    assert min_ritz >= min_true_eig - 0.1 * abs(min_true_eig)
    assert max_ritz <= max_true_eig + 0.1 * abs(max_true_eig)


def test_mlp_model_smoke_test():
    """
    A smoke test to ensure JaxHessian can be initialized and run with a real
    MLP model without crashing. This does not check for correctness of the
    Hessian values, only that the computation graph is valid.
    """

    from batch_size_studies.definitions import Parameterization
    from batch_size_studies.models import MLP

    key = jax.random.PRNGKey(42)
    data_key, hessian_key = jax.random.split(key, 2)

    # 1. Initialize a real model and its parameters
    model = MLP(Parameterization.SP)
    dummy_widths = [28 * 28, 128, 10]
    dummy_inputs = jax.random.normal(data_key, (1, 28 * 28))  # Example for MNIST
    params = model.init_params(42, dummy_widths)

    # 2. Define a realistic loss function
    # For this test, a simple MSE loss is sufficient.
    def mse_loss(logits, targets):
        return jnp.mean((logits - targets) ** 2)

    # 3. Create a mock data loader with appropriate shapes
    dummy_targets = jax.random.normal(data_key, (1, 10))  # Example for 10 classes
    mock_data_loader = [(dummy_inputs, dummy_targets)]

    # 4. Initialize and run a Hessian calculation
    hessian_calc = JaxHessian(
        model=model,
        loss_fn=mse_loss,
        data_loader=mock_data_loader,
    )

    # 5. Run one of the methods to ensure it executes without error
    try:
        # Check if we can compute the top eigenvalue
        eigenvalues, _ = hessian_calc.eigenvalues(params, hessian_key, top_n=1, max_iter=2)
        # Check if we can compute the trace
        trace, _ = hessian_calc.trace(params, hessian_key, max_iter=2)

        # Check that the results have the expected types/shapes
        assert isinstance(eigenvalues, list) and len(eigenvalues) == 1
        assert isinstance(eigenvalues[0], jax.Array)
        assert np.isscalar(trace)

    except Exception as e:
        pytest.fail(f"JaxHessian failed to run with MLP model: {e}")
