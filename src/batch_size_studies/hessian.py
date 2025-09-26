"""
A JAX-based adapter for Hessian analysis, inspired by the PyHessian library.

This module provides tools to compute properties of the Hessian matrix (e.g.,
eigenvalues, trace, density) for models written in JAX, using matrix-free
methods like power iteration and Hutchinson's method.

The original PyHessian library can be found at:
https://github.com/amirgholami/PyHessian
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.tree_util import tree_leaves, tree_map, tree_structure, tree_unflatten

# =============================================================================
# Pytree Utility Functions
# =============================================================================


def tree_dot(xs, ys):
    """Computes the dot product of two pytrees of arrays."""
    return sum(jnp.sum(x * y) for x, y in zip(tree_leaves(xs), tree_leaves(ys)))


def tree_add(xs, ys, alpha=1.0):
    """Computes xs + alpha * ys for two pytrees."""
    return tree_map(lambda x, y: x + alpha * y, xs, ys)


def tree_norm(xs):
    """Computes the L2 norm of a pytree of arrays."""
    return jnp.sqrt(tree_dot(xs, xs))


def tree_normalize(xs):
    """Normalizes a pytree of arrays to have a unit L2 norm."""
    norm = tree_norm(xs)
    return tree_map(lambda x: x / (norm + 1e-6), xs)


def tree_orthonormal(w, v_list):
    """Makes pytree w orthogonal to each pytree in v_list and normalizes."""
    for v in v_list:
        w = tree_add(w, v, alpha=-tree_dot(w, v))
    return tree_normalize(w)


def tree_random_like(key, target_tree, rademacher=False):
    """Creates a random pytree with the same structure as target_tree."""
    target_struct = tree_structure(target_tree)
    leaves = tree_leaves(target_tree)
    if rademacher:
        # Generate Rademacher random variables {-1, 1}
        new_leaves = [jax.random.rademacher(key, shape=leaf.shape, dtype=leaf.dtype) for leaf in leaves]
    else:
        # Generate standard normal random variables
        keys = jax.random.split(key, len(leaves))
        new_leaves = [jax.random.normal(k, shape=l.shape, dtype=l.dtype) for k, l in zip(keys, leaves)]
    return tree_unflatten(target_struct, new_leaves)


# =============================================================================
# JaxHessian Class
# =============================================================================


class JaxHessian:
    """
    The class to compute Hessian information for a JAX/Flax model.
    It computes:
        i) The top n eigenvalue(s).
        ii) The trace of the Hessian.
        iii) The estimated eigenvalue density.
    """

    def __init__(self, model, loss_fn, data_loader):
        """
        Args:
            model: A Flax nn.Module.
            loss_fn: A loss function of the form `loss_fn(logits, targets)`.
            data_loader: A generator yielding batches of (inputs, targets).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.data_loader = data_loader
        self.num_batches = len(data_loader)

    def _get_loss_fn_for_hessian(self, params, batch):
        """A closure for the model's loss function for a single batch."""
        inputs, targets = batch
        return self.loss_fn(self.model.apply({"params": params}, inputs), targets)

    @partial(jit, static_argnums=(0,))
    def _hvp_single_batch(self, params, v, batch):
        """Computes the Hessian-vector product for a single batch."""
        # Define the loss function for the given batch
        loss_fn_for_batch = lambda p: self._get_loss_fn_for_hessian(p, batch)
        # JAX's way to compute HVP is jvp(grad(f)).
        # It computes the gradient function, and then the Jacobian-vector product
        # of that gradient function.
        return jax.jvp(jax.grad(loss_fn_for_batch), (params,), (v,))[1]

    def _hvp_full_dataset(self, params, v):
        """Computes the Hessian-vector product averaged over the full dataset."""
        hvp_total = tree_map(jnp.zeros_like, params)
        for batch in self.data_loader:
            hvp_batch = self._hvp_single_batch(params, v, batch)
            # Accumulate the HVP from each batch
            hvp_total = tree_map(lambda x, y: x + y, hvp_total, hvp_batch)
        # Average over the number of batches
        return tree_map(lambda x: x / self.num_batches, hvp_total)

    def eigenvalues(self, params, key, max_iter=100, tol=1e-3, top_n=1):
        """
        Computes the top_n eigenvalues using the Power Iteration method.
        """
        assert top_n >= 1
        eigenvalues = []
        eigenvectors = []

        computed_dim = 0
        while computed_dim < top_n:
            # Generate a random vector for initialization
            key, subkey = jax.random.split(key)
            v = tree_random_like(subkey, params)
            v = tree_normalize(v)

            current_eigenvalue = None
            for j in range(max_iter):
                # Orthogonalize v against previously found eigenvectors
                v = tree_orthonormal(v, eigenvectors)

                Hv = self._hvp_full_dataset(params, v)
                tmp_eigenvalue = tree_dot(Hv, v)

                # Update v for the next iteration by normalizing the HVP result
                v = tree_normalize(Hv)

                # Check for convergence
                if current_eigenvalue is None:
                    current_eigenvalue = tmp_eigenvalue
                else:
                    if abs(current_eigenvalue - tmp_eigenvalue) / (abs(current_eigenvalue) + 1e-6) < tol:
                        break
                    else:
                        current_eigenvalue = tmp_eigenvalue

            eigenvalues.append(current_eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1

        return eigenvalues, eigenvectors

    def trace(self, params, key, max_iter=100):
        """
        Computes the trace of the Hessian using Hutchinson's method.
        """
        trace_estimates = []
        for i in range(max_iter):
            key, subkey = jax.random.split(key)
            # Generate a Rademacher random vector (entries are -1 or 1)
            v = tree_random_like(subkey, params, rademacher=True)
            Hv = self._hvp_full_dataset(params, v)
            trace_estimates.append(tree_dot(Hv, v))

        return np.mean(trace_estimates), trace_estimates

    def density(self, params, key, num_iterations=100, num_runs=10):
        """
        Computes the estimated eigenvalue density using the Stochastic Lanczos Quadrature algorithm.
        """
        eigen_list_full = []
        weight_list_full = []

        for k in range(num_runs):
            key, v_key, lanczos_key = jax.random.split(key, 3)
            v = tree_random_like(v_key, params, rademacher=True)
            v = tree_normalize(v)

            # Standard Lanczos algorithm initialization
            v_list = [v]
            alpha_list = []
            beta_list = []

            # Initial step
            w_prime = self._hvp_full_dataset(params, v)
            alpha = tree_dot(w_prime, v)
            alpha_list.append(alpha)
            w = tree_add(w_prime, v, alpha=-alpha)

            for i in range(1, num_iterations):
                beta = tree_norm(w)
                beta_list.append(beta)

                if beta == 0.0:
                    # If beta is 0, re-orthogonalize with a new random vector
                    lanczos_key, subkey = jax.random.split(lanczos_key)
                    w_rand = tree_random_like(subkey, params)
                    v = tree_orthonormal(w_rand, v_list)
                else:
                    v = tree_map(lambda x: x / beta, w)

                v_list.append(v)

                w_prime = self._hvp_full_dataset(params, v)
                alpha = tree_dot(w_prime, v)
                alpha_list.append(alpha)

                # w_tmp = w_prime - alpha * v
                w_tmp = tree_add(w_prime, v, alpha=-alpha)
                # w = w_tmp - beta * v_list[-2]
                w = tree_add(w_tmp, v_list[-2], alpha=-beta)

            # Construct the tridiagonal matrix T
            T = (
                jnp.diag(jnp.array(alpha_list))
                + jnp.diag(jnp.array(beta_list), k=1)
                + jnp.diag(jnp.array(beta_list), k=-1)
            )

            # Eigen-decomposition of T gives Ritz values and weights
            eigenvalues, eigenvectors = jnp.linalg.eigh(T)
            weights = jnp.power(eigenvectors[0, :], 2)

            eigen_list_full.append(list(np.array(eigenvalues)))
            weight_list_full.append(list(np.array(weights)))

        return eigen_list_full, weight_list_full
