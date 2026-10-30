"""Multilayer perceptron helpers: layout of the flat parameter vector and its
initialization (the network of the learned growth rate, bioreactor.BioModel)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def mlp_shapes(n_hidden: int, n_layers: int, n_out: int, n_in: int):
    """Weight and bias shapes [(W, b), ...], W of shape (n_out, n_in)."""
    sizes = [n_in] + [n_hidden] * n_layers + [n_out]
    return [((sizes[k + 1], sizes[k]), (sizes[k + 1],)) for k in range(len(sizes) - 1)]


def n_theta(shapes) -> int:
    return sum(int(np.prod(sw)) + int(np.prod(sb)) for sw, sb in shapes)


def unflatten(theta, shapes, reshape=None):
    """Flat theta -> [(W, b), ...]. Order: W1, b1, W2, b2, ... (row-major W)."""
    reshape = reshape or (lambda v, shape: v.reshape(shape))
    layers, k = [], 0
    for sw, sb in shapes:
        nw, nb = int(np.prod(sw)), int(np.prod(sb))
        layers.append((reshape(theta[k:k + nw], sw), theta[k + nw:k + nw + nb]))
        k += nw + nb
    return layers


def init_theta(key, shapes, last_layer_scale: float):
    """Glorot-uniform weights, zero biases, last layer scaled down."""
    parts = []
    keys = jax.random.split(key, len(shapes))
    for idx, ((sw, sb), kk) in enumerate(zip(shapes, keys)):
        n_out, n_in = sw
        lim = np.sqrt(6.0 / (n_in + n_out))
        W = jax.random.uniform(kk, sw, minval=-lim, maxval=lim)
        if idx == len(shapes) - 1:
            W = W * last_layer_scale
        parts += [W.ravel(), jnp.zeros(sb)]
    return jnp.concatenate(parts)


# ---------------------------------------------------------------- backends
