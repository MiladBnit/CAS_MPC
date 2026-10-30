"""Kernels of the SVGD update (paper, Section 4.2).

    k_theta(a, b) = exp(-||a - b||^2 / h_theta)
    k_Phi(a, b)   = exp(-||Phi_t(a) - Phi_t(b)||^2 / h_Phi)                 eq:kernels
    k_omega       = k_Phi + omega k_theta

    grad_a k_Phi(a, b)   = -(2 / h_Phi) k_Phi(a, b) J(a)^T (Phi_t(a) - Phi_t(b))   eq:kernel-grad
    grad_a k_theta(a, b) = -(2 / h_theta) k_theta(a, b) (a - b)

Bandwidths by the median heuristic, h = med_{i != j} ||z_i - z_j||^2 / log N,
with a floor. Kernel matrices are indexed K[j, i] = k(theta^(j), theta^(i)),
matching eq:svgd-dir, where the sum runs over the first argument j.

Kernel variants: "param" (k_theta) and "ca" (k_omega = k_Phi + omega k_theta;
omega = 0 gives k_Phi alone).
"""

from __future__ import annotations

import jax.numpy as jnp

VARIANTS = ("param", "ca")


def sq_dists(Z):
    """Pairwise squared distances, D[j, i] = ||z_j - z_i||^2, Z of shape (N, d)."""
    diff = Z[:, None, :] - Z[None, :, :]
    return jnp.sum(diff ** 2, axis=-1)


def median_bandwidth(D, h_floor, scale_h=1.0):
    """Median heuristic over the off-diagonal pairs, times scale_h (h = 1 for a
    single particle, whose kernel is then k = 1 whatever h)."""
    N = D.shape[0]
    if N == 1:
        return jnp.asarray(1.0)
    iu = jnp.triu_indices(N, k=1)
    return jnp.maximum(scale_h * jnp.median(D[iu]) / jnp.log(N), h_floor)


def gaussian_kernel(Z, h_floor, scale_h=1.0):
    """K[j, i] = exp(-||z_j - z_i||^2 / h), and the bandwidth h."""
    D = sq_dists(Z)
    h = median_bandwidth(D, h_floor, scale_h)
    return jnp.exp(-D / h), h


def grad_k_theta(thetas, K, h):
    """G[j, i] = grad_{theta_j} k_theta(theta_j, theta_i), shape (N, N, n_theta)."""
    diff = thetas[:, None, :] - thetas[None, :, :]
    return -(2.0 / h) * K[:, :, None] * diff


def grad_k_Phi(Phis, Js, K, h):
    """G[j, i] = grad_{theta_j} k_Phi(theta_j, theta_i) (eq:kernel-grad),
    Phis (N, d_Phi), Js (N, d_Phi, n_theta); shape (N, N, n_theta)."""
    diff = Phis[:, None, :] - Phis[None, :, :]                  # (N, N, d)
    return -(2.0 / h) * K[:, :, None] * jnp.einsum("jdn,jid->jin", Js, diff)


def kernel_and_grad(variant, thetas, Phis=None, Js=None, omega=0.0, h_floor=1e-8, scale_h=1.0,
                    scale_h_phi=None):
    """Kernel matrix K[j, i] and gradients G[j, i] = grad_{theta_j} k(theta_j, theta_i)
    for the given variant. Also returns the bandwidths used. scale_h scales the
    median-heuristic bandwidth of k_theta, scale_h_phi that of k_Phi (None: scale_h)."""
    info = {}
    if variant not in VARIANTS:
        raise ValueError(f"unknown kernel variant {variant!r}")
    K_th, h_th = gaussian_kernel(thetas, h_floor, scale_h)
    G_th = grad_k_theta(thetas, K_th, h_th)
    info["h_theta"] = h_th
    if variant == "param":
        return K_th, G_th, info
    K_ph, h_ph = gaussian_kernel(Phis, h_floor, scale_h if scale_h_phi is None else scale_h_phi)
    G_ph = grad_k_Phi(Phis, Js, K_ph, h_ph)
    info["h_Phi"] = h_ph
    return K_ph + omega * K_th, G_ph + omega * G_th, info


def svgd_direction(K, G, scores, alpha=1.0):
    """eq:svgd-dir: phi_i = (1/N) sum_j [ K[j, i] score_j + alpha G[j, i] ]
    (alpha = 1 in the paper; alpha weights the repulsion)."""
    N = K.shape[0]
    return (K.T @ scores + alpha * jnp.sum(G, axis=0)) / N
