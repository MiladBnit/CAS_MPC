"""CAS-MPC: control-aware online learning of gray-box neural state-space models for stochastic MPC.

Equations are referred to by their LaTeX labels in the paper (e.g. eq:score).
"""

import jax

# Float64 everywhere, so that the JAX and CasADi models agree to numerical
# precision.
jax.config.update("jax_enable_x64", True)
