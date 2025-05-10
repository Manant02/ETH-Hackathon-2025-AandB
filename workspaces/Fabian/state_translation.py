import dynamiqs as dq
import jax.numpy as jnp
import jax
from dynamiqs import *
import matplotlib.pyplot as plt
import numpy as np


def state_translation(state_dm, dim_new, x_max, p_max):
    dim_old = state_dm.shape[0]
    rho = dq.to_numpy(state_dm)

    rho_new = np.zeros((dim_new, dim_new), dtype=complex)
    rho_new[:dim_old, :dim_old] = rho

    state = dq.to_qutip(rho_new)

    displ_vector = -x_max - p_max * 1j

    displace_op = dq.displace(dim_new, displ_vector)

    state_displaced = displace_op @ state @ displace_op.dag()
    return state_displaced


state = coherent_dm(10, 1)

print(state_translation(state, 20, 1, 0))
