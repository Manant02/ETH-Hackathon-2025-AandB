#%%
import numpy as np
import matplotlib.pyplot as plt
import dynamiqs as dq
import jax.numpy as jnp


# %% simulate a lossy quantum harmonic oscillator

# parameters
n = 16          # Hilbert space dimension
omega = 1.0     # frequency
kappa = 0.1     # decay rate
alpha0 = 1.0    # initial coherent state amplitude
T = 2 * jnp.pi  # total evolution time (one full revolution)

# initialize operators, initial state and saving times
a = dq.destroy(n)
H = omega * a.dag() @ a
jump_ops = [jnp.sqrt(kappa) * a]
psi0 = dq.coherent(n, alpha0)
tsave = jnp.linspace(0, T, 101)

# run simulation
result = dq.mesolve(H, jump_ops, psi0, tsave)
print(result)


