import dynamiqs as dq
import jax.numpy as jnp

def get_measurement_probabilities(xvec, yvec, wigner, alpha_ks):
    w_ks = []
    for alpha_k in alpha_ks:
        xi = jnp.abs(xvec - alpha_k[0]).argmin()
        yi = jnp.abs(yvec - alpha_k[1]).argmin()
        w_ks += [1/2 * (1 + jnp.pi/2 * wigner[xi, yi])]
    
    return w_ks

def get_measurement_operators(N, alpha_ks):
    Eks = []
    for alpha_k in alpha_ks:
        alpha_z = alpha_k[0] + 1j* alpha_k[1]
        a=dq.destroy(N)
        D = lambda alph: dq.expm(alph*a.dag() - alph.conjugate()*a)
        P = dq.expm(1j*jnp.pi*a.dag()@a)
        E = 1/2 * (dq.eye(N) + D(alpha_z) @ P @ D(alpha_z).dag())
        Eks += [E]
    
    return Eks



if __name__ == "__main__":
        
    N = 20
    xvec, yvec, wigner = dq.wigner(dq.fock(N, 1))
    alpha_ks = [(1,1), (1,-1), (-1,-1), (-1,1)]

    wks = get_measurement_probabilities(xvec, yvec, wigner, alpha_ks)
    Eks = get_measurement_operators(N, alpha_ks)

    print("W_k list:")
    print(wks)

    print("Eks:")
    print(Eks)