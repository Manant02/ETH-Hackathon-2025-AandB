import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pickle


def plot_wigner(xvals, yvals, wigner_values):
    plt.contourf(
        xvals,
        yvals,
        wigner_values.T,
        levels=100,
        cmap="seismic",
        vmin=-2 / np.pi,
        vmax=2 / np.pi,
    )
    plt.colorbar()


def correcting_wigner(xvec, yvec, wigner_noisy):
    wigner_noisy = jnp.nan_to_num(wigner_noisy, nan=0.0)

    def calculate_offset(wigner_noisy):
        # 1. compute local mean of W and local mean of W² over a small window
        win = 10  # size of the patch in pixels
        mean1 = sp.ndimage.uniform_filter(wigner_noisy, size=win)
        mean2 = sp.ndimage.uniform_filter(wigner_noisy**2, size=win)

        # 2. estimate the local variance: Var = E[W²] – (E[W])²
        local_var = mean2 - mean1**2

        # 3. pick a threshold for “flatness” (tune this)
        var_thr = np.percentile(local_var, 5)
        # e.g. take the bottom 5% of variances

        # 4. build a mask of flat regions
        flat_mask = local_var <= var_thr

        # 5. estimate b by averaging wigner_noisy over all flat pixels
        b_est = np.mean(wigner_noisy[flat_mask])

        print("Estimated offset b =", b_est)
        return b_est

    # only keep important features
    def flattening_flat_regions(wigner_noisy, thr, b_est):

        # 1. build binary masks of positive and negative
        pos_mask = (wigner_noisy > b_est).astype(float)
        neg_mask = (wigner_noisy < b_est).astype(float)

        # 2. choose a window size (in pixels) over which to look for clusters
        win = 11  # e.g. 11×11 neighborhood

        # 3. compute local counts (actually local fractions) of positives/negatives
        # uniform_filter sums (here since mask is 0/1, it gives count) then we normalize by total window size
        local_pos_frac = sp.ndimage.uniform_filter(pos_mask, size=win)
        local_neg_frac = sp.ndimage.uniform_filter(neg_mask, size=win)

        # 4. define your cluster thresholds
        # for instance, if more than 30% of the window is positive (resp. negative):
        thr = thr

        cluster_mask = (local_pos_frac > thr) & (local_neg_frac > thr)

        # 5. make a heavily‐smoothed version of the whole noisy Wigner
        smooth_heavy = sp.ndimage.gaussian_filter(wigner_noisy, sigma=5)

        # 6. combine: apply the heavy smooth only inside clusters
        wigner_denoised = wigner_noisy.copy()
        wigner_denoised_np = np.array(wigner_denoised)
        wigner_denoised_np[cluster_mask] = smooth_heavy[cluster_mask]

        return wigner_denoised_np, cluster_mask

    b_est = calculate_offset(wigner_noisy)
    wigner_noisy_offsetted = wigner_noisy - b_est

    cluster_mask = flattening_flat_regions(wigner_noisy, 0.4, b_est)[1]
    relevant_mask = np.logical_not(cluster_mask)
    relevant_mask_true_values = relevant_mask.sum()

    integral = (
        np.sum(wigner_noisy_offsetted[relevant_mask])
        / wigner_noisy_offsetted.shape[0] ** 2
        * 12**2
    )

    wigner_noisy_corrected = wigner_noisy_offsetted / integral

    wigner_denoised_corrected = flattening_flat_regions(wigner_noisy_corrected, 0.3, 0)[
        0
    ]

    return xvec, yvec, wigner_denoised_corrected


def load_wigner(file):
    with open(file, "rb") as f:
        wigner_fct = pickle.load(f)
    return wigner_fct


wigner_fct = load_wigner("data/synthetic/noisy_wigner_5.pickle")
xvec, yvec, W_synthetic = correcting_wigner(*wigner_fct)

"""N = 20


def cat_factory(n, alpha=1):
    cat_n = dq.coherent(N, alpha)
    for i in range(1, n):
        cat_n += dq.coherent(N, np.exp(1j * 2 * i * np.pi / n) * alpha)

    return cat_n / cat_n.norm()


def cat_dm_factory(n, alpha=1):
    return cat_factory(n, alpha=alpha).todm()


state = cat_factory(2)
xvec, yvec, wigner = dq.wigner(state)

a = 1.0
b = 0.3
sigma = 0.1
noise = np.random.normal(0.0, scale=sigma, size=wigner.shape)

wigner_noisy = a * wigner + b * np.ones_like(wigner) + noise"""


def plot_wigner(xvals, yvals, wigner_values):
    plt.contourf(
        xvals,
        yvals,
        wigner_values.T,
        levels=100,
        cmap="seismic",
        vmin=-2 / np.pi,
        vmax=2 / np.pi,
    )
    plt.colorbar()


"""xvec, yvec, W = correcting_wigner(xvec, yvec, wigner_noisy)"""
plot_wigner(xvec, yvec, W_synthetic)
plt.show()
