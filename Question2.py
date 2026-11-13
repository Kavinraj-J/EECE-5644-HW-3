import sys
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.exceptions import ConvergenceWarning
 
warnings.filterwarnings("ignore", category=ConvergenceWarning)
rng = np.random.default_rng(2026)
 
#************* Question 2 Part 1 ************************
WEIGHTS = np.array([0.20, 0.30, 0.35, 0.15])       
MEANS = np.array([[0.0, 0.0],                       
                  [1.5, 0.5],                       
                  [6.0, 0.0],                       
                  [2.0, 6.0]])                      
COVS = np.array([[[1.0, 0.3], [0.3, 0.8]],         
                 [[0.7, -0.2], [-0.2, 1.1]],       
                 [[1.5, 0.0], [0.0, 0.5]],         
                 [[0.6, 0.2], [0.2, 1.2]]])        
 
d01 = np.linalg.norm(MEANS[0] - MEANS[1])
avg_eig = 0.5 * (np.linalg.eigvalsh(COVS[0]) + np.linalg.eigvalsh(COVS[1]))
 
 
def sample_gmm(N):
    z = rng.choice(4, size=N, p=WEIGHTS)
    X = np.empty((N, 2))
    for k in range(4):
        idx = z == k
        X[idx] = rng.multivariate_normal(MEANS[k], COVS[k], idx.sum())
    return X, z
 
 
#************* Question 2 Part 2 ************************
K_MAX = 10
N_FOLDS = 10
 
 
def cv_scores(X, n_init=2):
    """Average per-sample validation log-likelihood for K = 1..K_MAX.
    A model order that cannot be fit (training fold has fewer points than
    components) gets -inf, i.e. it can never be selected."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True,
               random_state=int(rng.integers(1 << 31)))
    scores = np.zeros(K_MAX)
    for K in range(1, K_MAX + 1):
        total, count = 0.0, 0
        for tr, va in kf.split(X):
            if len(tr) < K:
                total = -np.inf
                break
            gmm = GaussianMixture(n_components=K, covariance_type="full",
                                  n_init=n_init, max_iter=500, tol=1e-4,
                                  reg_covar=1e-6,
                                  random_state=int(rng.integers(1 << 31)))
            gmm.fit(X[tr])                               # EM = ML estimation
            total += gmm.score_samples(X[va]).sum()      # validation log-lik
            count += len(va)
        scores[K - 1] = total / count if np.isfinite(total) else -np.inf
    return scores
 
 
#************* Question 2 Part 3 ************************
if __name__ == "__main__":
    N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    N_LIST = [10, 100, 1000]
 
    print(f"Overlap check: ||mu0-mu1|| = {d01:.2f}, "
          f"sum of averaged eigenvalues = {avg_eig.sum():.2f}")

    Xbig, _ = sample_gmm(20000)
    g = GaussianMixture(4, n_init=5, random_state=0).fit(Xbig)
    order = np.argsort(g.weights_)
    print("EM check (N=20000, K=4): estimated weights",
          np.round(g.weights_[order], 3), "vs true", np.sort(WEIGHTS))
 
    selected = {N: np.zeros(N_REPS, dtype=int) for N in N_LIST}
    all_scores = {N: np.zeros((N_REPS, K_MAX)) for N in N_LIST}
    for r in range(N_REPS):
        for N in N_LIST:
            X, _ = sample_gmm(N)
            s = cv_scores(X)
            all_scores[N][r] = s
            selected[N][r] = int(np.argmax(s)) + 1
        if (r + 1) % 10 == 0:
            print(f"  finished {r + 1}/{N_REPS} repetitions", flush=True)
 
    rate = np.array([[np.mean(selected[N] == K) for K in range(1, K_MAX + 1)]
                     for N in N_LIST])
    header = "  N    | " + " ".join(f"K={K:<4d}" for K in range(1, K_MAX + 1))
    print("\nSelection rate (fraction of repetitions)\n" + header)
    for i, N in enumerate(N_LIST):
        print(f"{N:6d} | " + " ".join(f"{v:6.2f}" for v in rate[i]))
    for N in N_LIST:
        print(f"N={N}: mode = {np.bincount(selected[N]).argmax()}, "
              f"mean selected K = {selected[N].mean():.2f}")
 
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.27
    Ks = np.arange(1, K_MAX + 1)
    for i, N in enumerate(N_LIST):
        ax.bar(Ks + (i - 1) * w, rate[i], width=w, label=f"N={N}")
    ax.axvline(4, color="k", ls=":", lw=1)
    ax.set_xticks(Ks)
    ax.set_xlabel("GMM order K")
    ax.set_ylabel("Selection rate")
    ax.set_title(f"Q2: order selected by 10-fold CV ({N_REPS} repetitions; true K=4)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 2.8))
    im = ax.imshow(rate, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(K_MAX)); ax.set_xticklabels(Ks)
    ax.set_yticks(range(len(N_LIST))); ax.set_yticklabels([f"N={N}" for N in N_LIST])
    for i in range(len(N_LIST)):
        for j in range(K_MAX):
            ax.text(j, i, f"{rate[i, j]:.2f}", ha="center", va="center",
                    color="w" if rate[i, j] < 0.5 else "k", fontsize=8)
    ax.set_xlabel("GMM order K")
    fig.colorbar(im, ax=ax, label="selection rate")
    plt.tight_layout()
    plt.show()
 
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, N in zip(axes, N_LIST):
        S = all_scores[N].copy()
        S[~np.isfinite(S)] = np.nan
        med = np.nanmedian(S, axis=0)
        q1, q3 = np.nanpercentile(S, [25, 75], axis=0)
        ax.plot(Ks, med, "o-")
        ax.fill_between(Ks, q1, q3, alpha=0.3)
        ax.axvline(4, color="k", ls=":", lw=1)
        ax.set_title(f"N={N}")
        ax.set_xlabel("K")
        ax.set_xticks(Ks)
        if N == 10:
            ax.set_ylim(np.nanmin(q1[:3]) - 5, np.nanmax(q3) + 1)
    axes[0].set_ylabel("CV log-lik per sample (median, IQR)")
    plt.tight_layout()
    plt.show()
 
    X, z = sample_gmm(1000)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(X[:, 0], X[:, 1], c=z, s=6, cmap="tab10")
    t = np.linspace(0, 2 * np.pi, 200)
    for k in range(4):
        vals, vecs = np.linalg.eigh(COVS[k])
        ell = vecs @ (2 * np.sqrt(vals)[:, None] * np.vstack([np.cos(t), np.sin(t)]))
        ax.plot(MEANS[k, 0] + ell[0], MEANS[k, 1] + ell[1], "k")
    ax.set_title("True GMM (2-sigma contours), N=1000 sample")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()
    print("Saved q2_selection_rates.png, q2_selection_heatmap.png, "
          "q2_cv_loglik.png, q2_true_gmm.png")