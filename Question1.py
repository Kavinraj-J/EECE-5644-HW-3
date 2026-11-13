import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from scipy.optimize import minimize, check_grad
from scipy.stats import multivariate_normal as mvn
from scipy.special import logsumexp, expit

rng = np.random.default_rng(5644)

#**************** Question 1 Data Generation ********

C, D = 4, 3
PRIORS = np.full(C, 1 / C)
MEANS = np.array([[ 1.0,  1.0,  1.0], # m0
                  [ 1.0, -1.0, -1.0], # m1
                  [-1.0,  1.0, -1.0],  # m2
                  [-1.0, -1.0,  1.0]]) # m3
COVS = np.array([
    np.array([[1.0, 0.3, 0.2], [0.3, 0.8, -0.1], [0.2, -0.1, 0.6]]), # C0
    (1 / 2) * np.array([[2.0, -0.5, 0.3], [-0.5, 1.0, 0.0], [0.3, 0.0, 1.5]]), # C1
    np.array([[0.7, 0.0, 0.2], [0.0, 1.2, -0.3], [0.2, -0.3, 0.9]]), # C2
    (1 / 4) * np.array([[3.0, 1.0, 0.0], [1.0, 4.0, 0.5], [0.0, 0.5, 5.0]]), # C3
])
 
 
def generate(N):
    y = rng.choice(C, size=N, p=PRIORS)
    X = np.empty((N, D))
    for c in range(C):
        idx = y == c
        X[idx] = rng.multivariate_normal(MEANS[c], COVS[c], idx.sum())
    return X, y

#**************** Question 1 MLP ********
def unpack(theta, P):
    i = 0
    W1 = theta[i:i + P * D].reshape(P, D); i += P * D
    b1 = theta[i:i + P];                   i += P
    W2 = theta[i:i + C * P].reshape(C, P); i += C * P
    b2 = theta[i:i + C]
    return W1, b1, W2, b2
 
 
def n_params(P):
    return P * D + P + C * P + C
 
 
def forward(theta, X, P):
    W1, b1, W2, b2 = unpack(theta, P)
    A = X @ W1.T + b1               
    H = np.logaddexp(0.0, A)       
    Z = H @ W2.T + b2              
    logp = Z - logsumexp(Z, axis=1, keepdims=True)
    return A, H, logp
 
 
def loss_grad(theta, X, Y1h, P):
    """Average cross-entropy = -(1/N) * log-likelihood, and its gradient."""
    N = X.shape[0]
    W1, b1, W2, b2 = unpack(theta, P)
    A, H, logp = forward(theta, X, P)
    loss = -np.sum(Y1h * logp) / N
    G = (np.exp(logp) - Y1h) / N          
    gW2 = G.T @ H
    gb2 = G.sum(0)
    GA = (G @ W2) * expit(A)             
    gW1 = GA.T @ X
    gb1 = GA.sum(0)
    return loss, np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])
 
 
def train_mlp(X, y, P, n_restarts=5, maxiter=2000):
    """ML training with random restarts; keep the highest training log-likelihood."""
    Y1h = np.eye(C)[y]
    best_theta, best_loss = None, np.inf
    for _ in range(n_restarts):
        W1 = rng.normal(0, np.sqrt(2 / (D + P)), (P, D))
        W2 = rng.normal(0, np.sqrt(2 / (P + C)), (C, P))
        theta0 = np.concatenate([W1.ravel(), np.zeros(P), W2.ravel(), np.zeros(C)])
        res = minimize(loss_grad, theta0, args=(X, Y1h, P), jac=True,
                       method="L-BFGS-B", options={"maxiter": maxiter})
        if res.fun < best_loss:
            best_loss, best_theta = res.fun, res.x
    return best_theta, best_loss
 
 
def predict(theta, X, P):
    return forward(theta, X, P)[2].argmax(1)   # MAP decision on model posteriors
 
 
#**************** Question 1 10-Fold Cross Validation ********
P_CANDIDATES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
K_FOLDS = 10
 
 
def cross_validate(X, y, cv_restarts=2, cv_maxiter=500):
    N = len(y)
    perm = rng.permutation(N)
    folds = np.array_split(perm, K_FOLDS)
    cv_err = np.zeros(len(P_CANDIDATES))
    for j, P in enumerate(P_CANDIDATES):
        errs = []
        for k in range(K_FOLDS):
            val = folds[k]
            tr = np.concatenate([folds[m] for m in range(K_FOLDS) if m != k])
            th, _ = train_mlp(X[tr], y[tr], P, n_restarts=cv_restarts,
                              maxiter=cv_maxiter)
            errs.append(np.mean(predict(th, X[val], P) != y[val]) * len(val))
        cv_err[j] = np.sum(errs) / N           # pooled validation P(error)
    best = P_CANDIDATES[int(np.argmin(cv_err))]  # argmin picks smallest P on ties
    return best, cv_err
 
 

if __name__ == "__main__":
 
    N_TRAIN = [100, 500, 1000, 5000, 10000]
    train_sets = {N: generate(N) for N in N_TRAIN}
    Xtest, ytest = generate(100000)
 
    logpost = np.stack([np.log(PRIORS[c]) + mvn(MEANS[c], COVS[c]).logpdf(Xtest)
                        for c in range(C)], axis=1)
    p_err_opt = np.mean(logpost.argmax(1) != ytest)
    print(f"\nTheoretically optimal classifier: test P(error) = {p_err_opt:.4f}")
 
    results = {}
    cv_curves = {}
    for N in N_TRAIN:
        X, y = train_sets[N]
        bestP, cv_err = cross_validate(X, y)
        cv_curves[N] = cv_err
        theta, loss = train_mlp(X, y, bestP, n_restarts=10)
        p_err = np.mean(predict(theta, Xtest, bestP) != ytest)
        results[N] = (bestP, p_err, loss)
        print(f"N={N:6d}: CV P(error) per P = {np.round(cv_err, 3)}")
        print(f"          chosen P = {bestP:2d}, final train CE = {loss:.4f}, "
              f"test P(error) = {p_err:.4f}", flush=True)
 
    print("\n  N_train | best P | test P(error) | gap to optimal")
    for N in N_TRAIN:
        P, e, _ = results[N]
        print(f"  {N:7d} | {P:6d} | {e:13.4f} | {e - p_err_opt:+.4f}")
 
    plt.figure(figsize=(7, 4.5))
    errs = [results[N][1] for N in N_TRAIN]
    plt.semilogx(N_TRAIN, errs, "o-", label="Trained MLP (MAP on MLP posteriors)")
    for N, e in zip(N_TRAIN, errs):
        plt.annotate(f"P={results[N][0]}", (N, e), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9)
    plt.axhline(p_err_opt, color="r", ls="--",
                label=f"Theoretically optimal = {p_err_opt:.4f}")
    plt.xlabel("Number of training samples")
    plt.ylabel("Empirical test P(error)")
    plt.title("Q1: MLP test error vs training set size")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
 
    plt.figure(figsize=(7, 4.5))
    for N in N_TRAIN:
        plt.semilogx(P_CANDIDATES, cv_curves[N], "o-", label=f"N={N}")
    plt.xlabel("Number of hidden perceptrons P")
    plt.ylabel("10-fold CV P(error)")
    plt.title("Q1: cross-validation model order selection")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("\nSaved q1_perror_vs_N.png and q1_cv_curves.png")