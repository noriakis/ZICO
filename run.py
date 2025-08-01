import math
import numpy as np
import torch
from zico import ZICO
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def random_dag(d, edge_prob=0.15, seed=0):
    rng = np.random.default_rng(seed)
    B = np.triu((rng.random((d,d)) < edge_prob).astype(float), k=1)
    P = rng.permutation(d)
    return B[P][:, P]

def nb_rng(mu, theta, rng):
    p = theta / (theta + mu)
    r = theta
    return rng.negative_binomial(r, p)

def generate_zinb(B, n=2000, seed=0,
    gamma_mean=-1, clip_log_mu=20.0):

    rng = np.random.default_rng(seed)
    d = B.shape[0]
    W0 = rng.normal( 0.6, 0.2, size=B.shape) * B
    W1 = rng.normal( 0.8, 0.2, size=B.shape) * B
    gamma = rng.normal(gamma_mean, 0.2, size=d)
    delta = rng.normal( 1.5, 0.2, size=d)
    theta = np.full(d, 5.0)

    X = np.zeros((n, d), dtype=int)

    order = np.argsort(B.sum(axis=0))
    for i in range(n):
        for j in order:
            pa = np.where(B[:, j] != 0)[0]
            logit_pi = gamma[j] + (X[i, pa] @ W0[pa, j] if pa.size else 0.0)
            log_mu   = delta[j] + (X[i, pa] @ W1[pa, j] if pa.size else 0.0)
            pi = 1.0 / (1.0 + np.exp(-logit_pi))
            mu = math.exp(min(log_mu, clip_log_mu))
            if rng.random() < (1 - pi):
                X[i, j] = 0
            else:
                X[i, j] = nb_rng(mu, theta[j], rng)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(B, dtype=torch.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", default=10, type=int)
    ap.add_argument("--n", default=500, type=int)
    ap.add_argument("--seed", default=1, type=int)
    ap.add_argument("--device", default="cpu", type=str)
    args = vars(ap.parse_args())

    B_true = random_dag(args["d"], seed=args["seed"])
    X, _ = generate_zinb(B_true, args["n"], seed=args["seed"])
    
    model = ZICO(X.shape[1], device=args["device"]).to(args["device"])
    model.fit_logdet_batch(X)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.heatmap(B_true, ax=axes[0], cmap='viridis', square=True)
    axes[0].set_title('Reference')
    sns.heatmap(model.W0.detach().cpu().numpy(), ax=axes[1], cmap='viridis', square=True)
    axes[1].set_title('W0')
    sns.heatmap(model.W1.detach().cpu().numpy(), ax=axes[2], cmap='viridis', square=True)
    axes[2].set_title('W1')
    plt.savefig("results.png")

