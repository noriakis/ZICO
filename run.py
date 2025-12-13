import math
import numpy as np
import torch
from .zico import ZICO
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import random
import igraph as ig
from sklearn.metrics import precision_recall_curve, average_precision_score

def accuracy(true_B, est_B, thresh=0.3, eps=1e-8):
    est_bin = (est_B.abs() > thresh).float()
    true_bin = (true_B != 0).float()
    tp = (est_bin * true_bin).sum()
    fp = (est_bin * (1 - true_bin)).sum()
    fn = ((1 - est_bin) * true_bin).sum()
    fdr = fp / (fp + tp + eps)
    tpr = tp / (tp + fn + eps)

    ## AUPRC computing
    true_B = true_B.detach().cpu().numpy()
    est_B = est_B.detach().cpu().numpy()
    mask = ~np.eye(true_B.shape[0], dtype=bool)
    y_true = true_B[mask].ravel()
    y_score = np.abs(est_B[mask].ravel())
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    return {'TPR': tpr.item(), 'FDR': fdr.item(), 'AUPRC': auprc}


def random_dag(d, edge_prob=0.25, m=3, seed=1, type="er"):
    random.seed(seed)
    if type == "er":
        print("Generating based on ig.Graph.Erdos_Renyi")
        g = ig.Graph.Erdos_Renyi(n=d, p=edge_prob, directed=False, loops=False)
    elif type == "ba":
        print("Generating based on ig.Graph.Barabasi")
        g = ig.Graph.Barabasi(n=d, m=m)
    else:
        raise ValueError("please specify graph type by er or ba")
    g.to_directed(mode="acyclic")
    return np.array(g.get_adjacency()), g

def nb_rng(mu, theta, rng):
    p = theta / (theta + mu)
    r = theta
    return rng.negative_binomial(r, p)
    
def topo_order_igraph(B):
    d = B.shape[0]
    src, dst = np.where(B != 0)
    edges = list(zip(src.tolist(), dst.tolist()))
    g = ig.Graph(n=d, edges=edges, directed=True)
    if not g.is_dag():
        raise ValueError("B is not a DAG.")
    order = g.topological_sorting(mode="OUT")
    return np.array(order, dtype=int)

def generate_zinb(B, n=2000, seed=0,
    gamma_mean=0, clip_log_mu=3, runif=True,
    W0_low=0.5, W0_high=2, W1_low=-2, W1_high=-0.5,
    theta=5):

    rng = np.random.default_rng(seed)
    d = B.shape[0]
    
    if runif:
        ## Uniform distribution (by the spcified parameters)
        W0 = rng.uniform(W0_low, W0_high, size=B.shape) * B
        W1 = rng.uniform(W1_low, W1_high, size=B.shape) * B
    else:
        ## Normal distribution
        W0 = rng.normal(0.6, 0.2, size=B.shape) * B
        W1 =  rng.normal(0.8, 0.2, size=B.shape) * B

    gamma = rng.normal(gamma_mean, 0.2, size=d) # logit(pi)
    delta = rng.normal(1.5, 0.2, size=d) # log(mu)
    theta = np.full(d, theta)

    X = np.zeros((n, d), dtype=int)

    order = topo_order_igraph(B)
    for j in order:
        pa = np.where(B[:, j] != 0)[0]
        if pa.size:
            logit_pi = gamma[j] + X[:, pa] @ W0[pa, j]
            log_mu = delta[j] + X[:, pa] @ W1[pa, j]
        else:
            logit_pi = np.full(n, gamma[j])
            log_mu = np.full(n, delta[j])

        pi = 1.0 / (1.0 + np.exp(-logit_pi))
        mu = np.exp(np.minimum(log_mu, clip_log_mu))

        u = rng.random(n)
        zeros_mask = (u < pi)
        counts = np.where(zeros_mask, 0, [nb_rng(mu[k], theta[j], rng) for k in range(n)])
        X[:, j] = counts
        
    return torch.tensor(X, dtype=torch.float32), torch.tensor(B, dtype=torch.float32)

def graph_from_adj(adj):
    A = np.asarray(adj)
    n = A.shape[0]
    vertex_names = [str(i) for i in range(n)]
    A01 = (A != 0).astype(int)
    g = ig.Graph.Adjacency(A01.tolist(), mode=ig.ADJ_DIRECTED)
    g.vs["name"] = vertex_names
    return g

def plot_four_networks(adjs, layout_method="fr", titles=None,
    node_size=15, edge_width_scale=1.0, output_path="networks.png",
    dpi=300):

    n0 = np.asarray(adjs[0]).shape[0]
    graphs = []
    for i, A in enumerate(adjs):
        A = np.asarray(A)
        g = graph_from_adj(A)
        graphs.append(g)

    g0 = graphs[0]
    if layout_method.lower() in ["fr", "fruchterman-reingold", "fruchterman_reingold"]:
        layout0 = g0.layout_fruchterman_reingold()
    elif layout_method.lower() in ["kk", "kamada-kawai", "kamada_kawai"]:
        layout0 = g0.layout_kamada_kawai()
    else:
        layout0 = g0.layout(layout_method)

    coords = [tuple(p) for p in layout0]
    for g in graphs:
        g.vs["x"] = [c[0] for c in coords]
        g.vs["y"] = [c[1] for c in coords]

    if titles is None:
        titles = [f"Graph {i+1}" for i in range(4)]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for ax, g, title in zip(axes, graphs, titles):
        ax.set_title(title, fontsize=12)
        ax.set_axis_off()
        ig.plot(
            g,
            target=ax,
            layout=list(zip(g.vs["x"], g.vs["y"])),
            vertex_size=node_size,
            vertex_color="lightgray",
            vertex_frame_width=0.5,
            edge_color="gray",
            bbox=(400, 400)
        )

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", default=10, type=int, help="the number of variables")
    ap.add_argument("--n", default=500, type=int, help="the number of observations")
    ap.add_argument("--seed", default=1, type=int)
    ap.add_argument("--threshold", default=0.3, type=float)
    ap.add_argument("--lam", default=1e-3, type=float)
    ap.add_argument("--epochs", default=3000, type=int)
    ap.add_argument("--W0_sign", default=1, type=int)
    ap.add_argument("--W1_sign", default=1, type=int)
    ap.add_argument("--device", default="cpu", type=str)
    ap.add_argument("--loss", default="zinb", type=str)
    ap.add_argument("--graph_type", default="er", type=str)
    ap.add_argument("--lambda_align", default=1, type=float)
    ap.add_argument("--file", default=None, type=str)
    ap.add_argument("--reference", default=None, type=str)
    ap.add_argument("--transpose_reference", action="store_true")
    ap.add_argument("--save_graph", action="store_true")

    args = vars(ap.parse_args())

    if args["file"] is None:
        B_true, ig_true = random_dag(args["d"], seed=args["seed"], type=args["graph_type"])
        X, _ = generate_zinb(B_true, args["n"], seed=args["seed"], W0_sign=args["W0_sign"],
            W1_sign=args["W1_sign"])
    else:
        import pandas as pd
        X = torch.from_numpy(np.asarray(pd.read_csv(args["file"], sep="\t"))).float()
        if args["reference"] is not None:
            B_true = np.asarray(pd.read_csv(args["reference"], sep="\t"))
            if args["transpose_reference"]:
                B_true = B_true.T
        else:
            ##
            # Zero array if no reference is provided
            ##
            B_true = np.zeros((X.shape[1], X.shape[1]))

    zero_rates = (X.numpy() == 0).mean(axis=0)
    print("per-node zero rate (empirical):", zero_rates.round(3))
    loss = args["loss"]
    lam = args["lam"]
    lambda_align = args["lambda_align"]
    th = float(args["threshold"])
    model = ZICO(X.shape[1], device=args["device"]).to(args["device"])
    if loss == "nb":
        model.fit_logdet_batch_nb(X, max_iter=args["epochs"], lam=lam)
    elif loss == "poisson":
        model.fit_logdet_batch_nb(X, max_iter=args["epochs"], loss="Poisson", lam=lam)
    elif loss == "zinb":
        model.fit_logdet_batch(X, max_iter=args["epochs"], loss="NB", lam=lam,
            lambda_align=lambda_align)
    elif loss == "zip":
        model.fit_logdet_batch(X, max_iter=args["epochs"], loss="Poisson", lam=lam,
            lambda_align=lambda_align)
    else:
        raise ValueError("Unknown loss specified")

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    sns.heatmap(B_true, ax=axes[0], cmap='viridis', square=True)
    axes[0].set_title('Reference')
    sns.heatmap(model.W0.detach().cpu().numpy(), ax=axes[1], cmap='viridis', square=True)
    axes[1].set_title('W0')
    sns.heatmap(model.W1.detach().cpu().numpy(), ax=axes[2], cmap='viridis', square=True)
    axes[2].set_title('W1')
    sns.heatmap(model.W1.detach().cpu().abs().numpy()+model.W0.detach().cpu().abs().numpy(),
        ax=axes[3], cmap='viridis', square=True)
    axes[3].set_title('W0+W1')
    
    print("W0: %s" % accuracy(torch.from_numpy(B_true).to(args["device"]), model.W0, thresh=th, eps=1e-8))
    print("W1: %s" % accuracy(torch.from_numpy(B_true).to(args["device"]), model.W1, thresh=th, eps=1e-8))
    print("W0+W1: %s" % accuracy(torch.from_numpy(B_true).to(args["device"]),
        model.W1.abs()+model.W0.abs(), thresh=th, eps=1e-8))

    plt.savefig("results_"+str(loss)+".png")
    np.save("reference.npy", B_true)
    np.save("W1_"+str(loss)+".npy", model.W1.detach().cpu().numpy())
    np.save("W0_"+str(loss)+".npy", model.W0.detach().cpu().numpy())
    np.save("X.npy", X.cpu().numpy())

    if args["save_graph"]:
        plot_four_networks(
            [B_true,
            model.W0.detach().cpu().numpy()>th,
            model.W1.detach().cpu().numpy()>th,
            model.W0.detach().cpu().numpy()+model.W1.detach().cpu().numpy()>th],
            layout_method="fr",
            titles=["Reference", "W0", "W1", "W0+W1"],
            node_size=16,
            edge_width_scale=1.5,
            output_path="networks.png",
            dpi=300
        )