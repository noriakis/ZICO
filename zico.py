import math
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import trange

class ZICO(nn.Module):
    def __init__(self, d, init_scale=0.1, device="cpu"):
        
        super().__init__()
        self.d = d
        rnd = lambda: torch.randn(d, d, device=device) * init_scale
        self.W0 = nn.Parameter(rnd())
        self.W1 = nn.Parameter(rnd())
        with torch.no_grad():
            self.W0.fill_diagonal_(0); self.W1.fill_diagonal_(0)
        self.gamma = nn.Parameter(torch.zeros(d, device=device))
        self.delta = nn.Parameter(torch.zeros(d, device=device))
        self.theta = nn.Parameter(torch.ones (d, device=device))  # learnable

    def _acyclicity_logdet_from_W(self, W, s=1.0, eps=1e-8):
        d = self.d
        W = W.clone()
        W.fill_diagonal_(0.)
        S = W * W
        A = s * torch.eye(d, device=W.device) - S
        I = torch.eye(d, device=W.device)
        sign, logabsdet = torch.linalg.slogdet(A)
        h = -logabsdet + d * math.log(s + eps)
        return h

    def group_lasso(self, eps=1e-8):
        g = torch.sqrt(self.W0**2 + self.W1**2 + eps)
        return (g - torch.diag(torch.diag(g))).sum()
    
    def zinb_loglik_minibatch(self, X, idx,
        clamp_eta0=15.0, clamp_eta1_min=-10.0,
        clamp_eta1_max=8.0, r_min=1e-4, r_max=1e3,
        eps=1e-12):
        """
        Batch version
        """
        
        Xb = X.index_select(0, idx)
        eta0 = self.gamma.view(1, -1) + Xb @ self.W0
        eta1 = self.delta.view(1, -1) + Xb @ self.W1

    
        eta0 = eta0.clamp(-clamp_eta0, clamp_eta0)
        eta1 = eta1.clamp(clamp_eta1_min, clamp_eta1_max)
    
        log_pi = -F.softplus(-eta0)
        log_1mpi = -F.softplus( eta0)
    
        mu = torch.exp(eta1)
        r = F.softplus(self.theta).clamp(r_min, r_max).view(1, -1) # not theta[j]
        ###
        # Calculate in log scale
        ###
        log_r = torch.log(r)
        log_rpM = torch.log(r + mu + eps)
        log_p = log_r - log_rpM
        log_1mp = torch.log(mu + eps) - log_rpM
    
        k = Xb

        a = log_1mpi
        b = log_pi + r * log_p
        ll0 = torch.logsumexp(torch.stack([a, b], dim=-1), dim=-1)
    
        logpmf = (torch.lgamma(k + r) - torch.lgamma(r) - torch.lgamma(k + 1) + r * log_p + k * log_1mp)
        ll1 = log_pi + logpmf
        ll = torch.where(k == 0, ll0, ll1)
        """
        Put -1 before use
        """
        return ll.sum(dim=1).mean()
    
    def fit_logdet_batch(self, X, max_iter=3000, lr=3e-3, beta0=1e-3,
        beta_growth=1.5, beta_growth_per=100,
        s=1, batch_size=1024, verbose=True,
        warm=500, lam=1e-3):
        """
        Used for GPU training
        """
        X = X.to(self.W0.device)
        opt = optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.99))

        beta = beta0
        bar = trange(max_iter, disable=not verbose)
        for t in bar:
            lam_t = lam * 0.5 * (1 - math.cos(min(1., t / warm) * math.pi))
            for idx in batch_indices(X.size(0), batch_size, shuffle=False, device=X.device):
                opt.zero_grad()
                nll = -self.zinb_loglik_minibatch(X, idx)
                grp = self.group_lasso()
                h = self._acyclicity_logdet_from_W(self.W0, s=s) + self._acyclicity_logdet_from_W(self.W1, s=s)

                lambda_align = 1
                align = lambda_align * ((self.W0 - self.W1).pow(2).sum())
                # loss = nll + lam_t * self.penalty_W0() + lam_t * self.penalty_W1() + beta * h + align
                loss = nll + lam_t * grp + beta * h + align

                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()
    
                with torch.no_grad():
                    self.W0.fill_diagonal_(0); self.W1.fill_diagonal_(0)

            if (t + 1) % beta_growth_per == 0:
                beta *= beta_growth

            if verbose:
                bar.set_postfix(nll=float(nll.item()),
                                grp=float(grp.item()),
                                h=float(h.item()),
                                beta=float(beta))

        return self
    
    def nb_loglik_minibatch(self, X, idx, clamp_eta1_min=-10.0, clamp_eta1_max=8.0, r_min=1e-4, r_max=1e3, eps=1e-12):
        """
        This does not use W0
        """
        Xb = X.index_select(0, idx)

        eta1 = self.delta.view(1, -1) + Xb @ self.W1
        eta1 = eta1.clamp(clamp_eta1_min, clamp_eta1_max)
        mu = torch.exp(eta1)
        r = F.softplus(self.theta).clamp(r_min, r_max).view(1, -1)
        
        log_r = torch.log(r)
        log_rpM = torch.log(r + mu + eps)
        log_p = log_r - log_rpM
        log_1mp = torch.log(mu + eps) - log_rpM

        k = Xb
        logpmf = (torch.lgamma(k + r) - torch.lgamma(r) - torch.lgamma(k + 1)
                  + r * log_p + k * log_1mp)
        return logpmf.sum(dim=1).mean()

    def neg_loglik_minibatch_nb(self, X, idx, **kw):
        return -self.nb_loglik_minibatch(X, idx, **kw)
    
    def fit_logdet_batch_nb(self, X, max_iter=1000, lr=1e-3,
        batch_size=1024, s=1.0, lam=1e-3,
        beta0=1e-3, beta_growth=1.5, beta_growth_per_epoch=100,
        verbose=True):
        
        X = X.to(self.W0.device)
        n, d = X.shape
        opt = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9,0.99))
        beta = beta0
        bar = trange(max_iter, disable=not verbose)
        for ep in bar:
            batches = [torch.arange(i, min(i+batch_size, n), device=X.device)
                       for i in range(0, n, batch_size)]
            running = 0.0
            for idx in batches:
                opt.zero_grad()
                nll = self.neg_loglik_minibatch_nb(X, idx)
                h = self._acyclicity_logdet_from_W(self.W1.abs(), s=s)
                l1 = self.W1.abs().sum()

                loss = nll + (beta * h + lam * l1)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()

                with torch.no_grad():
                    self.W0.fill_diagonal_(0.0)
                    self.W1.fill_diagonal_(0.0)

                running += float(loss.item())

            if (ep + 1) % beta_growth_per_epoch == 0:
                beta *= beta_growth

            if verbose:
                bar.set_postfix(nll=float(nll.item()),
                                h=float(h.item()),
                                beta=float(beta))
        return self    

def batch_indices(n, batch_size, shuffle=True, device=None):
    idx = torch.arange(n, device=device)
    if shuffle:
        perm = torch.randperm(n, device=device)
        idx = idx[perm]
    for i in range(0, n, batch_size):
        yield idx[i:i+batch_size]
    
def accuracy(true_B, est_B, thresh=0.3, eps=1e-8):
    est_bin = (est_B.abs() > thresh).float()
    true_bin = (true_B != 0).float()
    tp = (est_bin * true_bin).sum()
    fp = (est_bin * (1 - true_bin)).sum()
    fn = ((1 - est_bin) * true_bin).sum()
    fdr = fp / (fp + tp + eps)
    tpr = tp / (tp + fn + eps)
    return {'TPR': tpr.item(), 'FDR': fdr.item()}
