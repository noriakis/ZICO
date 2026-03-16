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
        self.theta = nn.Parameter(torch.ones (d, device=device))

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


    def zip_loglik_minibatch(self, X, idx, clamp_eta0=15.0,
        clamp_eta1_min=-10.0, clamp_eta1_max=8.0, eps=1e-12):

        Xb = X.index_select(0, idx)
    
        eta0 = self.gamma.view(1, -1) + Xb @ self.W0
        eta1 = self.delta.view(1, -1) + Xb @ self.W1
    
        eta0 = eta0.clamp(-clamp_eta0, clamp_eta0)
        eta1 = eta1.clamp(clamp_eta1_min, clamp_eta1_max)

        log_pi0 = -F.softplus(-eta0)   # log pi
        log_1mpi0 = -F.softplus(eta0)   # log(1-pi)
        mu = torch.exp(eta1)
        k = Xb.to(dtype=eta1.dtype)

        ll0 = torch.logsumexp(torch.stack([log_pi0, log_1mpi0 - mu], dim=-1), dim=-1)

        logpmf_pos = k * eta1 - mu - torch.lgamma(k + 1.0)
        ll1 = log_1mpi0 + logpmf_pos
        ll = torch.where(k == 0, ll0, ll1)
        return ll.sum(dim=1).mean()
    

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
        
        ###
        # Calculate in log scale
        ###

        log_pi0  = -F.softplus(-eta0)   # log pi
        log_1mpi0 = -F.softplus(eta0)   # log(1-pi)

        mu = torch.exp(eta1)
        r = F.softplus(self.theta).clamp(r_min, r_max).view(1, -1) # not theta[j]

        log_r = torch.log(r)
        log_rpM = torch.log(r + mu + eps)
        log_p = log_r - log_rpM
        log_1mp = torch.log(mu + eps) - log_rpM
        k = Xb

        a = log_pi0
        b = log_1mpi0 + r * log_p
        ll0 = torch.logsumexp(torch.stack([a, b], dim=-1), dim=-1)

        logpmf = (torch.lgamma(k + r) - torch.lgamma(r) - torch.lgamma(k + 1)
                  + r * log_p + k * log_1mp)

        ll1 = log_1mpi0 + logpmf

        ll = torch.where(k == 0, ll0, ll1)
        """
        Put -1 before use
        """

        return ll.sum(dim=1).mean()


    def support_diff_penalty(self, tau=1):
        ind0 = torch.sigmoid(tau * torch.abs(self.W0))
        ind1 = torch.sigmoid(tau * torch.abs(self.W1))
        diff = torch.abs(ind0 - ind1)
        d = self.W0.shape[0]
        mask = ~torch.eye(d, dtype=bool, device=self.W0.device)
        return diff[mask].sum()


    def fit_logdet_batch(self, X, max_iter=3000, lr=3e-3,
        s=1, batch_size=1024, verbose=True, shuffle=True, loss_type="NB",
        warm=500, lam=1e-3, mu0=1, mu_decay=0.1, mu_decay_per_epoch=1000,
        lambda_align=0.1, norm_type="frobenius", logdet_both=False,
        ignore_logdet=False, logdet_only_W1=False):
        """
        Used for GPU training.
        """
        X = X.to(self.W0.device)
        opt = optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.99))

        mu = mu0
        bar = trange(max_iter, disable=not verbose)
        for t in bar:
            lam_t = lam * 0.5 * (1 - math.cos(min(1., t / warm) * math.pi))
            for idx in batch_indices(X.size(0), batch_size, shuffle=shuffle, device=X.device):
                opt.zero_grad()
                # nll = self.neg_loglik_minibatch(X, idx)
                if loss_type == "NB":
                    nll = -self.zinb_loglik_minibatch(X, idx)
                else:
                    nll = -self.zip_loglik_minibatch(X, idx)
                
                grp = self.group_lasso()
                
                if logdet_both:
                    h = self._acyclicity_logdet_from_W(torch.sqrt(self.W0**2+self.W1**2+1e-8), s=s)
                else:
                    if logdet_only_W1:
                        h = self._acyclicity_logdet_from_W(self.W1, s=s)
                    else:
                        h = self._acyclicity_logdet_from_W(self.W0, s=s) + self._acyclicity_logdet_from_W(self.W1, s=s)

                # h = self._acyclicity_logdet_from_W(self.W0.abs() + self.W1.abs(), s=s)


                if norm_type == "frobenius":
                    norm = (self.W0 + self.W1).pow(2).sum() # Frobenius
                elif norm_type == "l1":
                    norm = torch.sum(torch.abs(self.W0 + self.W1)) # L1
                elif norm_type == "mag":
                    norm = torch.norm(self.W0.abs() - self.W1.abs(), p='fro')**2
                elif norm_type == "support":
                    norm = self.support_diff_penalty()
                else:
                    raise ValueError("Unsupported norm")

                align = lambda_align * norm

                # loss = nll + lam_t * self.penalty_W0() + lam_t * self.penalty_W1() + beta * h + align
                # loss = nll + lam_t * grp + beta * h + align
                if ignore_logdet:
                    loss = nll + lam_t * grp + align
                else:
#                    loss = mu * (nll + lam_t * grp + align) + h
                    loss = mu * (nll + lam_t * grp) + h + align # Original

                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()
    
                with torch.no_grad():
                    self.W0.fill_diagonal_(0); self.W1.fill_diagonal_(0)

            if (t + 1) % mu_decay_per_epoch == 0:
                mu *= mu_decay

            if verbose:
                bar.set_postfix(nll=float(nll.item()),
                                grp=float(grp.item()),
                                h=float(h.item()),
                               mu=float(mu))

        return self
    
    def nb_loglik_minibatch(self, X, idx,
        clamp_eta1_min=-10.0, clamp_eta1_max=8.0,
        r_min=1e-4, r_max=1e3, eps=1e-12):
        """
        This does not use W0
        """
        Xb = X.index_select(0, idx)

        eta1 = self.delta.view(1, -1) + Xb @ self.W1
        eta1 = eta1.clamp(clamp_eta1_min, clamp_eta1_max)
        mu = torch.exp(eta1)
        r = F.softplus(self.theta).clamp(r_min, r_max).view(1, -1)
        # r = F.softplus(self.theta).view(1, -1)
        
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
    

    def poisson_loglik_minibatch_stable(self, X, idx, clamp_eta1_min=-10.0,
                                        clamp_eta1_max=8.0, offset_log=None,
                                        eps=1e-12):
        Xb = X.index_select(0, idx)
        eta1 = self.delta.view(1, -1) + Xb @ self.W1
    
        if offset_log is not None:
            ol = offset_log
            if ol.dim() == 1:
                if ol.shape[0] == X.shape[0]:
                    ol = ol.index_select(0, idx)
                ol = ol.view(-1, 1)
            elif ol.dim() == 2 and ol.shape[0] == X.shape[0]:
                ol = ol.index_select(0, idx)
            eta1 = eta1 + ol
    
        # numerical stability on the log-mean
        eta1 = eta1.clamp(clamp_eta1_min, clamp_eta1_max)
    
        k = Xb.to(dtype=eta1.dtype)
        mu = torch.exp(eta1)
        logpmf = k * eta1 - mu - torch.lgamma(k + 1.0)
    
        return logpmf.sum(dim=1).mean()
    
    def neg_loglik_minibatch_poisson(self, X, idx, **kw):
        return -self.poisson_loglik_minibatch_stable(X, idx, **kw)

    def fit_logdet_batch_nb(self, X, max_iter=1000, lr=1e-3,
        batch_size=1024, s=1.0, lam=1e-3, shuffle=True,
        warm=500, mu0=1, mu_decay=0.1, mu_decay_per_epoch=1000,
        verbose=True, loss_type="NB", ignore_logdet=False):
        
        X = X.to(self.W0.device)
        n, d = X.shape
        opt = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9,0.99))
        mu = mu0
        bar = trange(max_iter, disable=not verbose)
        for ep in bar:
            lam_t = lam * 0.5 * (1 - math.cos(min(1., ep / warm) * math.pi))
            for idx in batch_indices(X.size(0), batch_size,
                shuffle=shuffle, device=X.device):
                opt.zero_grad()
                if loss_type == "NB":
                    nll = self.neg_loglik_minibatch_nb(X, idx)
                else:
                    nll = self.neg_loglik_minibatch_poisson(X, idx)
                    
                h = self._acyclicity_logdet_from_W(self.W1.abs(), s=s)
                l1 = self.W1.abs().sum()
                
                if ignore_logdet:
                    loss = nll + lam_t * l1
                else:
                    loss = mu *(nll + lam_t * l1) + h

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()

                with torch.no_grad():
                    self.W1.fill_diagonal_(0.0)


            if (ep + 1) % mu_decay_per_epoch == 0:
                mu *= mu_decay

            if verbose:
                bar.set_postfix(nll=float(nll.item()),
                                h=float(h.item()),
                                mu=float(mu))
        return self    

def batch_indices(n, batch_size, shuffle=True, device=None):
    idx = torch.arange(n, device=device)
    if shuffle:
        perm = torch.randperm(n, device=device)
        idx = idx[perm]
    for i in range(0, n, batch_size):
        yield idx[i:i+batch_size]
    
