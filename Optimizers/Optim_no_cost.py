import torch

class BroydenRootFinder:
    """
    Jacobian-free inverse-Broyden method to solve f(w)=y by updating model weights w.
    Works with arbitrary batch outputs (M) and parameter size (N).
    H \in R^{N x M} maps residuals to parameter steps: s = -H r
    """
    def __init__(
        self,
        params,
        alpha_init=1.0,         # initial step scaling on s (line-search starts here)
        gamma_init=1e-2,        # scale for H0 = gamma * [random-ish or structured]; here gamma*U with U ~ 0 (we use gamma*J0^- approx via identity on columns)
        ls_shrink=0.5,          # backtracking shrink
        ls_max_iter=10,         # max backtracking iters
        restart_tol=1e-12,      # restart H if denom too small
        eps=1e-12,              # small epsilon to stabilize denominators
        max_hist=None           # placeholder for limited-memory (not used in full H version)
    ):
        self.params = list(params)
        self.alpha_init = alpha_init
        self.gamma_init = gamma_init
        self.ls_shrink = ls_shrink
        self.ls_max_iter = ls_max_iter
        self.restart_tol = restart_tol
        self.eps = eps

        # Flatten helpers
        self._shapes = [p.shape for p in self.params]
        self._sizes = [p.numel() for p in self.params]
        self.N = sum(self._sizes)

        self.device = self.params[0].device
        self.dtype = self.params[0].dtype

        # Lazy state (initialized on first call when we know M)
        self.H = None
        self.prev_residual = None

    def _flatten_params(self):
        return torch.cat([p.detach().reshape(-1) for p in self.params])

    def _assign_params(self, flat_w):
        with torch.no_grad():
            offset = 0
            for p, sz, shape in zip(self.params, self._sizes, self._shapes):
                p.copy_(flat_w[offset:offset+sz].view(shape))
                offset += sz

    def _flatten_residual(self, residual):
        # residual is f(x)-y of shape (batch, out_dim?) -> flatten to (M,)
        return residual.reshape(-1)

    @torch.no_grad()
    def step(self, model, x, y, max_iters=1, tol=1e-6, verbose=False):
        """
        Perform up to `max_iters` Broyden iterations in-place on model parameters.
        Stops early if ||r|| decreases below tol.
        """
        for it in range(max_iters):
            # Current residual r_k
            preds = model(x)
            residual = (preds - y)
            r_k = self._flatten_residual(residual).to(device=self.device, dtype=self.dtype)
            M = r_k.numel()

            # Initialize H on first use
            if self.H is None or self.H.shape != (self.N, M):
                # H0 = gamma * [I padded/truncated to N x M]; simple, well-scaled default
                self.H = torch.zeros(self.N, M, device=self.device, dtype=self.dtype)
                # Put gamma on the top-left min(N,M) diagonal to start with a small mapping
                d = min(self.N, M)
                self.H[:d, :d] = torch.eye(d, device=self.device, dtype=self.dtype) * self.gamma_init

            # Compute step s_k = -H_k r_k, with optional line search
            s_k_dir = - self.H @ r_k  # direction in parameter space (N,)

            # Flat w_k
            w_k = self._flatten_params()

            # Simple backtracking on ||r||^2
            alpha = self.alpha_init
            r_norm2 = (r_k @ r_k).item()
            accept = False
            for ls in range(self.ls_max_iter+1):
                w_trial = w_k + alpha * s_k_dir
                self._assign_params(w_trial)
                r_trial = self._flatten_residual(model(x) - y)
                r_trial = r_trial.to(device=self.device, dtype=self.dtype)
                if (r_trial @ r_trial).item() <= r_norm2:
                    accept = True
                    break
                alpha *= self.ls_shrink

            if not accept:
                # If no decrease, just keep the smallest tried step
                if verbose:
                    print("[Broyden] Line search failed to decrease residual; taking smallest step.")
                self._assign_params(w_trial)  # already smallest alpha

            # Now we have x_{k+1} and r_{k+1}
            r_k1 = r_trial
            y_k = r_k1 - r_k  # change in residual (M,)

            # Update H: rectangular inverse-Broyden (least-squares variant)
            denom = (y_k @ y_k).item() + self.eps
            if denom < self.restart_tol:
                # Restart H if the secant information is degenerate
                if verbose:
                    print("[Broyden] Restarting H due to tiny denom.")
                self.H.zero_()
                d = min(self.N, M)
                self.H[:d, :d] = torch.eye(d, device=self.device, dtype=self.dtype) * self.gamma_init
            else:
                # s_k actually taken = x_{k+1}-x_k = alpha*s_k_dir
                s_taken = (w_trial - w_k)  # (N,)
                Hy = self.H @ y_k          # (N,)
                correction = (s_taken - Hy).unsqueeze(1) @ y_k.unsqueeze(0) / denom  # (N,1)(1,M)->(N,M)
                self.H = self.H + correction

            # Check convergence
            if torch.linalg.norm(r_k1).item() <= tol:
                if verbose:
                    print(f"[Broyden] Converged: ||r|| = {torch.linalg.norm(r_k1).item():.3e}")
                return True

        return False
