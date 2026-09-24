"""CoFineUCB baseline for the TDGE MovieLens environment.

CoFineUCB (Yue et al., "Hierarchical Exploration for Accelerating
Contextual Bandits") is a coarse-to-fine linear UCB algorithm.  It learns a
low-dimensional feature subspace from historical user profiles, estimates a
coarse model in that subspace, and regularizes the full-space model toward the
coarse estimate.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from tqdm import tqdm


def _sherman_morrison_update(inv_mat: np.ndarray, x: np.ndarray,
                             sample_weight: float = 1.0) -> np.ndarray:
    x = x.reshape(-1, 1)
    w = float(sample_weight)
    inv_x = inv_mat @ x
    denom = 1.0 + w * float(x.T @ inv_x)
    return inv_mat - (w * (inv_x @ inv_x.T)) / denom


def _make_context(user_emb: np.ndarray, item_feats: np.ndarray) -> np.ndarray:
    if item_feats.ndim == 1:
        item_feats = item_feats.reshape(1, -1)
    n_items = item_feats.shape[0]
    raw = np.concatenate([np.tile(user_emb, (n_items, 1)), item_feats], axis=1)
    raw = np.concatenate([raw, np.ones((n_items, 1)) * 0.01], axis=1)
    return raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-9)


def learn_cofine_projection(user_features: np.ndarray,
                            item_features: np.ndarray,
                            rating_matrix: np.ndarray,
                            train_users: np.ndarray,
                            k: int = 5,
                            profile_lamb: float = 1.0,
                            max_profile_users: int | None = None,
                            min_interactions: int = 3,
                            use_learnu_scaling: bool = True,
                            random_state: int = 2026) -> np.ndarray:
    """Learn the CoFineUCB projection matrix U from train-user profiles.

    The original paper learns U from pre-existing user preference vectors.  In
    this offline recommendation setting, we estimate those vectors by ridge
    regression on the observed interactions of training users only.

    Returns
    -------
    U : np.ndarray, shape (context_dim, k)
        Projection matrix used by CoFineUCB, where projected features are
        ``x_tilde = U.T @ x``.
    """
    rng = np.random.default_rng(random_state)
    train_users = np.asarray(train_users, dtype=np.int64)
    if max_profile_users is not None and len(train_users) > max_profile_users:
        train_users = rng.choice(train_users, size=max_profile_users,
                                 replace=False)

    context_dim = user_features.shape[1] + item_features.shape[1] + 1
    profiles = []
    eye = np.eye(context_dim, dtype=np.float64)

    for uid in tqdm(train_users, desc="Learn CoFineUCB U", leave=False):
        ratings = rating_matrix[int(uid)]
        observed_items = np.flatnonzero(~np.isnan(ratings)).astype(np.int64)
        if len(observed_items) < min_interactions:
            continue

        x = _make_context(user_features[int(uid)], item_features[observed_items])
        y = np.nan_to_num(ratings[observed_items], nan=0.0).reshape(-1, 1)

        a = profile_lamb * eye + x.T @ x
        b = x.T @ y
        try:
            theta = np.linalg.solve(a, b).reshape(-1)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(a) @ b
            theta = theta.reshape(-1)
        profiles.append(theta)

    if len(profiles) == 0:
        print("[CoFineUCB] Warning: no train-user profiles were available; "
              "falling back to the first coordinate subspace.")
        u = np.zeros((context_dim, min(k, context_dim)), dtype=np.float64)
        u[np.arange(u.shape[1]), np.arange(u.shape[1])] = 1.0
        return u

    w = np.stack(profiles, axis=1)  # (D, N)
    u0, _, _ = np.linalg.svd(w, full_matrices=False)
    k_eff = min(int(k), u0.shape[1])
    u0 = u0[:, :k_eff]

    if not use_learnu_scaling:
        u = u0
    else:
        w0 = u0.T @ w
        cov = w0 @ w0.T
        eigvals, eigvecs = np.linalg.eigh((cov + cov.T) / 2.0)
        eigvals = np.clip(eigvals, 0.0, None)
        cov_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        trace = float(np.trace(cov_sqrt))
        if trace <= 1e-12:
            u = u0
        else:
            omega = (k_eff / trace) * cov_sqrt
            eigvals_o, eigvecs_o = np.linalg.eigh((omega + omega.T) / 2.0)
            eigvals_o = np.clip(eigvals_o, 0.0, None)
            omega_sqrt = eigvecs_o @ np.diag(np.sqrt(eigvals_o)) @ eigvecs_o.T
            u = u0 @ omega_sqrt

    print(f"[CoFineUCB] Learned U from {len(profiles)} train-user profiles: "
          f"context_dim={context_dim}, k={u.shape[1]}.")
    return np.asarray(u, dtype=np.float64)


class CoFineUCB:
    """Two-level coarse-to-fine linear UCB."""

    def __init__(self, dim: int, projection: np.ndarray, lamb: float = 1.0,
                 lamb_tilde: float = 1.0, alpha_fine: float = 1.0,
                 alpha_coarse: float = 1.0):
        self.d = int(dim)
        self.u = np.asarray(projection, dtype=np.float64)
        if self.u.ndim != 2 or self.u.shape[0] != self.d:
            raise ValueError(
                f"projection must have shape ({self.d}, k), got {self.u.shape}")
        self.k = int(self.u.shape[1])
        self.lamb = float(lamb)
        self.lamb_tilde = float(lamb_tilde)
        self.alpha_fine = float(alpha_fine)
        self.alpha_coarse = float(alpha_coarse)

        self.m_inv = (1.0 / self.lamb) * np.eye(self.d, dtype=np.float64)
        self.m_tilde_inv = (
            (1.0 / self.lamb_tilde) * np.eye(self.k, dtype=np.float64))
        self.b = np.zeros((self.d, 1), dtype=np.float64)
        self.b_tilde = np.zeros((self.k, 1), dtype=np.float64)

    def _theta_tilde(self) -> np.ndarray:
        return self.m_tilde_inv @ self.b_tilde

    def _theta(self) -> np.ndarray:
        return self.m_inv @ (self.b + self.lamb * self.u @ self._theta_tilde())

    def take_action(self, context: np.ndarray, beta=None):
        x = np.asarray(context, dtype=np.float64)
        theta = self._theta()
        mean = x @ theta

        fine_var = np.sum((x @ self.m_inv) * x, axis=1, keepdims=True)
        fine_bonus = np.sqrt(np.maximum(fine_var, 0.0))

        # Confidence term coupled through the full-space inverse covariance,
        # matching the coarse-to-fine confidence structure in the paper.
        probes = x @ self.m_inv.T
        coarse_x = probes @ self.u
        coarse_var = np.sum((coarse_x @ self.m_tilde_inv) * coarse_x,
                            axis=1, keepdims=True)
        coarse_bonus = np.sqrt(np.maximum(coarse_var, 0.0))

        fine_alpha = self.alpha_fine if beta is None else float(beta)
        scores = (mean
                  + fine_alpha * fine_bonus
                  + self.alpha_coarse * coarse_bonus)
        return int(np.argmax(scores)), scores.reshape(-1)

    def update(self, context: np.ndarray, action: int, reward: float,
               sample_weight: float = 1.0):
        x = np.asarray(context[action], dtype=np.float64)
        x_tilde = self.u.T @ x
        w = float(sample_weight)

        self.m_inv = _sherman_morrison_update(self.m_inv, x, w)
        self.m_tilde_inv = _sherman_morrison_update(
            self.m_tilde_inv, x_tilde, w)
        self.b += w * float(reward) * x.reshape(-1, 1)
        self.b_tilde += w * float(reward) * x_tilde.reshape(-1, 1)
        return 0.0


def run_cofine_ucb(env,
                   agent: CoFineUCB,
                   num_steps: int = 10000,
                   max_candidates: int = 10,
                   hit_threshold: float = 0.8,
                   allowed_users=None,
                   track_selection: bool = False,
                   allow_repeat: bool = True,
                   fixed_user=None,
                   funnel_recorder=None,
                   candidate_provider=None):
    """Run CoFineUCB with the same candidate-provider contract as TDGE."""
    regrets, rewards_log, hits, selections, loss_log = [], [], [], [], []
    user_history = defaultdict(set)

    pbar = tqdm(total=num_steps, desc="Train CoFineUCB", leave=False)
    steps_done = 0

    while steps_done < num_steps:
        if fixed_user is not None:
            env.current_user = fixed_user
        elif allowed_users is not None:
            env.current_user = int(np.random.choice(allowed_users))
        else:
            env.reset()

        user_id = int(env.current_user)
        if candidate_provider is None:
            positive_items = env.get_user_valid_items().astype(np.int64)
            candidate_items = positive_items
        else:
            candidate_items, positive_items = candidate_provider(
                env, user_history, user_id, allow_repeat)

        if len(positive_items) < 1 or len(candidate_items) < 1:
            if fixed_user is not None:
                break
            continue

        if len(candidate_items) > max_candidates:
            candidates = np.random.choice(candidate_items, size=max_candidates,
                                          replace=False)
        else:
            candidates = np.asarray(candidate_items, dtype=np.int64)

        context, rewards_batch = env.step_movie(candidates)
        rewards_batch = np.nan_to_num(rewards_batch, nan=0.0)
        action, _ = agent.take_action(context)
        best_item = int(candidates[action])
        reward = float(rewards_batch[action])

        loss = agent.update(context, action, reward)
        loss_log.append(float(loss) if loss is not None else 0.0)

        if not allow_repeat:
            user_history[user_id].add(best_item)
        if track_selection:
            selections.append(best_item)

        if funnel_recorder is not None:
            funnel_recorder.append({
                "step": int(steps_done),
                "user_id": int(user_id),
                "selected_clusters": np.array([], dtype=np.int64),
                "selected_tags": np.array([], dtype=np.int64),
                "final_candidates": np.asarray(candidates, dtype=np.int64),
                "best_item": int(best_item),
            })

        opt_reward = float(np.nanmax(env.rating_matrix[user_id, positive_items]))
        regrets.append(opt_reward - reward)
        rewards_log.append(reward)
        hits.append(1 if reward >= hit_threshold else 0)

        steps_done += 1
        pbar.update(1)

    pbar.close()
    return regrets, rewards_log, hits, hits, hits, selections, loss_log
