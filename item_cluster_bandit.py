"""
item_cluster_bandit.py
======================
2-level bandit using item-space clusters (item-space exploration).

Used as the direct ablation against TDGE's feature-cluster approach:
  TDGE:          feature cluster → tag → item   (feature space)
  This baseline: item cluster    → item          (item space)

Both use the same semantic item embeddings (semantic_movie_features) and the
same context construction, so any performance difference is attributable
solely to the choice of cluster space.

Provides:
  build_item_space_clusters(semantic_item_features, num_clusters=None, ...)
  build_genre_clusters(movies_csv, movie_ids, semantic_item_features)
  run_item_cluster_bandit(env, cluster_agent, item_agent, ...)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from sklearn.cluster import KMeans


# =============================================================================
# Preprocessing
# =============================================================================

def compute_wcss(embeddings: np.ndarray, labels: np.ndarray) -> float:
    wcss = 0.0
    for label in np.unique(labels):
        pts = embeddings[labels == label]
        if len(pts) > 0:
            wcss += float(np.sum((pts - np.mean(pts, axis=0)) ** 2))
    return wcss


def select_item_space_k_by_kgs(semantic_movie_features: np.ndarray,
                               k_range=range(10, 100),
                               min_cluster_size: int = 5,
                               random_state: int = 42):
    """Select item-space cluster count with the same KGS penalty principle."""
    x = []
    wcss_list = []
    valid_flags = []

    print(f"[ItemCluster] Evaluating KGS Penalty for item-space k in "
          f"{k_range.start} to {k_range.stop - 1}...")
    for k in tqdm(k_range, desc="Eval ItemCluster KGS", leave=False):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(semantic_movie_features).astype(int)
        x.append(k)
        wcss_list.append(compute_wcss(semantic_movie_features, labels))
        cluster_sizes = pd.Series(labels).value_counts()
        valid_flags.append(int(cluster_sizes.min()) >= min_cluster_size)

    x = np.asarray(x)
    wcss_list = np.asarray(wcss_list)
    valid_flags = np.asarray(valid_flags)

    wcss_norm = ((wcss_list - np.min(wcss_list))
                 / (np.max(wcss_list) - np.min(wcss_list) + 1e-9))
    k_norm = ((x - np.min(x))
              / (np.max(x) - np.min(x) + 1e-9))
    kgs_scores = wcss_norm + k_norm

    valid_indices = np.where(valid_flags)[0]
    if len(valid_indices) == 0:
        print(f"[ItemCluster] Warning: no valid k with min_cluster_size="
              f"{min_cluster_size}; falling back to first k.")
        best_idx = 0
    else:
        best_idx = valid_indices[np.argmin(kgs_scores[valid_indices])]

    best_k = int(x[best_idx])
    print(f"[ItemCluster] KGS selected item-space k={best_k} "
          f"(score={float(kgs_scores[best_idx]):.4f}).")
    return best_k, x, kgs_scores, best_idx, valid_flags

def build_item_space_clusters(semantic_movie_features: np.ndarray,
                              num_clusters: int | None = None,
                              random_state: int = 42):
    """
    Cluster items directly in the semantic movie embedding space.

    If num_clusters is None, it is selected by KGS penalty on the item
    embeddings, independently from TDGE's feature-space clusters.

    Returns
    -------
    cluster_names: list[str]
    centroids: np.ndarray (num_clusters, feat_dim)
    cluster_movie_dict: dict {cluster_idx: np.ndarray(env_movie_indices)}
    movie_to_clusters: dict {env_movie_idx: list[cluster_idx]}
    """
    if num_clusters is None:
        num_clusters, _, _, _, _ = select_item_space_k_by_kgs(
            semantic_movie_features, random_state=random_state)

    model = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(semantic_movie_features).astype(int)

    cluster_movie_dict: dict[int, np.ndarray] = {}
    centroids = []
    for cluster_idx in range(num_clusters):
        env_idxs = np.flatnonzero(labels == cluster_idx).astype(np.int64)
        cluster_movie_dict[cluster_idx] = env_idxs
        if len(env_idxs) == 0:
            centroids.append(np.zeros(semantic_movie_features.shape[1]))
        else:
            centroids.append(semantic_movie_features[env_idxs].mean(axis=0))

    centroids = np.asarray(centroids, dtype=np.float32)
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    movie_to_clusters = {int(i): [int(labels[i])] for i in range(len(labels))}
    cluster_names = [f"item_cluster_{i}" for i in range(num_clusters)]

    sizes = [len(cluster_movie_dict[i]) for i in range(num_clusters)]
    print(f"[ItemCluster] Built {num_clusters} item-space clusters.")
    print(f"[ItemCluster] Cluster sizes (items per item cluster): {sizes}")

    return cluster_names, centroids, cluster_movie_dict, movie_to_clusters

def build_genre_clusters(movies_csv_path: str,
                         movie_ids: np.ndarray,
                         semantic_movie_features: np.ndarray):
    """
    Build genre-based item clusters from MovieLens movies.csv.

    Each unique genre becomes one cluster.  The cluster centroid is the
    mean of semantic_movie_features for all movies in that genre —
    the exact same feature space TDGE uses, ensuring a fair comparison.

    Parameters
    ----------
    movies_csv_path        : path to MovieLens movies.csv
                             (columns: movieId, title, genres)
    movie_ids              : (num_movies,) array of real movieIds in env order
    semantic_movie_features: (num_movies, feat_dim) semantic embeddings

    Returns
    -------
    genre_names    : list[str]  — ordered list of genre names (idx = cluster id)
    genre_centroids: np.ndarray (num_genres, feat_dim) — cluster centroids
    genre_movie_dict: dict {genre_idx: np.ndarray(env_movie_indices)}
    movie_to_genres: dict {env_movie_idx: list[genre_idx]}
    """
    df = pd.read_csv(movies_csv_path)

    movie_id_to_env = {int(mid): i for i, mid in enumerate(movie_ids)}

    genre_to_env_movies: dict[str, list[int]] = defaultdict(list)
    env_movie_to_genres: dict[int, list[str]] = {}

    for _, row in df.iterrows():
        mid = int(row['movieId'])
        if mid not in movie_id_to_env:
            continue
        env_idx   = movie_id_to_env[mid]
        genres_str = str(row['genres'])
        if genres_str == '(no genres listed)':
            continue
        genres = genres_str.split('|')
        env_movie_to_genres[env_idx] = genres
        for g in genres:
            genre_to_env_movies[g].append(env_idx)

    # Reproducible ordering
    genre_names = sorted(genre_to_env_movies.keys())

    genre_movie_dict: dict[int, np.ndarray] = {}
    centroid_list: list[np.ndarray] = []

    for i, g in enumerate(genre_names):
        env_idxs = np.array(genre_to_env_movies[g], dtype=np.int64)
        genre_movie_dict[i] = env_idxs
        centroid = semantic_movie_features[env_idxs].mean(axis=0)
        centroid_list.append(centroid)

    genre_centroids = np.stack(centroid_list, axis=0)  # (num_genres, feat_dim)

    # movie_idx → list of genre indices
    movie_to_genres: dict[int, list[int]] = {
        env_idx: [genre_names.index(g) for g in gnames]
        for env_idx, gnames in env_movie_to_genres.items()
    }

    print(f"[ItemCluster] {len(genre_names)} genre clusters: {genre_names}")
    sizes = [len(genre_movie_dict[i]) for i in range(len(genre_names))]
    print(f"[ItemCluster] Cluster sizes (movies per genre): {sizes}")

    return genre_names, genre_centroids, genre_movie_dict, movie_to_genres


# =============================================================================
# Runner
# =============================================================================

def run_item_cluster_bandit(env,
                            cluster_agent,
                            item_agent,
                            item_cluster_centroids: np.ndarray,
                            item_cluster_movie_dict: dict,
                            movie_to_item_clusters: dict,
                            num_steps: int       = 10000,
                            max_candidates: int  = 50,
                            hit_threshold: float = 0.8,
                            top_k_clusters: int  = 2,
                            allowed_users        = None,
                            track_selection: bool= False,
                            allow_repeat: bool   = True,
                            fixed_user           = None,
                            funnel_recorder      = None,
                            candidate_provider   = None):
    """
    2-level item-cluster bandit runner.

    Level 1: cluster_agent selects top-K item clusters.
    Level 2: item_agent selects best item from candidates in those clusters.

    Update rule (mirrors TDGE's hard-intersection):
      - item_agent updated with selected item and reward.
      - cluster_agent updated for each item cluster in
        selected_clusters ∩ item's own clusters (active clusters).

    Returns
    -------
    7-tuple  (regrets, rewards, hits, hits_5, hits_10, selections, loss_log)
    compatible with train_movielens.py result aggregation.
    """
    regrets, rewards_log, hits, selections, loss_log = [], [], [], [], []
    user_history: dict = defaultdict(set)

    num_clusters = item_cluster_centroids.shape[0]

    def make_ctx(user_emb: np.ndarray, item_feats: np.ndarray) -> np.ndarray:
        """Identical to TDGE's make_context helper."""
        N   = item_feats.shape[0]
        raw = np.concatenate([np.tile(user_emb, (N, 1)), item_feats], axis=1)
        raw = np.concatenate([raw, np.ones((N, 1)) * 0.01], axis=1)
        return raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-9)

    pbar = tqdm(total=num_steps, desc="Train ItemCluster", leave=False)
    steps_done = 0

    while steps_done < num_steps:

        # --------------------------------------------------------- user/items
        if fixed_user is not None:
            env.current_user = fixed_user
            user_emb = env.user_features[fixed_user]
        elif allowed_users is not None:
            env.current_user = np.random.choice(allowed_users)
            user_emb = env.user_features[env.current_user]
        else:
            user_emb = env.reset()

        user_id = env.current_user
        if candidate_provider is None:
            positive_items = env.get_user_valid_items()
            candidate_items = positive_items
            if not allow_repeat:
                watched = user_history[user_id]
                if watched:
                    candidate_items = np.setdiff1d(candidate_items, list(watched))
                    positive_items = np.setdiff1d(positive_items, list(watched))
        else:
            candidate_items, positive_items = candidate_provider(
                env, user_history, user_id, allow_repeat)

        if len(positive_items) < 1 or len(candidate_items) < 1:
            if fixed_user is not None:
                break
            continue

        # --------------------------------------------------- cluster scoring
        cluster_ctx = make_ctx(user_emb, item_cluster_centroids)
        _, cluster_scores = cluster_agent.take_action(cluster_ctx)
        scores_arr  = np.asarray(cluster_scores).flatten()
        k           = min(top_k_clusters, num_clusters)
        top_k_idx   = np.argsort(scores_arr)[-k:]           # local cluster ids

        # ------------------------------------------- candidate item filtering
        cand_set: set = set()
        for c_idx in top_k_idx:
            items_in_c = item_cluster_movie_dict.get(int(c_idx), np.array([], dtype=np.int64))
            cand_set.update(np.intersect1d(candidate_items, items_in_c).tolist())

        if len(cand_set) == 0:
            candidates = candidate_items
        else:
            candidates = np.array(list(cand_set), dtype=np.int64)

        if len(candidates) > max_candidates:
            candidates = np.random.choice(candidates, size=max_candidates, replace=False)

        # ----------------------------------------------- item-level selection
        item_ctx, rewards = env.step_movie(candidates)
        rewards = np.nan_to_num(rewards, nan=0.0)
        m_idx, _          = item_agent.take_action(item_ctx)
        best_item         = int(candidates[m_idx])
        reward            = float(rewards[m_idx])

        # ---- hard-intersection update (mirrors TDGE cluster update logic) --
        # Active clusters = selected_clusters ∩ item_clusters(best_item)
        item_cluster_set = set(movie_to_item_clusters.get(best_item, []))
        active_clusters = [int(c) for c in top_k_idx if int(c) in item_cluster_set]
        if not active_clusters:
            active_clusters = [int(top_k_idx[-1])]  # fallback: top cluster

        update_losses = []
        item_loss = item_agent.update(item_ctx, m_idx, reward, sample_weight=1.0)
        if item_loss is not None:
            update_losses.append(float(item_loss))

        for ac in active_clusters:
            cluster_loss = cluster_agent.update(
                cluster_ctx, ac, reward, sample_weight=1.0)
            if cluster_loss is not None:
                update_losses.append(float(cluster_loss))

        # ----------------------------------------------- bookkeeping
        if not allow_repeat:
            user_history[user_id].add(best_item)
        if track_selection:
            selections.append(best_item)

        if funnel_recorder is not None:
            funnel_recorder.append({
                'step':             steps_done,
                'user_id':          int(user_id),
                'selected_clusters': np.array(top_k_idx, dtype=np.int64),
                'selected_tags':    np.array([], dtype=np.int64),
                'final_candidates': candidates,
                'best_item':        best_item,
            })

        # ----------------------------------------------- metrics
        opt_reward = float(np.nanmax(env.rating_matrix[user_id, positive_items]))
        regrets.append(opt_reward - reward)
        rewards_log.append(reward)
        hits.append(1 if reward >= hit_threshold else 0)
        loss_log.append(float(np.mean(update_losses)) if update_losses else 0.0)

        steps_done += 1
        pbar.update(1)

    pbar.close()
    return regrets, rewards_log, hits, hits, hits, selections, loss_log
