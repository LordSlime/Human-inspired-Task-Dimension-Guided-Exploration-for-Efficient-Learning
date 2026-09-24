"""
TDGE ablation runner — no feature-dimension (cluster) level.

Mirrors `run_lowd_hierec` exactly except for the tag-candidate generation
step: instead of pulling tags from clusters selected by a cluster_agent,
this runner draws `num_tag_arms` tags uniformly at random from the full
tag pool each round and lets the tag_agent rank them.

Everything else — context construction, item filtering, item-agent selection,
hard-intersection update of (item, active tags), regret/reward/hit logging,
fallback path — should be configured to match `run_lowd_hierec`, so any
performance gap vs the full TDGE is attributable to the cluster level.

The runner is dataset-agnostic. Pass the dataset's `item_to_tags` and
`item_tag_relevance` dicts (movie_to_tags, artist_to_tags, ... are all fine).

Return signature is identical to `run_lowd_hierec`:
    (regrets, rewards_log, hits, hits_5, hits_10, selections, loss_log)
"""

import numpy as np
from collections import defaultdict
from tqdm import tqdm


def _build_tag_scored_candidates(candidate_items, selected_tag_scores,
                                 item_to_tags, item_tag_relevance,
                                 max_candidates):
    candidate_items = np.asarray(candidate_items, dtype=np.int64)
    if len(candidate_items) == 0:
        return candidate_items

    target_size = min(max_candidates, len(candidate_items))
    item_scores = defaultdict(float)

    for item_idx in candidate_items:
        item_idx = int(item_idx)
        current_relevances = item_tag_relevance.get(item_idx, {})
        for tag_idx in item_to_tags.get(item_idx, set()):
            tag_idx = int(tag_idx)
            if tag_idx in selected_tag_scores:
                relevance = current_relevances.get(tag_idx, 1.0)
                item_scores[item_idx] += (
                    float(selected_tag_scores[tag_idx]) * float(relevance)
                )

    if item_scores:
        ranked_items = sorted(
            item_scores,
            key=lambda item_idx: (item_scores[item_idx], item_idx),
            reverse=True,
        )[:target_size]
        return np.asarray(ranked_items, dtype=np.int64)

    fill = np.random.choice(candidate_items, size=target_size, replace=False)
    return fill.astype(np.int64)


def run_lowd_hierec_no_fd(env,
                          tag_agent,
                          item_agent,
                          tag_features,
                          item_to_tags,
                          item_tag_relevance,
                          num_tag_arms=50,
                          top_k_tags=12,
                          max_candidates=50,
                          hit_threshold=0.8,
                          num_steps=10000,
                          allowed_users=None,
                          track_selection=False,
                          allow_repeat=True,
                          fixed_user=None,
                          funnel_recorder=None,
                          tag_selection_counter=None,
                          candidate_provider=None,
                          item_filter_mode="fusion"):
    """
    Ablation: TDGE without the feature-dimension (cluster) level.

    Parameters
    ----------
    env                 : Movielens/LastFM/Amazon env (env.movie_features is
                          the item embedding matrix in all three).
    tag_agent           : feature-level agent (LinUCB / TagNeuralUCB / TagNeuralTS)
    item_agent          : item-level agent (LinUCB / NeuralUCBDiag / NeuralTSDiag)
    tag_features        : (num_tags, d_tag) tag embeddings
    item_to_tags        : dict item_idx -> set of tag indices
    item_tag_relevance  : dict item_idx -> {tag_idx: relevance_weight}
    num_tag_arms        : number of tags sampled uniformly each round (default 50)
    top_k_tags          : how many of the sampled tags survive to item filtering
    max_candidates      : item-pool size for the item_agent
    hit_threshold       : reward threshold for hit-rate metric
    candidate_provider  : optional callback matching the main training candidate
                          source, returning (candidate_items, positive_items)
    item_filter_mode    : "score" for tag-score retrieval, "fusion" for the
                          legacy intent-vector retrieval
    """

    regrets, rewards_log, hits, selections, loss_log = [], [], [], [], []
    user_history = defaultdict(set)
    num_tags = tag_features.shape[0]
    all_tag_indices = np.arange(num_tags)
    n_arms = min(num_tag_arms, num_tags)

    def make_context(user_vec, item_feats):
        N = item_feats.shape[0]
        ctx_raw = np.concatenate([np.tile(user_vec, (N, 1)), item_feats], axis=1)
        ctx = np.concatenate([ctx_raw, np.ones((N, 1)) * 0.01], axis=1)
        return ctx / (np.linalg.norm(ctx, axis=1, keepdims=True) + 1e-9)

    pbar = tqdm(total=num_steps,
                desc="Train LowD_HieRec NoFD (random tag arms)",
                leave=False)
    steps_done = 0

    while steps_done < num_steps:
        # ---------------------------------------------------------------- user
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
                watched = list(user_history[user_id])
                if len(watched) > 0:
                    candidate_items = np.setdiff1d(candidate_items, watched)
                    positive_items = np.setdiff1d(positive_items, watched)
        else:
            candidate_items, positive_items = candidate_provider(
                env, user_history, user_id, allow_repeat)

        if len(positive_items) < 1 or len(candidate_items) < 1:
            if fixed_user is not None:
                break
            continue

        # =================================================================
        # Tag level (NO cluster level) — uniform-random arm sampling
        # =================================================================
        candidate_tags = np.random.choice(all_tag_indices,
                                          size=n_arms,
                                          replace=False)
        tag_contexts = make_context(user_emb, tag_features[candidate_tags])
        _, tag_scores_subset = tag_agent.take_action(tag_contexts)
        k_t = min(top_k_tags, n_arms)
        selected_positions = np.argsort(tag_scores_subset)[-k_t:]
        selected_tag_indices = candidate_tags[selected_positions]
        selected_tag_scores = {
            int(candidate_tags[pos]): float(tag_scores_subset[pos])
            for pos in selected_positions
        }

        if tag_selection_counter is not None and len(selected_tag_indices) > 0:
            for t_idx in selected_tag_indices:
                tag_selection_counter[int(t_idx)] += 1

        # =================================================================
        # Item-level: keep the filtering mode aligned with the main TDGE path.
        # =================================================================
        is_fallback = len(selected_tag_indices) == 0
        if item_filter_mode == "score":
            final_candidates = _build_tag_scored_candidates(
                candidate_items, selected_tag_scores, item_to_tags,
                item_tag_relevance, max_candidates)
        elif len(selected_tag_indices) > 0:
            selected_tag_feats = tag_features[selected_tag_indices]
            intent_vector = np.mean(selected_tag_feats, axis=0)
            intent_vector = intent_vector / (np.linalg.norm(intent_vector) + 1e-9)

            candidate_item_feats = env.movie_features[candidate_items]
            sim_scores = np.dot(candidate_item_feats, intent_vector)

            top_n = min(max_candidates, len(candidate_items))
            top_idx = np.argsort(sim_scores)[-top_n:]
            final_candidates = candidate_items[top_idx]
        else:
            top_n = min(max_candidates, len(candidate_items))
            final_candidates = np.random.choice(candidate_items, size=top_n, replace=False)

        final_candidates = np.array(final_candidates)
        if len(final_candidates) < 1:
            if fixed_user is not None:
                break
            continue

        cand_item_feats = env.movie_features[final_candidates]
        cand_contexts = make_context(user_emb, cand_item_feats)

        best_cand_idx, _ = item_agent.take_action(cand_contexts)
        best_item_idx = final_candidates[best_cand_idx]

        _, rewards_batch = env.step_movie(final_candidates)
        rewards_batch = np.nan_to_num(rewards_batch, nan=0.0)
        reward = rewards_batch[best_cand_idx]

        if track_selection:
            selections.append(best_item_idx)
        if not allow_repeat:
            user_history[user_id].add(best_item_idx)

        # =================================================================
        # Hard-intersection update — item + tag only (no cluster)
        # =================================================================
        loss_val = item_agent.update(cand_contexts, best_cand_idx, reward,
                                     sample_weight=1.0)
        loss_log.append(loss_val if loss_val is not None else 0.0)

        if not is_fallback:
            real_item_tags = set(item_to_tags.get(best_item_idx, set()))
            current_item_relevances = item_tag_relevance.get(best_item_idx, {})
            active_tags = list(real_item_tags.intersection(set(selected_tag_indices)))

            for t_idx in active_tags:
                t_ctx = make_context(user_emb, tag_features[t_idx].reshape(1, -1))
                rel_weight = current_item_relevances.get(t_idx, 1.0)
                tag_agent.update(t_ctx, 0, reward, sample_weight=rel_weight)

        # =================================================================
        # Metrics
        # =================================================================
        opt_reward = np.nanmax(env.rating_matrix[user_id, positive_items])
        regrets.append(opt_reward - reward)
        rewards_log.append(reward)
        hits.append(1 if reward >= hit_threshold else 0)

        if funnel_recorder is not None:
            funnel_recorder.append({
                'step':              steps_done,
                'user_id':           int(user_id),
                'selected_clusters': np.asarray([], dtype=np.int64),   # no FD
                'selected_tags':     np.asarray(selected_tag_indices, dtype=np.int64),
                'final_candidates':  np.asarray(final_candidates,     dtype=np.int64),
                'best_item':         int(best_item_idx),
            })

        steps_done += 1
        pbar.update(1)

    pbar.close()
    # match run_lowd_hierec's 7-tuple (hits is duplicated for h5/h10 slots)
    return regrets, rewards_log, hits, hits, hits, selections, loss_log
