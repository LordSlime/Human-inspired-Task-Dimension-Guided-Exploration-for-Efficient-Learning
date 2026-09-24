"""Generic multi-level TDGE runner used by hierarchy-depth ablations."""

from collections import defaultdict
import heapq

import numpy as np
from tqdm import tqdm


def _make_context(user_vec, action_features):
    action_features = np.asarray(action_features)
    count = action_features.shape[0]
    raw = np.concatenate(
        [np.tile(user_vec, (count, 1)), action_features], axis=1)
    context = np.concatenate(
        [raw, np.full((count, 1), 0.01, dtype=raw.dtype)], axis=1)
    return context / (np.linalg.norm(context, axis=1, keepdims=True) + 1e-9)


def _rank_items_by_selected_features(
        valid_items, selected_feature_scores, feature_to_items,
        item_feature_relevance, max_candidates):
    """Return relevance-weighted items with nonempty selected-feature overlap."""
    valid_items = np.asarray(valid_items, dtype=np.int64)
    if len(valid_items) == 0:
        return valid_items

    target_size = min(int(max_candidates), len(valid_items))
    valid_set = set(valid_items.tolist())
    item_scores = defaultdict(float)

    for feature_idx, feature_score in selected_feature_scores.items():
        for item_idx in feature_to_items.get(int(feature_idx), set()):
            item_idx = int(item_idx)
            if item_idx not in valid_set:
                continue
            relevance = item_feature_relevance.get(item_idx, {}).get(
                int(feature_idx), 1.0)
            item_scores[item_idx] += float(feature_score) * float(relevance)

    if not item_scores:
        return np.asarray([], dtype=np.int64)

    ranked = heapq.nlargest(
        target_size,
        item_scores,
        key=lambda item_idx: (item_scores[item_idx], item_idx),
    )
    return np.asarray(ranked, dtype=np.int64)


def validate_dimension_hierarchy(dimension_features, parent_to_children):
    """Validate an ordered coarse-to-fine hierarchy with one parent per child."""
    num_levels = len(dimension_features)
    if num_levels < 1:
        raise ValueError("TDGE requires at least one dimension level.")
    if len(parent_to_children) != num_levels - 1:
        raise ValueError(
            "parent_to_children must contain one mapping per adjacent "
            "dimension-level pair."
        )

    for level_idx, mapping in enumerate(parent_to_children):
        parent_count = int(dimension_features[level_idx].shape[0])
        child_count = int(dimension_features[level_idx + 1].shape[0])
        child_parents = defaultdict(set)

        for parent_idx, children in mapping.items():
            parent_idx = int(parent_idx)
            if not 0 <= parent_idx < parent_count:
                raise ValueError(
                    f"Invalid parent {parent_idx} at dimension level {level_idx}."
                )
            for child_idx in children:
                child_idx = int(child_idx)
                if not 0 <= child_idx < child_count:
                    raise ValueError(
                        f"Invalid child {child_idx} at dimension level {level_idx + 1}."
                    )
                child_parents[child_idx].add(parent_idx)

        missing = sorted(set(range(child_count)) - set(child_parents))
        multiple = sorted(
            child for child, parents in child_parents.items() if len(parents) != 1)
        if missing or multiple:
            raise ValueError(
                f"Dimension levels {level_idx}->{level_idx + 1} must form a "
                f"strict hierarchy; missing children={missing[:5]}, "
                f"multi-parent children={multiple[:5]}."
            )


def run_multilevel_tdge(
        env,
        dimension_agents,
        feature_agent,
        item_agent,
        dimension_features,
        parent_to_children,
        lowest_dimension_to_features,
        feature_features,
        feature_to_items,
        item_to_features,
        item_feature_relevance,
        candidate_provider,
        num_steps=10000,
        max_candidates=10,
        hit_threshold=0.8,
        top_k_dimensions=None,
        top_k_features=10,
        allowed_users=None,
        track_selection=False,
        allow_repeat=True,
        fixed_user=None,
        funnel_recorder=None,
        feature_selection_counter=None,
        show_progress=True,
):
    """Run TDGE with one or more semantic dimension levels.

    ``dimension_features`` are ordered from the coarsest to the finest level.
    With one dimension level this is the original three-level TDGE; with two it
    is the four-level dimension-1 -> dimension-2 -> feature -> item ablation.
    """
    dimension_agents = list(dimension_agents)
    dimension_features = [np.asarray(features) for features in dimension_features]
    parent_to_children = list(parent_to_children)

    if len(dimension_agents) != len(dimension_features):
        raise ValueError("Each dimension level requires one bandit agent.")
    validate_dimension_hierarchy(dimension_features, parent_to_children)

    if top_k_dimensions is None:
        top_k_dimensions = [features.shape[0] for features in dimension_features]
    top_k_dimensions = [int(value) for value in top_k_dimensions]
    if len(top_k_dimensions) != len(dimension_features):
        raise ValueError("top_k_dimensions must specify one value per dimension level.")
    if any(value < 1 for value in top_k_dimensions):
        raise ValueError("All top-k values must be positive.")

    regrets, rewards_log, hits, selections, loss_log = [], [], [], [], []
    user_history = defaultdict(set)
    progress = tqdm(
        total=num_steps,
        desc=f"Train TDGE-{len(dimension_features) + 2}L",
        leave=False,
        disable=not show_progress,
    )
    steps_done = 0

    while steps_done < num_steps:
        if fixed_user is not None:
            env.current_user = fixed_user
            user_emb = env.user_features[fixed_user]
        elif allowed_users is not None:
            env.current_user = np.random.choice(allowed_users)
            user_emb = env.user_features[env.current_user]
        else:
            user_emb = env.reset()

        user_id = int(env.current_user)
        candidate_items, positive_items = candidate_provider(
            env, user_history, user_id, allow_repeat)
        candidate_items = np.asarray(candidate_items, dtype=np.int64)
        positive_items = np.asarray(positive_items, dtype=np.int64)

        if len(positive_items) < 1 or len(candidate_items) < 1:
            if fixed_user is not None:
                break
            continue

        selected_dimensions = []
        child_to_selected_parent = []
        level_candidates = np.arange(
            dimension_features[0].shape[0], dtype=np.int64)

        for level_idx, (agent, level_features, top_k) in enumerate(zip(
                dimension_agents, dimension_features, top_k_dimensions)):
            contexts = _make_context(user_emb, level_features[level_candidates])
            _, scores = agent.take_action(contexts)
            selected_positions = np.argsort(scores)[-min(top_k, len(level_candidates)):]
            selected = level_candidates[selected_positions]
            selected_dimensions.append(np.asarray(selected, dtype=np.int64))

            if level_idx < len(dimension_features) - 1:
                next_candidates = []
                source_parent = {}
                mapping = parent_to_children[level_idx]
                for parent_idx in selected:
                    for child_idx in mapping.get(int(parent_idx), []):
                        child_idx = int(child_idx)
                        next_candidates.append(child_idx)
                        source_parent[child_idx] = int(parent_idx)
                level_candidates = np.unique(next_candidates).astype(np.int64)
                child_to_selected_parent.append(source_parent)
                if len(level_candidates) == 0:
                    break

        complete_route = len(selected_dimensions) == len(dimension_features)
        selected_features = np.asarray([], dtype=np.int64)
        selected_feature_scores = {}
        feature_source_dimension = {}

        if complete_route:
            candidate_features = []
            for dimension_idx in selected_dimensions[-1]:
                for feature_idx in lowest_dimension_to_features.get(
                        int(dimension_idx), set()):
                    feature_idx = int(feature_idx)
                    candidate_features.append(feature_idx)
                    feature_source_dimension[feature_idx] = int(dimension_idx)
            candidate_features = np.unique(candidate_features).astype(np.int64)

            if len(candidate_features) > 0:
                feature_contexts = _make_context(
                    user_emb, feature_features[candidate_features])
                _, feature_scores = feature_agent.take_action(feature_contexts)
                selected_positions = np.argsort(feature_scores)[
                    -min(int(top_k_features), len(candidate_features)):]
                selected_features = candidate_features[selected_positions]
                selected_feature_scores = {
                    int(candidate_features[pos]): float(feature_scores[pos])
                    for pos in selected_positions
                }

        if feature_selection_counter is not None:
            for feature_idx in selected_features:
                feature_selection_counter[int(feature_idx)] += 1

        final_candidates = _rank_items_by_selected_features(
            candidate_items,
            selected_feature_scores,
            feature_to_items,
            item_feature_relevance,
            max_candidates,
        )
        if len(final_candidates) == 0:
            # Do not create an item outside the selected semantic route: such
            # a fallback would reintroduce zero-credit upper-level updates.
            if fixed_user is not None:
                break
            continue

        item_contexts = _make_context(user_emb, env.movie_features[final_candidates])
        selected_item_position, _ = item_agent.take_action(item_contexts)
        selected_item = int(final_candidates[selected_item_position])

        _, reward_batch = env.step_movie(final_candidates)
        reward_batch = np.nan_to_num(reward_batch, nan=0.0)
        reward = float(reward_batch[selected_item_position])

        if track_selection:
            selections.append(selected_item)
        if not allow_repeat:
            user_history[user_id].add(selected_item)

        loss = item_agent.update(
            item_contexts, selected_item_position, reward, sample_weight=1.0)
        loss_log.append(loss if loss is not None else 0.0)

        credited_dimensions = [dict() for _ in dimension_features]
        selected_item_features = set(item_to_features.get(selected_item, set()))
        active_features = selected_item_features.intersection(
            set(selected_features.tolist()))
        current_relevances = item_feature_relevance.get(selected_item, {})

        for feature_idx in active_features:
            relevance = float(current_relevances.get(int(feature_idx), 1.0))
            feature_context = _make_context(
                user_emb, feature_features[int(feature_idx)].reshape(1, -1))
            feature_agent.update(
                feature_context, 0, reward, sample_weight=relevance)

            dimension_idx = feature_source_dimension.get(int(feature_idx))
            if dimension_idx is not None:
                previous = credited_dimensions[-1].get(dimension_idx, 0.0)
                credited_dimensions[-1][dimension_idx] = max(previous, relevance)

        for level_idx in range(len(dimension_features) - 1, -1, -1):
            for dimension_idx, relevance in credited_dimensions[level_idx].items():
                dimension_context = _make_context(
                    user_emb,
                    dimension_features[level_idx][dimension_idx].reshape(1, -1),
                )
                dimension_agents[level_idx].update(
                    dimension_context, 0, reward, sample_weight=relevance)

                if level_idx > 0:
                    parent_idx = child_to_selected_parent[level_idx - 1].get(
                        int(dimension_idx))
                    if parent_idx is not None:
                        previous = credited_dimensions[level_idx - 1].get(
                            parent_idx, 0.0)
                        credited_dimensions[level_idx - 1][parent_idx] = max(
                            previous, relevance)

        optimal_reward = float(np.nanmax(env.rating_matrix[user_id, positive_items]))
        regrets.append(optimal_reward - reward)
        rewards_log.append(reward)
        hits.append(1 if reward >= hit_threshold else 0)

        if funnel_recorder is not None:
            funnel_recorder.append({
                "step": steps_done,
                "user_id": user_id,
                "selected_dimension_levels": [
                    values.copy() for values in selected_dimensions],
                "selected_clusters": selected_dimensions[-1].copy()
                    if selected_dimensions else np.asarray([], dtype=np.int64),
                "selected_tags": selected_features.copy(),
                "credited_dimension_levels": [
                    np.asarray(sorted(values), dtype=np.int64)
                    for values in credited_dimensions],
                "credited_dimensions": np.asarray(
                    sorted(credited_dimensions[-1]), dtype=np.int64),
                "final_candidates": final_candidates.copy(),
                "best_item": selected_item,
                "reward": reward,
            })

        steps_done += 1
        progress.update(1)

    progress.close()
    return regrets, rewards_log, hits, hits, hits, selections, loss_log
