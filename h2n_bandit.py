import math
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _FCNetwork(nn.Module):
    def __init__(self, dim, hidden_size=200):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))


class H2N_Bandit:
    """H2N-style two-level hierarchical neural contextual bandit.

    This is an adapter of the original H2N implementation for the TDGE
    recommendation environments. The algorithmic structure is kept as the original
    H2N exploitation-exploration design: category-level neural exploitation
    plus gradient-based exploration, followed by the H2N neural arm-selection
    model with a diagonal confidence term.
    """

    def __init__(self, dim, lambda_=1e-3, arm_nu=1e-3, hidden=200,
                 sample_num=1000, pool_size=50, category_lr=1e-3,
                 arm_lr=1e-3, buffer_size=-1, training_freq_category=(1, 1),
                 device_override=None):
        self.device = device_override or device
        self.input_dim = int(dim)
        self.lambda_ = float(lambda_)
        self.arm_nu = float(arm_nu)
        self.sample_num = int(sample_num)
        self.pool_size = int(pool_size)
        self.category_lr = float(category_lr)
        self.arm_lr = float(arm_lr)
        self.buffer_size = int(buffer_size)
        self.training_freq_category = tuple(training_freq_category)

        self.category_exploit_func = _FCNetwork(
            2 * self.input_dim, hidden_size=hidden).to(self.device)
        self.arm_func = _FCNetwork(
            self.input_dim, hidden_size=hidden).to(self.device)

        self.context_list_category = []
        self.reward_category = []
        self.gradient_list_category = []
        self.exp_score_category = []
        self.context_list_arm = []
        self.reward_arm = []

        self.total_param_category = sum(
            p.numel() for p in self.category_exploit_func.parameters()
            if p.requires_grad)
        self.total_param_arm = sum(
            p.numel() for p in self.arm_func.parameters() if p.requires_grad)

        self.pool_multiplier = 3
        if self.pool_size > 0:
            self.category_reduced_grad_dim = (
                math.ceil(self.total_param_category / self.pool_size)
                - (self.pool_multiplier - 1)
            )
        else:
            self.category_reduced_grad_dim = self.total_param_category
        self.category_reduced_grad_dim = max(1, self.category_reduced_grad_dim)

        self.category_explore_func = _FCNetwork(
            self.category_reduced_grad_dim, hidden_size=hidden).to(self.device)
        self.arm_U = self.lambda_ * torch.ones(
            (self.total_param_arm,), device=self.device)

    def select_category_neural_exp(self, category_context, top_k_category_num,
                                   t, category_exploration_coef=1.0):
        if len(category_context) == 0:
            return np.asarray([], dtype=np.int64), None, None

        f_weights = OrderedDict(self.category_exploit_func.named_parameters())
        tensor = torch.from_numpy(category_context).float().to(self.device)
        category_rwd_est = self.category_exploit_func(tensor).reshape(-1)

        g_list = []
        for fx in category_rwd_est:
            grad_tuple = torch.autograd.grad(
                fx, f_weights.values(), create_graph=True)
            g_list.append(torch.cat([g.detach().reshape(-1) for g in grad_tuple]))

        g_vec = torch.stack(g_list, dim=0)
        if self.pool_size > 0:
            g_vec = F.avg_pool1d(
                g_vec.unsqueeze(0),
                kernel_size=self.pool_multiplier * self.pool_size,
                stride=self.pool_size,
                ceil_mode=True,
            ).squeeze(0)
        g_vec = F.normalize(g_vec, dim=1)

        exp_score_est = self.category_explore_func(g_vec).reshape(-1)
        score = category_rwd_est + category_exploration_coef * exp_score_est
        k = min(int(top_k_category_num), score.numel())
        chosen = torch.topk(score, k).indices.detach().cpu().numpy()
        return chosen.astype(np.int64), category_rwd_est.detach(), g_vec.detach()

    def select_arm(self, arm_context, t, arm_exploitation_coef=1.0):
        if len(arm_context) == 0:
            return None, []

        f_weights = OrderedDict(self.arm_func.named_parameters())
        tensor = torch.from_numpy(arm_context).float().to(self.device)
        mu = self.arm_func(tensor).reshape(-1)

        g_list = []
        scores = []
        for fx in mu:
            grad_tuple = torch.autograd.grad(
                fx, f_weights.values(), create_graph=True)
            g = torch.cat([grad.detach().reshape(-1) for grad in grad_tuple])
            g_list.append(g)

            sigma2 = self.lambda_ * self.arm_nu * g * g / self.arm_U
            sigma = torch.sqrt(torch.sum(sigma2))
            scores.append(float(arm_exploitation_coef * fx.item() + sigma.item()))

        arm = int(np.argmax(scores))
        self.arm_U += g_list[arm] * g_list[arm]
        return arm, scores

    def update_category_info(self, category_contexts, category_rewards,
                             chosen_category_ids, category_gradients,
                             category_exp_scores,
                             artificial_category_exp_val=-1.0):
        if len(chosen_category_ids) == 0:
            return

        for local_pos, category_id in enumerate(chosen_category_ids):
            self.context_list_category.append(torch.from_numpy(
                category_contexts[local_pos].reshape(1, -1)).float())
            self.reward_category.append(float(category_rewards[local_pos]))

            self.gradient_list_category.append(
                category_gradients[int(category_id)].reshape(1, -1).float())
            self.exp_score_category.append(torch.tensor(
                [float(category_exp_scores[local_pos])], dtype=torch.float32))

        if artificial_category_exp_val > 0:
            chosen_set = set(int(i) for i in chosen_category_ids)
            for c_i in range(category_gradients.shape[0]):
                if c_i not in chosen_set:
                    self.gradient_list_category.append(
                        category_gradients[c_i].reshape(1, -1).float())
                    self.exp_score_category.append(torch.tensor(
                        [float(artificial_category_exp_val)], dtype=torch.float32))

    def update_arm_info(self, arm_context, reward):
        self.context_list_arm.append(torch.from_numpy(
            arm_context.reshape(1, -1)).float())
        self.reward_arm.append(float(reward))

    def _trim_buffers(self):
        if self.buffer_size <= 0:
            return
        self.context_list_category = self.context_list_category[-self.buffer_size:]
        self.reward_category = self.reward_category[-self.buffer_size:]
        self.gradient_list_category = self.gradient_list_category[-self.buffer_size:]
        self.exp_score_category = self.exp_score_category[-self.buffer_size:]
        self.context_list_arm = self.context_list_arm[-self.buffer_size:]
        self.reward_arm = self.reward_arm[-self.buffer_size:]

    def train_category_model(self, tt):
        self._trim_buffers()
        losses = []

        if (len(self.reward_category) > 0
                and tt % self.training_freq_category[0] == 0):
            optimizer_exploit = optim.AdamW(
                self.category_exploit_func.parameters(), lr=self.category_lr)
            losses.append(self._train_supervised(
                optimizer_exploit, self.category_exploit_func,
                self.context_list_category, self.reward_category,
                sample_num=self.sample_num))

        if (len(self.exp_score_category) > 0
                and tt % self.training_freq_category[1] == 0):
            optimizer_explore = optim.AdamW(
                self.category_explore_func.parameters(), lr=self.category_lr)
            losses.append(self._train_supervised(
                optimizer_explore, self.category_explore_func,
                self.gradient_list_category, self.exp_score_category,
                sample_num=self.sample_num, rewards_are_tensors=True))

        return float(np.mean(losses)) if losses else 0.0

    def train_arm_model(self, tt):
        self._trim_buffers()
        if len(self.reward_arm) == 0:
            return 0.0

        optimizer_arm = optim.AdamW(self.arm_func.parameters(), lr=self.arm_lr)
        return self._train_supervised(
            optimizer_arm, self.arm_func, self.context_list_arm,
            self.reward_arm, sample_num=self.sample_num)

    def _train_supervised(self, optimizer, model, contexts, rewards, sample_num,
                          rewards_are_tensors=False):
        model.train()
        length = len(rewards)
        indices = np.arange(length)
        np.random.shuffle(indices)

        total_loss = 0.0
        count = 0
        while True:
            batch_loss = 0.0
            for idx in indices:
                c = contexts[idx].to(self.device)
                if rewards_are_tensors:
                    r = rewards[idx].to(self.device)
                else:
                    r = torch.tensor([[float(rewards[idx])]],
                                     dtype=torch.float32, device=self.device)

                optimizer.zero_grad()
                loss = (model(c) - r) ** 2
                loss = loss.mean()
                loss.backward()
                optimizer.step()

                loss_val = float(loss.item())
                batch_loss += loss_val
                total_loss += loss_val
                count += 1
                if count >= sample_num:
                    model.eval()
                    return total_loss / max(count, 1)

            if batch_loss / max(length, 1) <= 1e-3:
                model.eval()
                return batch_loss / max(length, 1)


def _normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _make_category_contexts(item_contexts, item_indices, item_to_categories):
    category_to_local_items = defaultdict(list)
    for local_idx, item_idx in enumerate(item_indices):
        for c in item_to_categories.get(int(item_idx), []):
            category_to_local_items[int(c)].append(local_idx)

    categories = sorted(category_to_local_items)
    contexts = []
    for c in categories:
        local_items = np.asarray(category_to_local_items[c], dtype=np.int64)
        ctx = item_contexts[local_items]
        contexts.append(np.concatenate([ctx.mean(axis=0), ctx.std(axis=0)]))

    if not contexts:
        return categories, np.zeros((0, item_contexts.shape[1] * 2),
                                    dtype=np.float32), category_to_local_items
    return categories, _normalize_rows(np.vstack(contexts)), category_to_local_items


def _sample_from_selected_categories(selected_categories,
                                     category_to_local_items, max_candidates):
    selected_categories = list(selected_categories)
    if len(selected_categories) == 0:
        return np.asarray([], dtype=np.int64)

    union = []
    for category_id in selected_categories:
        union.extend(
            int(i) for i in category_to_local_items.get(int(category_id), []))
    union = np.asarray(sorted(set(union)), dtype=np.int64)

    if max_candidates is not None and max_candidates > 0 and len(union) > max_candidates:
        union = np.random.choice(union, size=max_candidates, replace=False)

    return np.asarray(union, dtype=np.int64)


def run_h2n_bandit(env, agent, item_to_categories, num_steps=10000,
                   max_candidates=None, hit_threshold=0.8, top_k_clusters=4,
                   category_exploration_coef=1.0,
                   artificial_exp_score=-1.0, train_threshold=1,
                   train_freq=20, allowed_users=None, track_selection=False,
                   allow_repeat=True, fixed_user=None, funnel_recorder=None,
                   candidate_provider=None):
    regrets, rewards_log, hits, selections, loss_log = [], [], [], [], []
    user_history = defaultdict(set)

    pbar = tqdm(total=num_steps, desc="Train H2N-Bandit", leave=False)
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

        base_contexts, _ = env.step_movie(candidate_items)
        categories, category_contexts, category_to_local_items = (
            _make_category_contexts(base_contexts, candidate_items,
                                    item_to_categories)
        )

        if len(categories) == 0:
            if (max_candidates is not None and max_candidates > 0
                    and len(candidate_items) > max_candidates):
                final_candidates = np.random.choice(
                    candidate_items, size=max_candidates, replace=False)
            else:
                final_candidates = np.asarray(candidate_items, dtype=np.int64)
            final_contexts, rewards_batch = env.step_movie(final_candidates)
            selected_local_categories = np.asarray([], dtype=np.int64)
            category_rwd_est = None
            category_gradients = None
            selected_category_contexts = np.zeros(
                (0, 2 * agent.input_dim), dtype=np.float32)
        else:
            selected_local_categories, category_rwd_est, category_gradients = (
                agent.select_category_neural_exp(
                    category_contexts,
                    top_k_category_num=top_k_clusters,
                    t=steps_done,
                    category_exploration_coef=category_exploration_coef,
                )
            )
            selected_categories = np.asarray([
                int(categories[int(i)]) for i in selected_local_categories
            ], dtype=np.int64)
            selected_local_items = _sample_from_selected_categories(
                selected_categories, category_to_local_items,
                max_candidates=max_candidates)

            if len(selected_local_items) == 0:
                selected_local_items = np.arange(len(candidate_items))
                if (max_candidates is not None and max_candidates > 0
                        and len(selected_local_items) > max_candidates):
                    selected_local_items = np.random.choice(
                        selected_local_items, size=max_candidates,
                        replace=False)

            final_candidates = np.asarray(
                candidate_items[selected_local_items], dtype=np.int64)
            final_contexts, rewards_batch = env.step_movie(final_candidates)
            selected_category_contexts = category_contexts[
                selected_local_categories]

        rewards_batch = np.nan_to_num(rewards_batch, nan=0.0)
        arm_idx, arm_scores = agent.select_arm(final_contexts, steps_done)
        if arm_idx is None:
            if fixed_user is not None:
                break
            continue

        best_item_idx = int(final_candidates[arm_idx])
        reward = float(rewards_batch[arm_idx])

        if not allow_repeat:
            user_history[user_id].add(best_item_idx)
        if track_selection:
            selections.append(best_item_idx)

        category_loss = 0.0
        arm_loss = 0.0
        if category_rwd_est is not None and category_gradients is not None:
            local_item_pos = {int(item): i for i, item in enumerate(candidate_items)}
            final_local_items = [
                local_item_pos[int(item)] for item in final_candidates
                if int(item) in local_item_pos
            ]

            category_rewards = []
            category_exp_scores = []
            arm_scores_arr = np.asarray(arm_scores, dtype=np.float32)
            final_local_items_arr = np.asarray(final_local_items, dtype=np.int64)
            for selected_pos, local_c in enumerate(selected_local_categories):
                true_c = int(categories[int(local_c)])
                local_members = set(category_to_local_items.get(true_c, []))
                retained_positions = [
                    i for i, local_item in enumerate(final_local_items_arr)
                    if int(local_item) in local_members
                ]
                if retained_positions:
                    best_pseudo = float(np.max(arm_scores_arr[retained_positions]))
                else:
                    best_pseudo = float(np.max(arm_scores_arr))
                category_rewards.append(best_pseudo)
                est = float(category_rwd_est[int(local_c)].item())
                category_exp_scores.append(best_pseudo - est)

            agent.update_category_info(
                selected_category_contexts, category_rewards,
                selected_local_categories, category_gradients,
                category_exp_scores,
                artificial_category_exp_val=artificial_exp_score,
            )

        agent.update_arm_info(final_contexts[arm_idx], reward)

        if steps_done < train_threshold or steps_done % train_freq == 0:
            category_loss = agent.train_category_model(steps_done)
            arm_loss = agent.train_arm_model(steps_done)
        loss_log.append(float(category_loss + arm_loss))

        opt_reward = float(np.nanmax(env.rating_matrix[user_id, positive_items]))
        regrets.append(opt_reward - reward)
        rewards_log.append(reward)
        hits.append(1 if reward >= hit_threshold else 0)

        if funnel_recorder is not None:
            selected_true_categories = [
                int(categories[int(i)]) for i in selected_local_categories
            ] if len(categories) > 0 else []
            funnel_recorder.append({
                "step": int(steps_done),
                "user_id": int(user_id),
                "selected_clusters": np.asarray(
                    selected_true_categories, dtype=np.int64),
                "selected_tags": np.asarray([], dtype=np.int64),
                "final_candidates": np.asarray(final_candidates, dtype=np.int64),
                "best_item": int(best_item_idx),
            })

        steps_done += 1
        pbar.update(1)

    pbar.close()
    return regrets, rewards_log, hits, hits, hits, selections, loss_log
