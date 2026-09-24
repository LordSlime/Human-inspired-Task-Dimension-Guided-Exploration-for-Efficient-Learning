import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
from collections import defaultdict

def check_hit_at_k(scores, rewards, k, threshold):
    if scores is None or len(scores) == 0: return 0
    k = min(k, len(scores))
    top_k_indices = np.argsort(scores)[-k:]
    top_k_rewards = rewards[top_k_indices]
    valid_rewards = top_k_rewards[~np.isnan(top_k_rewards)]
    if len(valid_rewards) > 0 and np.max(valid_rewards) >= threshold:
        return 1
    return 0

def run_standard_bandit(env, agent, num_steps=10000, max_candidates=500, hit_threshold=0.8, allowed_items=None, track_selection=False, allow_repeat=True):
    """
    Standard Bandit Runner (LinUCB, NeuralUCB, etc.)
    [Fixed] Uses while loop to guarantee exactly num_steps valid interactions.
    """
    regrets = []
    rewards_log = [] 
    hits = []
    hits_5 = []
    hits_10 = []
    selections = [] 
    
    # History tracker: user_id -> set of item_ids
    user_history = defaultdict(set)
    
    all_users = np.arange(env.num_users)
    
    # [FIX] Use manual tqdm with total=num_steps
    pbar = tqdm(total=num_steps, desc=f"Train {agent.__class__.__name__}")
    steps_done = 0
    
    while steps_done < num_steps:
        user_id = np.random.choice(all_users)
        env.reset(user_id) 
        valid_indices = env.get_user_valid_items()
        
        # 1. Warm/Cold Filter
        if allowed_items is not None:
            valid_indices = np.intersect1d(valid_indices, allowed_items)
            
        # 2. History Filter (No Repeat)
        if not allow_repeat:
            watched_history = list(user_history[user_id])
            if len(watched_history) > 0:
                valid_indices = np.setdiff1d(valid_indices, watched_history)
        
        # If no valid items, skip this user but DO NOT advance progress
        if len(valid_indices) == 0:
            continue
            
        # Candidate Generation
        if len(valid_indices) > max_candidates:
            candidates = np.random.choice(valid_indices, size=max_candidates, replace=False)
        else:
            candidates = valid_indices
            
        # Action
        context, rewards = env.step_movie(candidates)
        best_cand_idx, scores = agent.take_action(context)
        best_movie_idx = candidates[best_cand_idx]
        reward = rewards[best_cand_idx]
        
        # Update
        agent.update(context, best_cand_idx, reward)
        
        # Record History
        if not allow_repeat:
            user_history[user_id].add(best_movie_idx)
        
        if track_selection:
            selections.append(best_movie_idx)
            
        # Metrics
        all_rewards = env.rating_matrix[user_id, valid_indices]
        opt_reward = np.nanmax(all_rewards)
        regrets.append(opt_reward - reward)
        rewards_log.append(reward)
        
        is_hit = 1 if reward >= hit_threshold else 0
        hits.append(is_hit)
        hits_5.append(is_hit) 
        hits_10.append(is_hit)
        
        # [FIX] Increment and update progress bar only on success
        steps_done += 1
        pbar.update(1)
        
    pbar.close()
    return regrets, rewards_log, hits, hits_5, hits_10, selections

def run_hierarchical_bandit(env, cluster_agent, movie_agent, num_steps=10000, max_candidates=500, hit_threshold=0.8, allowed_items=None, track_selection=False, allow_repeat=True):
    """
    Hierarchical Bandit Runner (Baseline Hierarchical)
    [Fixed] Uses while loop to guarantee exactly num_steps valid interactions.
    """
    regrets = []
    rewards_log = []
    hits = []
    hits_5 = []
    hits_10 = []
    selections = []
    
    user_history = defaultdict(set)
    
    all_users = np.arange(env.num_users)
    
    # [FIX] Use manual tqdm
    pbar = tqdm(total=num_steps, desc="Train Hierarchical")
    steps_done = 0
    
    while steps_done < num_steps:
        user_id = np.random.choice(all_users)
        env.reset(user_id)
        valid_indices = env.get_user_valid_items()
        
        if allowed_items is not None:
            valid_indices = np.intersect1d(valid_indices, allowed_items)
            
        # History Filter
        if not allow_repeat:
            watched_history = list(user_history[user_id])
            if len(watched_history) > 0:
                valid_indices = np.setdiff1d(valid_indices, watched_history)
        
        if len(valid_indices) == 0: 
            continue
            
        # 1. Select Cluster
        cluster_context = env.step_cluster()
        # Predict
        c_idx, c_scores = cluster_agent.take_action(cluster_context)
        
        if isinstance(c_idx, (list, np.ndarray, tuple)):
            best_c_idx = c_idx[0]
        else:
            best_c_idx = c_idx
        
        # 2. Filter Movies in Cluster
        cluster_movies = env.cluster_movie_dict.get(best_c_idx, [])
        candidates = np.intersect1d(valid_indices, cluster_movies)
        
        if len(candidates) == 0:
            candidates = valid_indices 
            
        if len(candidates) > max_candidates:
            candidates = np.random.choice(candidates, size=max_candidates, replace=False)
            
        # 3. Select Movie
        item_context, rewards = env.step_movie(candidates)
        m_idx, m_scores = movie_agent.take_action(item_context) 
        
        if isinstance(m_idx, (list, np.ndarray, tuple)):
            best_cand_idx = m_idx[0]
        else:
            best_cand_idx = m_idx
            
        best_movie_idx = candidates[best_cand_idx]
        reward = rewards[best_cand_idx]
        
        # 4. Update
        movie_agent.update(item_context, best_cand_idx, reward)
        cluster_agent.update(cluster_context, best_c_idx, reward)
        
        # Record History
        if not allow_repeat:
            user_history[user_id].add(best_movie_idx)
            
        if track_selection:
            selections.append(best_movie_idx)
            
        # Metrics
        all_rewards = env.rating_matrix[user_id, valid_indices]
        opt_reward = np.nanmax(all_rewards)
        regrets.append(opt_reward - reward)
        rewards_log.append(reward)
        hits.append(1 if reward >= hit_threshold else 0)
        hits_5.append(1 if reward >= hit_threshold else 0)
        hits_10.append(1 if reward >= hit_threshold else 0)
        
        # [FIX] Increment
        steps_done += 1
        pbar.update(1)
        
    pbar.close()
    return regrets, rewards_log, hits, hits_5, hits_10, selections

def plot_results(results, window_size=500, save_dir="./results"):
    # Clear previous plots
    plt.close('all')
    
    num_algs = len(results)
    if num_algs == 0: return

    # Define Metrics
    # (row_idx, key, title, ylabel, is_cumulative)
    metrics_config = [
        (1, 'regrets', "Cumulative Regret", "Regret", True),
        (2, 'rewards', "Cumulative Reward", "Reward", True),
        (3, 'regrets', f"Average Regret (MA-{window_size})", "Regret", False),
        (4, 'rewards', f"Average Reward (MA-{window_size})", "Reward", False),
        (5, 'hits', f"Hit Rate (MA-{window_size})", "Hit Rate", False)
    ]
    
    plt.figure(figsize=(15, 18))
    
    for i, (plot_idx, key, title, ylabel, is_cumulative) in enumerate(metrics_config):
        plt.subplot(3, 2, plot_idx)
        
        for alg_name, data in results.items():
            if key not in data: continue
            
            # Data is a list of lists (num_runs x num_steps)
            # If alg_runner is fixed, these lists will all have the same length.
            try:
                matrix = np.array(data[key]) 
            except ValueError:
                # Fallback if lists are still jagged (e.g. mixed runners)
                min_len = min([len(x) for x in data[key]])
                matrix = np.array([x[:min_len] for x in data[key]])

            if matrix.ndim == 1: matrix = matrix.reshape(1, -1)
            
            if is_cumulative:
                # Cumulative Sum along steps
                cumsum = np.cumsum(matrix, axis=1)
                y_data = np.mean(cumsum, axis=0)
                y_std = np.std(cumsum, axis=0)
            else:
                # Moving Average
                smoothed_runs = []
                for j in range(matrix.shape[0]):
                    series = pd.Series(matrix[j])
                    smoothed = series.rolling(window=window_size, min_periods=1).mean().values
                    smoothed_runs.append(smoothed)
                y_data = np.mean(np.array(smoothed_runs), axis=0)
                y_std = np.std(np.array(smoothed_runs), axis=0)

            x_axis = np.arange(len(y_data))
            plt.plot(x_axis, y_data, label=alg_name)
            plt.fill_between(x_axis, y_data - y_std, y_data + y_std, alpha=0.15)
            
        plt.title(title)
        plt.xlabel("Steps")
        plt.ylabel(ylabel)
        plt.legend(fontsize='small')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "comparison_results.png")
    plt.savefig(save_path)
    print(f"Results plot saved to {save_path}")