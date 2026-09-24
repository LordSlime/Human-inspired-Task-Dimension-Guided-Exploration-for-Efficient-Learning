import numpy as np
import pickle
import torch
import os
import random
import heapq
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE 
from scipy.sparse import csr_matrix
from tqdm import tqdm
from collections import defaultdict
from collections import Counter 
import copy
import matplotlib.patches as patches

from env import Movielens 
from alg import LinUCB, LinUCB_cluster, LinUCB_movie, TagNeuralUCB, TagNeuralTS
from neuralucb import NeuralUCBDiag, NeuralTSDiag
from alg_runner import run_hierarchical_bandit, plot_results
from ablation_no_fd import run_lowd_hierec_no_fd
from h2n_bandit import H2N_Bandit, run_h2n_bandit
from item_cluster_bandit import build_item_space_clusters, run_item_cluster_bandit
from cofine_ucb import CoFineUCB, learn_cofine_projection, run_cofine_ucb
from datetime import datetime

# ==========================================
# Configuration & Constants (Last.fm)
# ==========================================

# Offline candidate protocol:
#   sampled_unobserved: observed items plus uniformly sampled missing items
#   observed_only:      observed-feedback items only (I_raw = I_positive)
CANDIDATE_PROTOCOL = "observed_only"
if CANDIDATE_PROTOCOL not in {"sampled_unobserved", "observed_only"}:
    raise ValueError(
        "TDGE_CANDIDATE_PROTOCOL must be 'sampled_unobserved' or "
        "'observed_only'."
    )
INCLUDE_UNINTERACTED_ITEMS = CANDIDATE_PROTOCOL == "sampled_unobserved"

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_DIR = "./datasets/lastfm_data"       
FM_RAW_DIR = "./datasets/lastfm"
result_prefix = "observed_only_results" if CANDIDATE_PROTOCOL == "observed_only" else "results"
RESULTS_DIR = f"./results_lastfm/{result_prefix}_{current_time}/npy"         
PLOT_DIR = f"./results_lastfm/{result_prefix}_{current_time}/plots"                

DATA_DIR = os.environ.get("TDGE_DATA_DIR", DATA_DIR)

if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
print(f"[Evaluation] Candidate protocol: {CANDIDATE_PROTOCOL}")

NUM_STEPS = int(os.environ.get("TDGE_NUM_STEPS", 10000))
NUM_RUNS = int(os.environ.get("TDGE_NUM_RUNS", 5))
VISUALIZE_SELECTION = False

DO_COLD_START = os.environ.get("TDGE_DO_COLD_START", "1") != "0"
TRAIN_USER_RATIO = 0.9        
COLD_START_USERS = int(os.environ.get("TDGE_COLD_START_USERS", 100))
COLD_START_STEPS = int(os.environ.get("TDGE_COLD_START_STEPS", 10))
# When enabled, SVD is fitted only on online-training users and every sampled
# cold-start user receives the same normalized mean training-user vector.
# Their held-out interactions remain available only to the evaluation
# environment for candidate construction, rewards, and regret computation.
ZERO_HISTORY_COLD_START = True
    
MAX_CANDIDATES = 10    
# Cluster recall candidates
MIN_CANDIDATES = 50
UNINTERACTED_SAMPLE_RATIO = 1.0
UNINTERACTED_SAMPLE_CAP = 100

# 🌟 Per-algorithm (K1, K2) overrides
PER_ALG_TOPK = {
    'LowD_HieRec_LinearUCB':  (2, 4),
    'LowD_HieRec_NeuralUCB':  (3, 10),
    'LowD_HieRec_NeuralTS':   (1, 24),   
    'LowD_HieRec_LinearUCB_Fusion':  (2, 4),
    'LowD_HieRec_NeuralUCB_Fusion':  (3, 10),
    'LowD_HieRec_NeuralTS_Fusion':   (1, 24),
}          

ITEM_CLUSTER_TOPK = {
    'ItemCluster_LinearUCB': 5,
    'ItemCluster_NeuralUCB': 5,
    'ItemCluster_NeuralTS': 5,
}

ITEM_SPACE_CLUSTER_FILE = f"{DATA_DIR}/item_space_cluster_features.npy"
ITEM_SPACE_CLUSTER_DICT_FILE = f"{DATA_DIR}/item_space_cluster_artist_dict.pkl"
ITEM_SPACE_ITEM_CLUSTERS_FILE = f"{DATA_DIR}/item_space_artist_clusters.csv"

# H2N-Bandit settings. LastFM has no official genre field, so H2N uses raw
# item-side artist tags as categories rather than TDGE semantic clusters.
H2N_TOP_RAW_TAGS = 50
H2N_TOP_K_CATEGORIES = 4
H2N_EXPLORE_COEF = 1.0
H2N_ARTIFICIAL_EXP = -1.0
H2N_TRAIN_THRESHOLD = 1
H2N_TRAIN_FREQ = 20
H2N_LAMBDA = 0.001
H2N_ARM_NU = 0.001
H2N_CATEGORY_LR = 1e-3
H2N_ARM_LR = 1e-3
H2N_SAMPLE_NUM = 2000
H2N_CATEGORY_PARAM_POOL_SIZE = 50
H2N_BUFFER_SIZE = 2000
H2N_HIDDEN_DIM = 200
H2N_MAX_FILTERED_ARMS = MAX_CANDIDATES

LINUCB_BETA = 0.01
LINUCB_LAMBDA = 1.0

# CoFineUCB settings.  This linear structured contextual bandit learns a
# coarse feature subspace from train-user profiles only, then scores the same
# item-level candidate pool as the flat and TDGE baselines.
COFINE_SUBSPACE_DIM = 5
COFINE_LAMBDA = LINUCB_LAMBDA
COFINE_LAMBDA_TILDE = LINUCB_LAMBDA
COFINE_ALPHA_FINE = LINUCB_BETA
COFINE_ALPHA_COARSE = LINUCB_BETA
COFINE_PROFILE_LAMBDA = LINUCB_LAMBDA
COFINE_MAX_PROFILE_USERS = None
COFINE_MIN_PROFILE_INTERACTIONS = 3
COFINE_USE_LEARNU_SCALING = True

HIT_THRESHOLD = 0.8      

ALLOW_REPEAT_RECOMMENDATION = True      
ALLOW_REPEAT_RECOMMENDATION_CS = True   

# Feature Dimensions
BASELINE_DIM = 101
SEMANTIC_DIM = 101
RANDOM_SEED = 2026        

ALGORITHMS_TO_RUN = [
    "LinUCB",
    "NeuralUCB",
    "NeuralTS",
    "LowD_HieRec_LinearUCB",
    "LowD_HieRec_NeuralUCB",
    "LowD_HieRec_NeuralTS",
    "H2N_Bandit",
    "CoFineUCB",
]

_ALG_OVERRIDE = os.environ.get("TDGE_ALGORITHMS")
if _ALG_OVERRIDE:
    ALGORITHMS_TO_RUN = [
        alg.strip() for alg in _ALG_OVERRIDE.split(",") if alg.strip()
    ]

# Number of tags sampled uniformly each round for the NoFD ablation
NUM_TAG_ARMS = 50


HIDDEN_DIM = 128

      
CS_CONVERGENCE_MIN_TAIL = 3           
CS_CONVERGENCE_MIN_PREFIX_DIVERSITY = 2  

RECORD_ALL_CS_TRAJECTORIES = True

LOWD_TO_BASELINE = {
    "LowD_HieRec_LinearUCB": "LinUCB",
    "LowD_HieRec_NeuralUCB": "NeuralUCB",
    "LowD_HieRec_NeuralTS":  "NeuralTS",
}

LOWD_FUNNEL_ALGS = set(LOWD_TO_BASELINE) | {
    "LowD_HieRec_LinearUCB_Fusion",
    "LowD_HieRec_NeuralUCB_Fusion",
    "LowD_HieRec_NeuralTS_Fusion",
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample_unobserved_items(user_ratings, sample_size):
    if sample_size <= 0:
        return np.asarray([], dtype=np.int64)

    num_items = user_ratings.shape[0]
    sampled = []
    sampled_set = set()
    max_draws = max(sample_size * 20, 100)
    draws_done = 0

    while len(sampled) < sample_size and draws_done < max_draws:
        needed = sample_size - len(sampled)
        draws = np.random.randint(0, num_items, size=needed * 2)
        draws_done += len(draws)
        for item_idx in draws:
            item_idx = int(item_idx)
            if item_idx not in sampled_set and np.isnan(user_ratings[item_idx]):
                sampled.append(item_idx)
                sampled_set.add(item_idx)
                if len(sampled) == sample_size:
                    break

    if len(sampled) < sample_size:
        unobserved_items = np.flatnonzero(np.isnan(user_ratings)).astype(np.int64)
        remaining_items = np.asarray(
            [item for item in unobserved_items if int(item) not in sampled_set],
            dtype=np.int64,
        )
        fill_count = min(sample_size - len(sampled), len(remaining_items))
        if fill_count > 0:
            fill_items = np.random.choice(
                remaining_items, size=fill_count, replace=False)
            sampled.extend(fill_items.astype(np.int64).tolist())

    return np.asarray(sampled, dtype=np.int64)


def get_candidate_and_positive_items(env, user_history, user_id, allow_repeat):
    user_ratings = env.rating_matrix[user_id]
    positive_items = np.flatnonzero(~np.isnan(user_ratings)).astype(np.int64)
    candidate_items = positive_items

    if INCLUDE_UNINTERACTED_ITEMS and len(positive_items) > 0:
        sample_size = min(
            env.rating_matrix.shape[1] - len(positive_items),
            int(np.ceil(len(positive_items) * UNINTERACTED_SAMPLE_RATIO)),
            UNINTERACTED_SAMPLE_CAP,
        )
        if sample_size > 0:
            sampled_unobserved = sample_unobserved_items(
                user_ratings, sample_size)
            candidate_items = np.concatenate([positive_items, sampled_unobserved])

    if not allow_repeat:
        watched = list(user_history[user_id])
        if len(watched) > 0:
            candidate_items = np.setdiff1d(candidate_items, watched)
            positive_items = np.setdiff1d(positive_items, watched)

    return np.asarray(candidate_items, dtype=np.int64), np.asarray(positive_items, dtype=np.int64)

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def compute_train_mean_user_vector(user_features, train_users):
    """Return a unit-normalized mean vector computed only from train users."""
    train_users = np.asarray(train_users, dtype=np.int64)
    if train_users.size == 0:
        raise ValueError("Cannot compute a cold-start context without train users.")

    mean_user = np.mean(user_features[train_users], axis=0)
    norm = np.linalg.norm(mean_user)
    if norm <= 1e-12:
        raise ValueError("The mean training-user vector has near-zero norm.")
    return mean_user / norm


def fit_train_only_user_svd(rating_matrix, train_users, n_components):
    """Fit normalized user embeddings without using held-out user rows."""
    train_users = np.asarray(train_users, dtype=np.int64)
    if train_users.size == 0:
        raise ValueError("Cannot fit user SVD without train users.")

    sparse_rows = []
    sparse_cols = []
    sparse_values = []
    for local_row, user_idx in enumerate(train_users):
        user_ratings = rating_matrix[int(user_idx)]
        observed_cols = np.flatnonzero(~np.isnan(user_ratings))
        if observed_cols.size == 0:
            continue
        sparse_rows.append(
            np.full(observed_cols.size, local_row, dtype=np.int64))
        sparse_cols.append(observed_cols.astype(np.int64, copy=False))
        sparse_values.append(
            user_ratings[observed_cols].astype(np.float32, copy=False))

    if not sparse_values:
        raise ValueError("The training-user interaction matrix is empty.")

    train_interactions = csr_matrix(
        (
            np.concatenate(sparse_values),
            (np.concatenate(sparse_rows), np.concatenate(sparse_cols)),
        ),
        shape=(len(train_users), rating_matrix.shape[1]),
    )
    svd = TruncatedSVD(
        n_components=n_components, random_state=RANDOM_SEED)
    train_embeddings = svd.fit_transform(train_interactions)
    train_norms = np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    train_embeddings = train_embeddings / (train_norms + 1e-12)

    user_features = np.zeros(
        (rating_matrix.shape[0], n_components), dtype=np.float32)
    user_features[train_users] = train_embeddings.astype(
        np.float32, copy=False)
    return user_features, svd

set_seed(RANDOM_SEED)


def load_all_data():
    print("Loading Last.fm datasets for all algorithms...")
    user_features = np.load(f"{DATA_DIR}/user_features.npy")
    rating_matrix = np.load(f"{DATA_DIR}/rating_matrix.npy")
    artist_ids = np.load(f"{DATA_DIR}/artist_ids.npy")
    
    artist_features_base = np.load(f"{DATA_DIR}/artist_features.npy")
    cluster_features_base = np.load(f"{DATA_DIR}/cluster_features.npy")
    with open(f"{DATA_DIR}/cluster_artist_dict.pkl", "rb") as f:
        cluster_dict_base = pickle.load(f)
        
    semantic_artist_features = np.load(f"{DATA_DIR}/semantic_artist_features.npy")
    semantic_cluster_features = np.load(f"{DATA_DIR}/semantic_cluster_features.npy") 
    semantic_cluster_l1_features = np.load(f"{DATA_DIR}/semantic_cluster_l1_features.npy") 
    
    l0_norms = np.linalg.norm(semantic_cluster_features, axis=1, keepdims=True)
    semantic_cluster_features = semantic_cluster_features / (l0_norms + 1e-9)
    
    l1_norms = np.linalg.norm(semantic_cluster_l1_features, axis=1, keepdims=True)
    semantic_cluster_l1_features = semantic_cluster_l1_features / (l1_norms + 1e-9)
    
    with open(f"{DATA_DIR}/semantic_cluster_artist_dict.pkl", "rb") as f:
        semantic_cluster_dict = pickle.load(f)
    
    tag_features_matrix = np.load(f"{DATA_DIR}/semantic_tag_embeddings.npy")

    print("Loading Multi-level Cluster Mappings...")
    df_clusters = pd.read_csv(f"{DATA_DIR}/artist_multi_level_clusters.csv")
    artist_id_map = {aid: i for i, aid in enumerate(artist_ids)}
    
    cluster_to_tags = defaultdict(set) 
    tag_to_artists = defaultdict(set)
    artist_to_tags = defaultdict(set)
    artist_to_clusters = defaultdict(set) 
    l1_to_l0 = defaultdict(set)           
    artist_tag_relevance = defaultdict(dict)
    has_relevance = 'relevance' in df_clusters.columns

    artist_l1_scores = defaultdict(lambda: defaultdict(float))
    artist_l0_scores = defaultdict(lambda: defaultdict(float))

    for _, row in df_clusters.iterrows():
        aid = int(row['artistId'])
        if aid not in artist_id_map: continue
        
        a_idx = artist_id_map[aid]
        l1_idx = int(row['lvl1_cluster'])
        c_idx = int(row['lvl0_cluster']) 
        t_idx = int(row['tag_index'])
        rel_score = float(row['relevance']) if has_relevance else 1.0
        
        cluster_to_tags[c_idx].add(t_idx)
        tag_to_artists[t_idx].add(a_idx)
        artist_to_tags[a_idx].add(t_idx)
        artist_to_clusters[a_idx].add(c_idx)
        l1_to_l0[l1_idx].add(c_idx)
        artist_tag_relevance[a_idx][t_idx] = rel_score

        artist_l1_scores[a_idx][l1_idx] += rel_score
        artist_l0_scores[a_idx][c_idx] += rel_score  

    artist_primary_l1 = {}
    for a_idx, scores in artist_l1_scores.items():
        artist_primary_l1[a_idx] = max(scores.items(), key=lambda x: x[1])[0]

    artist_primary_l0 = {}
    for a_idx, scores in artist_l0_scores.items():
        artist_primary_l0[a_idx] = max(scores.items(), key=lambda x: x[1])[0]

    return (user_features, rating_matrix, artist_ids,
            artist_features_base, cluster_features_base, cluster_dict_base,
            semantic_artist_features, semantic_cluster_features, semantic_cluster_l1_features,
            semantic_cluster_dict, tag_features_matrix, 
            cluster_to_tags, tag_to_artists, artist_to_tags, artist_to_clusters,
            l1_to_l0, artist_tag_relevance, artist_primary_l1, artist_primary_l0)   


def load_or_build_item_space_clusters(item_ids, semantic_item_features):
    if (os.path.exists(ITEM_SPACE_CLUSTER_FILE)
            and os.path.exists(ITEM_SPACE_CLUSTER_DICT_FILE)
            and os.path.exists(ITEM_SPACE_ITEM_CLUSTERS_FILE)):
        item_cluster_features = np.load(ITEM_SPACE_CLUSTER_FILE)
        with open(ITEM_SPACE_CLUSTER_DICT_FILE, "rb") as f:
            item_cluster_dict = pickle.load(f)
        df_item_clusters = pd.read_csv(ITEM_SPACE_ITEM_CLUSTERS_FILE)
        item_to_clusters = {
            int(row["item_index"]): [int(row["item_cluster"])]
            for _, row in df_item_clusters.iterrows()
        }
        item_cluster_names = [
            f"item_cluster_{i}" for i in range(item_cluster_features.shape[0])
        ]
        item_cluster_dict = {
            int(k): np.asarray(v, dtype=np.int64)
            for k, v in item_cluster_dict.items()
        }
        print(f"[ItemCluster] Loaded {len(item_cluster_names)} item-space clusters.")
        return item_cluster_names, item_cluster_features, item_cluster_dict, item_to_clusters

    print("[ItemCluster] Item-space cluster files not found; building from semantic item features.")
    return build_item_space_clusters(
        semantic_item_features,
        num_clusters=None,
        random_state=RANDOM_SEED,
    )


def build_lastfm_raw_tag_categories(artist_ids, top_k_tags=50):
    tag_path = os.path.join(FM_RAW_DIR, "tags.dat")
    tagged_path = os.path.join(FM_RAW_DIR, "user_taggedartists.dat")
    tags_df = pd.read_csv(
        tag_path, sep="\t", encoding="latin-1", usecols=["tagID", "tagValue"])
    tagged_df = pd.read_csv(tagged_path, sep="\t")

    artist_id_to_env = {int(aid): i for i, aid in enumerate(artist_ids)}
    tagged_df = tagged_df[tagged_df["artistID"].isin(artist_id_to_env)].copy()

    top_tag_ids = tagged_df["tagID"].value_counts().head(top_k_tags).index.tolist()
    top_tag_set = set(int(tid) for tid in top_tag_ids)
    tag_id_to_name = dict(zip(tags_df["tagID"], tags_df["tagValue"]))
    tag_to_category = {int(tid): i for i, tid in enumerate(top_tag_ids)}

    category_item_dict = {i: [] for i in range(len(top_tag_ids))}
    item_to_categories = defaultdict(list)

    artist_tag_counts = (
        tagged_df[tagged_df["tagID"].isin(top_tag_set)]
        .groupby(["artistID", "tagID"])
        .size()
        .reset_index(name="count")
    )
    for _, row in artist_tag_counts.iterrows():
        env_idx = artist_id_to_env.get(int(row["artistID"]))
        if env_idx is None:
            continue
        category_id = tag_to_category[int(row["tagID"])]
        item_to_categories[int(env_idx)].append(category_id)
        category_item_dict[category_id].append(int(env_idx))

    item_to_categories = {
        int(item): sorted(set(categories))
        for item, categories in item_to_categories.items()
    }
    category_item_dict = {
        int(category): np.asarray(sorted(set(items)), dtype=np.int64)
        for category, items in category_item_dict.items()
    }
    category_names = [
        str(tag_id_to_name.get(int(tid), tid)) for tid in top_tag_ids
    ]
    sizes = [len(category_item_dict[i]) for i in range(len(category_names))]
    print(f"[H2N] Loaded {len(category_names)} raw LastFM tag categories.")
    print(f"[H2N] Top raw tag categories: {category_names[:10]}")
    print(f"[H2N] Category sizes (artists per tag): {sizes}")
    print(f"[H2N] Artists with at least one H2N raw-tag category: "
          f"{len(item_to_categories)}/{len(artist_ids)}")
    return category_names, category_item_dict, item_to_categories


def run_standard_bandit(env, agent, num_steps=10000, max_candidates=200, allowed_users=None, track_selection=False, allow_repeat=True, fixed_user=None):
    regrets, rewards_log, hits, hits_5, hits_10, selections, loss_log = [], [], [], [], [], [], []
    user_history = defaultdict(set)
    pbar = tqdm(total=num_steps, desc="Train Standard", leave=False)
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
            
        user_id = env.current_user
        candidate_items, positive_items = get_candidate_and_positive_items(
            env, user_history, user_id, allow_repeat)

        if len(positive_items) < 1 or len(candidate_items) < 1:
            if fixed_user is not None: break
            continue
        
        candidates = np.random.choice(candidate_items, size=max_candidates, replace=False) if len(candidate_items) > max_candidates else candidate_items

        cand_feats = env.movie_features[candidates]
        N = cand_feats.shape[0]
        ctx_raw = np.concatenate([np.tile(user_emb, (N, 1)), cand_feats], axis=1)
        ctx = np.concatenate([ctx_raw, np.ones((N, 1)) * 0.01], axis=1)
        ctx = ctx / (np.linalg.norm(ctx, axis=1, keepdims=True) + 1e-9)
        
        best_cand_idx, _ = agent.take_action(ctx)
        best_item_idx = candidates[best_cand_idx]
        
        _, rewards_batch = env.step_movie(candidates)
        rewards_batch = np.nan_to_num(rewards_batch, nan=0.0)
        reward = rewards_batch[best_cand_idx]
        
        if track_selection: selections.append(best_item_idx)
        if not allow_repeat: user_history[user_id].add(best_item_idx)
        
        loss = agent.update(ctx, best_cand_idx, reward)
        loss_log.append(loss if loss is not None else 0.0)
        
        opt_reward = np.nanmax(env.rating_matrix[user_id, positive_items])
        regrets.append(opt_reward - reward)
        rewards_log.append(reward)
        hits.append(1 if reward >= HIT_THRESHOLD else 0)
        
        steps_done += 1
        pbar.update(1)
    pbar.close()
    return regrets, rewards_log, hits, hits, hits, selections, loss_log


def build_tag_scored_candidates(candidate_items, selected_tag_scores,
                                tag_to_items, item_tag_relevance,
                                max_candidates):
    """Rank items by selected tag-score relevance and randomly fill."""
    candidate_items = np.asarray(candidate_items, dtype=np.int64)
    if len(candidate_items) == 0:
        return candidate_items

    target_size = min(max_candidates, len(candidate_items))
    candidate_set = set(candidate_items.tolist())

    item_scores = defaultdict(float)
    for tag_idx, tag_score in selected_tag_scores.items():
        for item_idx in tag_to_items.get(int(tag_idx), set()):
            item_idx = int(item_idx)
            if item_idx in candidate_set:
                relevance = item_tag_relevance.get(item_idx, {}).get(
                    int(tag_idx), 1.0)
                item_scores[item_idx] += float(tag_score) * float(relevance)

    if item_scores:
        ranked_items = heapq.nlargest(
            target_size,
            item_scores,
            key=lambda item_idx: (item_scores[item_idx], item_idx),
        )
        return np.asarray(ranked_items, dtype=np.int64)

    fill = np.random.choice(candidate_items, size=target_size, replace=False)
    return fill.astype(np.int64)


def build_fusion_candidates(candidate_items, selected_tag_indices, tag_features,
                            item_features, max_candidates):
    candidate_items = np.asarray(candidate_items, dtype=np.int64)
    if len(candidate_items) == 0:
        return candidate_items

    target_size = min(max_candidates, len(candidate_items))
    if len(selected_tag_indices) > 0:
        selected_tag_feats = tag_features[np.asarray(selected_tag_indices, dtype=np.int64)]
        intent_vector = np.mean(selected_tag_feats, axis=0)
        intent_vector = intent_vector / (np.linalg.norm(intent_vector) + 1e-9)

        candidate_feats = item_features[candidate_items]
        sim_scores = np.dot(candidate_feats, intent_vector)
        top_idx = np.argsort(sim_scores)[-target_size:]
        return candidate_items[top_idx].astype(np.int64)

    fill = np.random.choice(candidate_items, size=target_size, replace=False)
    return fill.astype(np.int64)


def run_lowd_hierec(env, cluster_agent, tag_agent, artist_agent, semantic_cluster_features, tag_features, cluster_to_tags, tag_to_artists, artist_to_tags, artist_to_clusters, artist_tag_relevance, num_steps=10000, allowed_users=None, track_selection=False, allow_repeat=True, fixed_user=None, funnel_recorder=None, top_k_clusters=None, top_k_tags=None, item_filter_mode="score"):
    # Per-algorithm override: fall back to module globals if None
    _top_k_clusters = top_k_clusters
    _top_k_tags =  top_k_tags
    regrets, rewards_log, hits, selections, loss_log = [], [], [], [], []
    user_history = defaultdict(set)
    num_clusters = semantic_cluster_features.shape[0]
    
    def make_context(user_vec, item_feats):
        N = item_feats.shape[0]
        ctx_raw = np.concatenate([np.tile(user_vec, (N, 1)), item_feats], axis=1)
        ctx = np.concatenate([ctx_raw, np.ones((N, 1)) * 0.01], axis=1)
        return ctx / (np.linalg.norm(ctx, axis=1, keepdims=True) + 1e-9)

    filter_label = "Feature-Fusion" if item_filter_mode == "fusion" else "Feature-Score"
    pbar = tqdm(total=num_steps, desc=f"Train LowD_HieRec ({filter_label} Retrieval)", leave=False)
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
            
        user_id = env.current_user
        candidate_items, positive_items = get_candidate_and_positive_items(
            env, user_history, user_id, allow_repeat)

        if len(positive_items) < 1 or len(candidate_items) < 1:
            if fixed_user is not None: break
            continue

        cluster_contexts = make_context(user_emb, semantic_cluster_features)
        _, cluster_scores = cluster_agent.take_action(cluster_contexts)
        k_c = min(_top_k_clusters, num_clusters)
        selected_cluster_indices = np.argsort(cluster_scores)[-k_c:]
        
        candidate_tags = []
        tag_source_cluster = {} 
        for c_idx in selected_cluster_indices:
            tags_in_c = list(cluster_to_tags[c_idx])
            candidate_tags.extend(tags_in_c)
            for tag in tags_in_c: tag_source_cluster[tag] = c_idx

        candidate_tags = np.unique(candidate_tags)
        selected_tag_indices = []
        selected_tag_scores = {}
        if len(candidate_tags) > 0:
            tag_contexts = make_context(user_emb, tag_features[candidate_tags])
            _, tag_scores_subset = tag_agent.take_action(tag_contexts)
            k_t = min(_top_k_tags, len(candidate_tags)) 
            selected_positions = np.argsort(tag_scores_subset)[-k_t:]
            selected_tag_indices = candidate_tags[selected_positions]
            selected_tag_scores = {
                int(candidate_tags[pos]): float(tag_scores_subset[pos])
                for pos in selected_positions
            }

        is_fallback = len(selected_tag_indices) == 0
        if item_filter_mode == "fusion":
            final_candidates = build_fusion_candidates(
                candidate_items, selected_tag_indices, tag_features,
                env.movie_features, MAX_CANDIDATES)
        else:
            final_candidates = build_tag_scored_candidates(
                candidate_items, selected_tag_scores, tag_to_artists,
                artist_tag_relevance, MAX_CANDIDATES)
            
        final_candidates = np.array(final_candidates)
        
        cand_artist_feats = env.movie_features[final_candidates] 
        cand_contexts = make_context(user_emb, cand_artist_feats)
        
        best_cand_idx, _ = artist_agent.take_action(cand_contexts)
        best_artist_idx = final_candidates[best_cand_idx]
        
        _, rewards_batch = env.step_movie(final_candidates)
        rewards_batch = np.nan_to_num(rewards_batch, nan=0.0)
        reward = rewards_batch[best_cand_idx]
        
        if track_selection: selections.append(best_artist_idx)
        if not allow_repeat: user_history[user_id].add(best_artist_idx)
        
        loss_val = artist_agent.update(cand_contexts, best_cand_idx, reward, sample_weight=1.0)
        loss_log.append(loss_val if loss_val is not None else 0.0)
        
        if not is_fallback:
            real_artist_tags = set(artist_to_tags.get(best_artist_idx, set()))
            current_artist_relevances = artist_tag_relevance.get(best_artist_idx, {})
            
            active_tags = list(real_artist_tags.intersection(set(selected_tag_indices)))
            active_clusters = set()

            for t_idx in active_tags:
                t_ctx = make_context(user_emb, tag_features[t_idx].reshape(1, -1))
                rel_weight = current_artist_relevances.get(t_idx, 1.0)
                tag_agent.update(t_ctx, 0, reward, sample_weight=rel_weight)
                if t_idx in tag_source_cluster: active_clusters.add(tag_source_cluster[t_idx])

            for c_idx in active_clusters:
                c_ctx = make_context(user_emb, semantic_cluster_features[c_idx].reshape(1, -1))
                contributing_tags = cluster_to_tags[c_idx].intersection(set(active_tags))
                cluster_weight = np.max([current_artist_relevances.get(t, 1.0) for t in contributing_tags]) if len(contributing_tags) > 0 else 1.0
                cluster_agent.update(c_ctx, 0, reward, sample_weight=cluster_weight)

        opt_reward = np.nanmax(env.rating_matrix[user_id, positive_items])
        regrets.append(opt_reward - reward)
        rewards_log.append(reward)
        hits.append(1 if reward >= HIT_THRESHOLD else 0)

        # 🌟 action-space-reduction 演示数据: 只在传入 funnel_recorder 时落盘
        if funnel_recorder is not None:
            funnel_recorder.append({
                'step':               steps_done,
                'user_id':            int(user_id),
                'selected_clusters':  np.asarray(selected_cluster_indices, dtype=np.int64),
                'selected_tags':      np.asarray(selected_tag_indices,     dtype=np.int64),
                'final_candidates':   np.asarray(final_candidates,         dtype=np.int64),
                'best_item':          int(best_artist_idx),
            })

        steps_done += 1
        pbar.update(1)
        
    pbar.close()
    return regrets, rewards_log, hits, hits, hits, selections, loss_log


def evaluate_cold_start(agent_tuple, run_func, cs_users, steps, track_users=None, funnel_track_users=None):
    cs_regrets, cs_rewards, cs_hits = [], [], []
    tracked_selections = {}
    all_selections = {}    
    funnel_data = {}       
    base_states = copy.deepcopy(agent_tuple)
    
    for uid in tqdm(cs_users, desc="Cold-Start Users", leave=False):
        set_seed(RANDOM_SEED + uid)
        eval_agents = copy.deepcopy(base_states)

        
        need_funnel = funnel_track_users is not None and uid in funnel_track_users
        recorder = [] if need_funnel else None

        reg, rew, hits, selections = run_func(eval_agents, uid, steps, recorder)
        
        while len(reg) < steps:
            reg.append(reg[-1] if len(reg) > 0 else 0)
            rew.append(rew[-1] if len(rew) > 0 else 0)
            hits.append(0)
            
        cs_regrets.append(reg[:steps])
        cs_rewards.append(rew[:steps])
        cs_hits.append(hits[:steps])

        all_selections[int(uid)] = list(selections[:steps])

        if track_users is not None and uid in track_users:
            tracked_selections[uid] = selections[:steps]

        if recorder is not None:
            funnel_data[int(uid)] = recorder
        
    return np.mean(cs_regrets, axis=0), np.mean(cs_rewards, axis=0), np.mean(cs_hits, axis=0), tracked_selections, all_selections, funnel_data


def filter_convergence_trajectories(all_selections, artist_primary_l0,
                                    min_tail=3, min_prefix_diversity=2):
    filtered = {}
    for uid, artist_seq in all_selections.items():
        artist_seq = list(artist_seq)
        if len(artist_seq) == 0:
            continue

        l0_seq = []
        ok = True
        for a in artist_seq:
            a_int = int(a)
            if a_int not in artist_primary_l0:
                ok = False
                break
            l0_seq.append(int(artist_primary_l0[a_int]))
        if not ok or len(l0_seq) < min_tail + 1:
            continue

        tail_cluster = l0_seq[-1]
        tail_length = 1
        for i in range(len(l0_seq) - 2, -1, -1):
            if l0_seq[i] == tail_cluster:
                tail_length += 1
            else:
                break

        if tail_length < min_tail:
            continue

        prefix = l0_seq[:len(l0_seq) - tail_length]
        prefix_diversity = len(set(prefix))
        if prefix_diversity < min_prefix_diversity:
            continue

        filtered[int(uid)] = {
            'l0_sequence':      np.array(l0_seq, dtype=np.int64),
            'artist_sequence':  np.array(artist_seq, dtype=np.int64),
            'tail_length':      int(tail_length),
            'prefix_diversity': int(prefix_diversity),
            'focused_cluster':  int(tail_cluster),
        }

    return filtered


def save_convergence_trajectories(filtered, alg_name, results_dir):
    if len(filtered) == 0:
        np.save(os.path.join(results_dir, f"cs_convergence_users_{alg_name}.npy"),
                np.array([], dtype=np.int64))
        np.save(os.path.join(results_dir, f"cs_convergence_meta_{alg_name}.npy"),
                np.zeros((0, 3), dtype=np.int64))
        print(f"  [{alg_name}] No convergence trajectories found; saved empty arrays.")
        return

    filtered_uids = np.array(sorted(filtered.keys()), dtype=np.int64)
    np.save(os.path.join(results_dir, f"cs_convergence_users_{alg_name}.npy"), filtered_uids)

    meta = np.array([
        [filtered[int(u)]['tail_length'],
         filtered[int(u)]['prefix_diversity'],
         filtered[int(u)]['focused_cluster']]
        for u in filtered_uids
    ], dtype=np.int64)
    np.save(os.path.join(results_dir, f"cs_convergence_meta_{alg_name}.npy"), meta)

    for uid in filtered_uids:
        info = filtered[int(uid)]
        np.save(os.path.join(results_dir, f"cs_l0_traj_{alg_name}_user_{int(uid)}.npy"),
                info['l0_sequence'])
        np.save(os.path.join(results_dir, f"cs_artist_traj_{alg_name}_user_{int(uid)}.npy"),
                info['artist_sequence'])


def compute_trajectory_metadata(all_selections, artist_primary_l0, uids_subset):
    result = {}
    for uid in uids_subset:
        uid = int(uid)
        if uid not in all_selections:
            continue
        artist_seq = list(all_selections[uid])
        if len(artist_seq) == 0:
            continue

        l0_seq = []
        ok = True
        for a in artist_seq:
            a_int = int(a)
            if a_int not in artist_primary_l0:
                ok = False
                break
            l0_seq.append(int(artist_primary_l0[a_int]))
        if not ok:
            continue

        tail_cluster = l0_seq[-1]
        tail_length = 1
        for i in range(len(l0_seq) - 2, -1, -1):
            if l0_seq[i] == tail_cluster:
                tail_length += 1
            else:
                break
        prefix = l0_seq[:len(l0_seq) - tail_length]
        prefix_diversity = len(set(prefix))

        result[uid] = {
            'l0_sequence':      np.array(l0_seq, dtype=np.int64),
            'artist_sequence':  np.array(artist_seq, dtype=np.int64),
            'tail_length':      int(tail_length),
            'prefix_diversity': int(prefix_diversity),
            'focused_cluster':  int(tail_cluster),
        }

    return result

def wrap_cs_returns(res):
    return res[0], res[1], res[2], res[5]

def main():
    data = load_all_data()
    (user_features, rating_matrix, artist_ids, 
     artist_features_base, cluster_features_base, cluster_dict_base,
     semantic_artist_features, semantic_cluster_features, semantic_cluster_l1_features,
     semantic_cluster_dict, tag_features_matrix, 
     cluster_to_tags, tag_to_artists, artist_to_tags, artist_to_clusters,
     l1_to_l0, artist_tag_relevance, artist_primary_l1, artist_primary_l0) = data 
    
    env_semantic = Movielens(user_features, semantic_artist_features, rating_matrix, artist_ids, semantic_cluster_features, semantic_cluster_dict)

    print("\n[ItemCluster] Loading item-space clusters ...")
    (item_cluster_names, item_cluster_centroids, item_cluster_item_dict,
     item_to_item_clusters) = load_or_build_item_space_clusters(
        artist_ids, semantic_artist_features)
    print(f"[ItemCluster] Done — {len(item_cluster_names)} item-space clusters.\n")

    print("\n[H2N] Loading raw LastFM tag categories ...")
    (h2n_category_names, h2n_category_item_dict,
     item_to_h2n_categories) = build_lastfm_raw_tag_categories(
        artist_ids, top_k_tags=H2N_TOP_RAW_TAGS)
    print(f"[H2N] Done — {len(h2n_category_names)} raw tag categories.\n")
    
    set_seed(RANDOM_SEED)
    
    num_users = rating_matrix.shape[0]
    shuffled_users = np.random.permutation(num_users)
    num_train_users = int(num_users * TRAIN_USER_RATIO)
    
    train_users = shuffled_users[:num_train_users]
    test_users = shuffled_users[num_train_users:]
    print(f"\n[Dataset Split] Total Users: {num_users} -> Train: {len(train_users)} | Test (Cold-Start OOV): {len(test_users)}")
    
    cs_users = np.random.choice(test_users, COLD_START_USERS, replace=False)
    track_cs_users = cs_users[:10]
    print(f"Tracking 2D Trajectories for users: {track_cs_users}")

    if ZERO_HISTORY_COLD_START:
        user_features, train_user_svd = fit_train_only_user_svd(
            rating_matrix,
            train_users,
            n_components=user_features.shape[1],
        )
        train_mean_user = compute_train_mean_user_vector(
            user_features, train_users)
        user_features[cs_users] = train_mean_user
        env_semantic.user_features = np.array(user_features, copy=True)

        np.save(
            os.path.join(RESULTS_DIR, "zero_history_train_mean_user.npy"),
            train_mean_user,
        )
        np.save(
            os.path.join(RESULTS_DIR, "zero_history_train_svd_components.npy"),
            train_user_svd.components_,
        )
        np.save(
            os.path.join(RESULTS_DIR, "zero_history_train_users.npy"),
            train_users,
        )
        np.save(
            os.path.join(RESULTS_DIR, "zero_history_cold_start_users.npy"),
            cs_users,
        )
        print(
            "[Cold-Start Context] Zero-history mode enabled: "
            f"SVD was fitted on {len(train_users)} online-training users only; "
            f"{len(cs_users)} held-out users share their normalized mean vector."
        )
    else:
        env_semantic.user_features = np.array(user_features, copy=True)
        print(
            "[Cold-Start Context] Precomputed per-user representations enabled."
        )

    cofine_projection = None
    if "CoFineUCB" in ALGORITHMS_TO_RUN:
        print("\n[CoFineUCB] Learning coarse feature subspace from train users ...")
        cofine_projection = learn_cofine_projection(
            user_features=user_features,
            item_features=semantic_artist_features,
            rating_matrix=rating_matrix,
            train_users=train_users,
            k=COFINE_SUBSPACE_DIM,
            profile_lamb=COFINE_PROFILE_LAMBDA,
            max_profile_users=COFINE_MAX_PROFILE_USERS,
            min_interactions=COFINE_MIN_PROFILE_INTERACTIONS,
            use_learnu_scaling=COFINE_USE_LEARNU_SCALING,
            random_state=RANDOM_SEED,
        )
        print(f"[CoFineUCB] Done — U shape={cofine_projection.shape}.\n")
    
    RESULTS = {}
    RESULTS_CS = {}
    TRACKED_TRAJECTORIES = {}         
    ALL_CS_TRAJECTORIES = {}          
    FUNNEL_DATA = {}
    
    for alg_name in ALGORITHMS_TO_RUN:
        print(f"\n--- Running {alg_name} ({NUM_RUNS} runs) ---")
        
        regrets_all, rewards_all, hits_all, hits_5_all, hits_10_all, selections_all, losses_all = [], [], [], [], [], [], []
        cs_regrets_all, cs_rewards_all, cs_hits_all = [], [], []
        
        for i in range(NUM_RUNS):
            print(f" > Run {i+1}/{NUM_RUNS}...")
            set_seed(RANDOM_SEED + i)

            if alg_name == "LinUCB":
                agent = LinUCB(d=BASELINE_DIM, beta=LINUCB_BETA, lamb=LINUCB_LAMBDA)
                agents_tuple = (agent,)
                run_main = lambda: run_standard_bandit(env_semantic, agent, num_steps=NUM_STEPS, max_candidates=MAX_CANDIDATES, allowed_users=train_users, track_selection=VISUALIZE_SELECTION, allow_repeat=ALLOW_REPEAT_RECOMMENDATION)
                run_cs = lambda eval_ag, uid, steps, recorder=None: wrap_cs_returns(run_standard_bandit(env_semantic, eval_ag[0], num_steps=steps, max_candidates=MAX_CANDIDATES, fixed_user=uid, track_selection=True, allow_repeat=ALLOW_REPEAT_RECOMMENDATION_CS))

            elif alg_name == "NeuralUCB":
                agent = NeuralUCBDiag(d=BASELINE_DIM, num_arm=MAX_CANDIDATES, beta=0.01, lamb=0.01, hidden=HIDDEN_DIM, buffer_size=2000)
                agents_tuple = (agent,)
                run_main = lambda: run_standard_bandit(env_semantic, agent, num_steps=NUM_STEPS, max_candidates=MAX_CANDIDATES, allowed_users=train_users, track_selection=VISUALIZE_SELECTION, allow_repeat=ALLOW_REPEAT_RECOMMENDATION)
                run_cs = lambda eval_ag, uid, steps, recorder=None: wrap_cs_returns(run_standard_bandit(env_semantic, eval_ag[0], num_steps=steps, max_candidates=MAX_CANDIDATES, fixed_user=uid, track_selection=True, allow_repeat=ALLOW_REPEAT_RECOMMENDATION_CS))
            
            elif alg_name == "NeuralTS":
                agent = NeuralTSDiag(d=BASELINE_DIM, num_arm=MAX_CANDIDATES, beta=0.01, lamb=10, hidden=HIDDEN_DIM, buffer_size=2000)
                agents_tuple = (agent,)
                run_main = lambda: run_standard_bandit(env_semantic, agent, num_steps=NUM_STEPS, max_candidates=MAX_CANDIDATES, allowed_users=train_users, track_selection=VISUALIZE_SELECTION, allow_repeat=ALLOW_REPEAT_RECOMMENDATION)
                run_cs = lambda eval_ag, uid, steps, recorder=None: wrap_cs_returns(run_standard_bandit(env_semantic, eval_ag[0], num_steps=steps, max_candidates=MAX_CANDIDATES, fixed_user=uid, track_selection=True, allow_repeat=ALLOW_REPEAT_RECOMMENDATION_CS))

            elif alg_name in ("LowD_HieRec_LinearUCB", "LowD_HieRec_LinearUCB_Fusion"):

                _base_alg = alg_name.replace("_Fusion", "")
                _filter_mode = "fusion" if alg_name.endswith("_Fusion") else "score"
                _k1, _k2 = PER_ALG_TOPK[_base_alg]
                c_ag = LinUCB(d=SEMANTIC_DIM, beta=0.1, lamb=1)
                t_ag = LinUCB(d=SEMANTIC_DIM, beta=1, lamb=0.1)
                m_ag = LinUCB_movie(d=SEMANTIC_DIM, beta=0.01, lamb=1)
                # m_ag = LinUCB_movie(d=SEMANTIC_DIM, beta=0.01, lamb=1)
                agents_tuple = (c_ag, t_ag, m_ag)
                run_main = lambda _k1=_k1, _k2=_k2, _fm=_filter_mode: run_lowd_hierec(env_semantic, c_ag, t_ag, m_ag, semantic_cluster_features, tag_features_matrix, cluster_to_tags, tag_to_artists, artist_to_tags, artist_to_clusters, artist_tag_relevance, num_steps=NUM_STEPS, allowed_users=train_users, track_selection=VISUALIZE_SELECTION, allow_repeat=ALLOW_REPEAT_RECOMMENDATION, top_k_clusters=_k1, top_k_tags=_k2, item_filter_mode=_fm)
                run_cs = lambda eval_ag, uid, steps, recorder=None, _k1=_k1, _k2=_k2, _fm=_filter_mode: wrap_cs_returns(run_lowd_hierec(env_semantic, eval_ag[0], eval_ag[1], eval_ag[2], semantic_cluster_features, tag_features_matrix, cluster_to_tags, tag_to_artists, artist_to_tags, artist_to_clusters, artist_tag_relevance, num_steps=steps, fixed_user=uid, track_selection=True, allow_repeat=ALLOW_REPEAT_RECOMMENDATION_CS, funnel_recorder=recorder, top_k_clusters=_k1, top_k_tags=_k2, item_filter_mode=_fm))

            elif alg_name in ("LowD_HieRec_NeuralUCB", "LowD_HieRec_NeuralUCB_Fusion"):

                _base_alg = alg_name.replace("_Fusion", "")
                _filter_mode = "fusion" if alg_name.endswith("_Fusion") else "score"
                _k1, _k2 = PER_ALG_TOPK[_base_alg]
                c_ag = TagNeuralUCB(d=SEMANTIC_DIM, beta=1, lamb=10, hidden=HIDDEN_DIM, buffer_size=2000)
                t_ag = TagNeuralUCB(d=SEMANTIC_DIM, beta=0.1, lamb=10, hidden=HIDDEN_DIM, buffer_size=2000)
                m_ag = NeuralUCBDiag(d=SEMANTIC_DIM, num_arm=MAX_CANDIDATES, beta=0.01, lamb=0.01, hidden=HIDDEN_DIM, buffer_size=2000)
                agents_tuple = (c_ag, t_ag, m_ag)
                run_main = lambda _k1=_k1, _k2=_k2, _fm=_filter_mode: run_lowd_hierec(env_semantic, c_ag, t_ag, m_ag, semantic_cluster_features, tag_features_matrix, cluster_to_tags, tag_to_artists, artist_to_tags, artist_to_clusters, artist_tag_relevance, num_steps=NUM_STEPS, allowed_users=train_users, track_selection=VISUALIZE_SELECTION, allow_repeat=ALLOW_REPEAT_RECOMMENDATION, top_k_clusters=_k1, top_k_tags=_k2, item_filter_mode=_fm)
                run_cs = lambda eval_ag, uid, steps, recorder=None, _k1=_k1, _k2=_k2, _fm=_filter_mode: wrap_cs_returns(run_lowd_hierec(env_semantic, eval_ag[0], eval_ag[1], eval_ag[2], semantic_cluster_features, tag_features_matrix, cluster_to_tags, tag_to_artists, artist_to_tags, artist_to_clusters, artist_tag_relevance, num_steps=steps, fixed_user=uid, track_selection=True, allow_repeat=ALLOW_REPEAT_RECOMMENDATION_CS, funnel_recorder=recorder, top_k_clusters=_k1, top_k_tags=_k2, item_filter_mode=_fm))
            
            elif alg_name in ("LowD_HieRec_NeuralTS", "LowD_HieRec_NeuralTS_Fusion"):

                _base_alg = alg_name.replace("_Fusion", "")
                _filter_mode = "fusion" if alg_name.endswith("_Fusion") else "score"
                _k1, _k2 = PER_ALG_TOPK[_base_alg]
                c_ag = TagNeuralTS(d=SEMANTIC_DIM, beta=0.1, lamb=10, hidden=HIDDEN_DIM, buffer_size=2000)
                t_ag = TagNeuralTS(d=SEMANTIC_DIM, beta=0.1, lamb=10, hidden=HIDDEN_DIM, buffer_size=2000)
                m_ag = NeuralTSDiag(d=SEMANTIC_DIM, num_arm=MAX_CANDIDATES, beta=0.01, lamb=10, hidden=HIDDEN_DIM, buffer_size=2000)
                agents_tuple = (c_ag, t_ag, m_ag)
                run_main = lambda _k1=_k1, _k2=_k2, _fm=_filter_mode: run_lowd_hierec(env_semantic, c_ag, t_ag, m_ag, semantic_cluster_features, tag_features_matrix, cluster_to_tags, tag_to_artists, artist_to_tags, artist_to_clusters, artist_tag_relevance, num_steps=NUM_STEPS, allowed_users=train_users, track_selection=VISUALIZE_SELECTION, allow_repeat=ALLOW_REPEAT_RECOMMENDATION, top_k_clusters=_k1, top_k_tags=_k2, item_filter_mode=_fm)
                run_cs = lambda eval_ag, uid, steps, recorder=None, _k1=_k1, _k2=_k2, _fm=_filter_mode: wrap_cs_returns(run_lowd_hierec(env_semantic, eval_ag[0], eval_ag[1], eval_ag[2], semantic_cluster_features, tag_features_matrix, cluster_to_tags, tag_to_artists, artist_to_tags, artist_to_clusters, artist_tag_relevance, num_steps=steps, fixed_user=uid, track_selection=True, allow_repeat=ALLOW_REPEAT_RECOMMENDATION_CS, funnel_recorder=recorder, top_k_clusters=_k1, top_k_tags=_k2, item_filter_mode=_fm))

            elif alg_name == "H2N_Bandit":
                agent = H2N_Bandit(
                    dim=SEMANTIC_DIM,
                    lambda_=H2N_LAMBDA,
                    arm_nu=H2N_ARM_NU,
                    hidden=H2N_HIDDEN_DIM,
                    sample_num=H2N_SAMPLE_NUM,
                    pool_size=H2N_CATEGORY_PARAM_POOL_SIZE,
                    category_lr=H2N_CATEGORY_LR,
                    arm_lr=H2N_ARM_LR,
                    buffer_size=H2N_BUFFER_SIZE,
                    training_freq_category=(1, 1),
                )
                agents_tuple = (agent,)
                run_main = lambda: run_h2n_bandit(
                    env_semantic, agent, item_to_h2n_categories,
                    num_steps=NUM_STEPS,
                    max_candidates=H2N_MAX_FILTERED_ARMS,
                    hit_threshold=HIT_THRESHOLD,
                    top_k_clusters=H2N_TOP_K_CATEGORIES,
                    category_exploration_coef=H2N_EXPLORE_COEF,
                    artificial_exp_score=H2N_ARTIFICIAL_EXP,
                    train_threshold=H2N_TRAIN_THRESHOLD,
                    train_freq=H2N_TRAIN_FREQ,
                    allowed_users=train_users,
                    track_selection=VISUALIZE_SELECTION,
                    allow_repeat=ALLOW_REPEAT_RECOMMENDATION,
                    candidate_provider=get_candidate_and_positive_items)
                run_cs = lambda eval_ag, uid, steps, recorder=None: wrap_cs_returns(
                    run_h2n_bandit(
                        env_semantic, eval_ag[0], item_to_h2n_categories,
                        num_steps=steps,
                        max_candidates=H2N_MAX_FILTERED_ARMS,
                        hit_threshold=HIT_THRESHOLD,
                        top_k_clusters=H2N_TOP_K_CATEGORIES,
                        category_exploration_coef=H2N_EXPLORE_COEF,
                        artificial_exp_score=H2N_ARTIFICIAL_EXP,
                        train_threshold=H2N_TRAIN_THRESHOLD,
                        train_freq=H2N_TRAIN_FREQ,
                        fixed_user=uid,
                        track_selection=True,
                        allow_repeat=ALLOW_REPEAT_RECOMMENDATION_CS,
                        funnel_recorder=recorder,
                        candidate_provider=get_candidate_and_positive_items))

            elif alg_name == "CoFineUCB":
                agent = CoFineUCB(
                    dim=SEMANTIC_DIM,
                    projection=cofine_projection,
                    lamb=COFINE_LAMBDA,
                    lamb_tilde=COFINE_LAMBDA_TILDE,
                    alpha_fine=COFINE_ALPHA_FINE,
                    alpha_coarse=COFINE_ALPHA_COARSE,
                )
                agents_tuple = (agent,)
                run_main = lambda: run_cofine_ucb(
                    env_semantic, agent,
                    num_steps=NUM_STEPS,
                    max_candidates=MAX_CANDIDATES,
                    hit_threshold=HIT_THRESHOLD,
                    allowed_users=train_users,
                    track_selection=VISUALIZE_SELECTION,
                    allow_repeat=ALLOW_REPEAT_RECOMMENDATION,
                    candidate_provider=get_candidate_and_positive_items)
                run_cs = lambda eval_ag, uid, steps, recorder=None: wrap_cs_returns(
                    run_cofine_ucb(
                        env_semantic, eval_ag[0],
                        num_steps=steps,
                        max_candidates=MAX_CANDIDATES,
                        hit_threshold=HIT_THRESHOLD,
                        fixed_user=uid,
                        track_selection=True,
                        allow_repeat=ALLOW_REPEAT_RECOMMENDATION_CS,
                        funnel_recorder=recorder,
                        candidate_provider=get_candidate_and_positive_items))

            elif alg_name == "ItemCluster_LinearUCB":
                _k1 = ITEM_CLUSTER_TOPK[alg_name]
                c_ag = LinUCB(d=SEMANTIC_DIM, beta=0.1, lamb=1)
                m_ag = LinUCB_movie(d=SEMANTIC_DIM, beta=0.01, lamb=1)
                agents_tuple = (c_ag, m_ag)
                run_main = lambda _k1=_k1: run_item_cluster_bandit(
                    env_semantic, c_ag, m_ag,
                    item_cluster_centroids, item_cluster_item_dict, item_to_item_clusters,
                    num_steps=NUM_STEPS, max_candidates=MAX_CANDIDATES,
                    hit_threshold=HIT_THRESHOLD,
                    top_k_clusters=_k1,
                    allowed_users=train_users,
                    track_selection=VISUALIZE_SELECTION,
                    allow_repeat=ALLOW_REPEAT_RECOMMENDATION,
                    candidate_provider=get_candidate_and_positive_items)
                run_cs = lambda eval_ag, uid, steps, recorder=None, _k1=_k1: wrap_cs_returns(
                    run_item_cluster_bandit(
                        env_semantic, eval_ag[0], eval_ag[1],
                        item_cluster_centroids, item_cluster_item_dict, item_to_item_clusters,
                        num_steps=steps, max_candidates=MAX_CANDIDATES,
                        hit_threshold=HIT_THRESHOLD,
                        top_k_clusters=_k1,
                        fixed_user=uid, track_selection=True,
                        allow_repeat=ALLOW_REPEAT_RECOMMENDATION_CS,
                        funnel_recorder=recorder,
                        candidate_provider=get_candidate_and_positive_items))

            elif alg_name == "ItemCluster_NeuralUCB":
                _k1 = ITEM_CLUSTER_TOPK[alg_name]
                c_ag = TagNeuralUCB(d=SEMANTIC_DIM, beta=1, lamb=10, hidden=HIDDEN_DIM, buffer_size=2000)
                m_ag = NeuralUCBDiag(d=SEMANTIC_DIM, num_arm=MAX_CANDIDATES, beta=0.01, lamb=0.01, hidden=HIDDEN_DIM, buffer_size=2000)
                agents_tuple = (c_ag, m_ag)
                run_main = lambda _k1=_k1: run_item_cluster_bandit(
                    env_semantic, c_ag, m_ag,
                    item_cluster_centroids, item_cluster_item_dict, item_to_item_clusters,
                    num_steps=NUM_STEPS, max_candidates=MAX_CANDIDATES,
                    hit_threshold=HIT_THRESHOLD,
                    top_k_clusters=_k1,
                    allowed_users=train_users,
                    track_selection=VISUALIZE_SELECTION,
                    allow_repeat=ALLOW_REPEAT_RECOMMENDATION,
                    candidate_provider=get_candidate_and_positive_items)
                run_cs = lambda eval_ag, uid, steps, recorder=None, _k1=_k1: wrap_cs_returns(
                    run_item_cluster_bandit(
                        env_semantic, eval_ag[0], eval_ag[1],
                        item_cluster_centroids, item_cluster_item_dict, item_to_item_clusters,
                        num_steps=steps, max_candidates=MAX_CANDIDATES,
                        hit_threshold=HIT_THRESHOLD,
                        top_k_clusters=_k1,
                        fixed_user=uid, track_selection=True,
                        allow_repeat=ALLOW_REPEAT_RECOMMENDATION_CS,
                        funnel_recorder=recorder,
                        candidate_provider=get_candidate_and_positive_items))

            elif alg_name == "ItemCluster_NeuralTS":
                _k1 = ITEM_CLUSTER_TOPK[alg_name]
                c_ag = TagNeuralTS(d=SEMANTIC_DIM, beta=0.1, lamb=10, hidden=HIDDEN_DIM, buffer_size=2000)
                m_ag = NeuralTSDiag(d=SEMANTIC_DIM, num_arm=MAX_CANDIDATES, beta=0.01, lamb=10, hidden=HIDDEN_DIM, buffer_size=2000)
                agents_tuple = (c_ag, m_ag)
                run_main = lambda _k1=_k1: run_item_cluster_bandit(
                    env_semantic, c_ag, m_ag,
                    item_cluster_centroids, item_cluster_item_dict, item_to_item_clusters,
                    num_steps=NUM_STEPS, max_candidates=MAX_CANDIDATES,
                    hit_threshold=HIT_THRESHOLD,
                    top_k_clusters=_k1,
                    allowed_users=train_users,
                    track_selection=VISUALIZE_SELECTION,
                    allow_repeat=ALLOW_REPEAT_RECOMMENDATION,
                    candidate_provider=get_candidate_and_positive_items)
                run_cs = lambda eval_ag, uid, steps, recorder=None, _k1=_k1: wrap_cs_returns(
                    run_item_cluster_bandit(
                        env_semantic, eval_ag[0], eval_ag[1],
                        item_cluster_centroids, item_cluster_item_dict, item_to_item_clusters,
                        num_steps=steps, max_candidates=MAX_CANDIDATES,
                        hit_threshold=HIT_THRESHOLD,
                        top_k_clusters=_k1,
                        fixed_user=uid, track_selection=True,
                        allow_repeat=ALLOW_REPEAT_RECOMMENDATION_CS,
                        funnel_recorder=recorder,
                        candidate_provider=get_candidate_and_positive_items))

            # ================================================================
            # Ablation: No Feature-Dimension (cluster) Level
            # Tag candidates are drawn uniformly at random (NUM_TAG_ARMS arms)
            # instead of being funnelled through a cluster agent.
            # ================================================================
            elif alg_name == "LowD_HieRec_LinearUCB_NoFD":

                _k2 = PER_ALG_TOPK["LowD_HieRec_LinearUCB"][1]
                t_ag = LinUCB(d=SEMANTIC_DIM, beta=1, lamb=0.1)
                m_ag = LinUCB_movie(d=SEMANTIC_DIM, beta=0.01, lamb=1)
                agents_tuple = (t_ag, m_ag)
                run_main = lambda _k2=_k2: run_lowd_hierec_no_fd(
                    env_semantic, t_ag, m_ag, tag_features_matrix,
                    artist_to_tags, artist_tag_relevance,
                    num_tag_arms=NUM_TAG_ARMS, top_k_tags=_k2,
                    max_candidates=MAX_CANDIDATES, hit_threshold=HIT_THRESHOLD,
                    num_steps=NUM_STEPS, allowed_users=train_users,
                    track_selection=VISUALIZE_SELECTION,
                    allow_repeat=ALLOW_REPEAT_RECOMMENDATION,
                    candidate_provider=get_candidate_and_positive_items,
                    item_filter_mode="score")
                run_cs = lambda eval_ag, uid, steps, recorder=None, _k2=_k2: wrap_cs_returns(
                    run_lowd_hierec_no_fd(
                        env_semantic, eval_ag[0], eval_ag[1], tag_features_matrix,
                        artist_to_tags, artist_tag_relevance,
                        num_tag_arms=NUM_TAG_ARMS, top_k_tags=_k2,
                        max_candidates=MAX_CANDIDATES, hit_threshold=HIT_THRESHOLD,
                        num_steps=steps, fixed_user=uid,
                        track_selection=True,
                        allow_repeat=ALLOW_REPEAT_RECOMMENDATION_CS,
                        funnel_recorder=recorder,
                        candidate_provider=get_candidate_and_positive_items,
                        item_filter_mode="score"))

            elif alg_name == "LowD_HieRec_NeuralUCB_NoFD":

                _k2 = PER_ALG_TOPK["LowD_HieRec_NeuralUCB"][1]
                t_ag = TagNeuralUCB(d=SEMANTIC_DIM, beta=0.1, lamb=10, hidden=HIDDEN_DIM, buffer_size=2000)
                m_ag = NeuralUCBDiag(d=SEMANTIC_DIM, num_arm=MAX_CANDIDATES, beta=0.01, lamb=0.01, hidden=HIDDEN_DIM, buffer_size=2000)
                agents_tuple = (t_ag, m_ag)
                run_main = lambda _k2=_k2: run_lowd_hierec_no_fd(
                    env_semantic, t_ag, m_ag, tag_features_matrix,
                    artist_to_tags, artist_tag_relevance,
                    num_tag_arms=NUM_TAG_ARMS, top_k_tags=_k2,
                    max_candidates=MAX_CANDIDATES, hit_threshold=HIT_THRESHOLD,
                    num_steps=NUM_STEPS, allowed_users=train_users,
                    track_selection=VISUALIZE_SELECTION,
                    allow_repeat=ALLOW_REPEAT_RECOMMENDATION,
                    candidate_provider=get_candidate_and_positive_items,
                    item_filter_mode="score")
                run_cs = lambda eval_ag, uid, steps, recorder=None, _k2=_k2: wrap_cs_returns(
                    run_lowd_hierec_no_fd(
                        env_semantic, eval_ag[0], eval_ag[1], tag_features_matrix,
                        artist_to_tags, artist_tag_relevance,
                        num_tag_arms=NUM_TAG_ARMS, top_k_tags=_k2,
                        max_candidates=MAX_CANDIDATES, hit_threshold=HIT_THRESHOLD,
                        num_steps=steps, fixed_user=uid,
                        track_selection=True,
                        allow_repeat=ALLOW_REPEAT_RECOMMENDATION_CS,
                        funnel_recorder=recorder,
                        candidate_provider=get_candidate_and_positive_items,
                        item_filter_mode="score"))

            elif alg_name == "LowD_HieRec_NeuralTS_NoFD":

                _k2 = PER_ALG_TOPK["LowD_HieRec_NeuralTS"][1]
                t_ag = TagNeuralTS(d=SEMANTIC_DIM, beta=0.1, lamb=10, hidden=HIDDEN_DIM, buffer_size=2000)
                m_ag = NeuralTSDiag(d=SEMANTIC_DIM, num_arm=MAX_CANDIDATES, beta=0.01, lamb=10, hidden=HIDDEN_DIM, buffer_size=2000)
                agents_tuple = (t_ag, m_ag)
                run_main = lambda _k2=_k2: run_lowd_hierec_no_fd(
                    env_semantic, t_ag, m_ag, tag_features_matrix,
                    artist_to_tags, artist_tag_relevance,
                    num_tag_arms=NUM_TAG_ARMS, top_k_tags=_k2,
                    max_candidates=MAX_CANDIDATES, hit_threshold=HIT_THRESHOLD,
                    num_steps=NUM_STEPS, allowed_users=train_users,
                    track_selection=VISUALIZE_SELECTION,
                    allow_repeat=ALLOW_REPEAT_RECOMMENDATION,
                    candidate_provider=get_candidate_and_positive_items,
                    item_filter_mode="score")
                run_cs = lambda eval_ag, uid, steps, recorder=None, _k2=_k2: wrap_cs_returns(
                    run_lowd_hierec_no_fd(
                        env_semantic, eval_ag[0], eval_ag[1], tag_features_matrix,
                        artist_to_tags, artist_tag_relevance,
                        num_tag_arms=NUM_TAG_ARMS, top_k_tags=_k2,
                        max_candidates=MAX_CANDIDATES, hit_threshold=HIT_THRESHOLD,
                        num_steps=steps, fixed_user=uid,
                        track_selection=True,
                        allow_repeat=ALLOW_REPEAT_RECOMMENDATION_CS,
                        funnel_recorder=recorder,
                        candidate_provider=get_candidate_and_positive_items,
                        item_filter_mode="score"))

            else:
                continue
            
            reg, rew, hits, h5, h10, sels, loss_run = run_main()
            regrets_all.append(reg)
            rewards_all.append(rew)
            hits_all.append(hits)
            hits_5_all.append(h5)
            hits_10_all.append(h10)
            losses_all.append(loss_run)
            if VISUALIZE_SELECTION: selections_all.append(sels)
            
            if DO_COLD_START:
                current_track_users = track_cs_users if i == 0 else None
                is_lowd = alg_name in LOWD_FUNNEL_ALGS and i == 0
                funnel_track = track_cs_users if is_lowd else None
                cs_reg, cs_rew, cs_hit, tracked_sels, all_cs_sels, funnel_data = evaluate_cold_start(
                    agents_tuple, run_cs, cs_users, COLD_START_STEPS,
                    track_users=current_track_users,
                    funnel_track_users=funnel_track)
                
                cs_regrets_all.append(cs_reg)
                cs_rewards_all.append(cs_rew)
                cs_hits_all.append(cs_hit)

                if i == 0:
                    TRACKED_TRAJECTORIES[alg_name] = tracked_sels
                    ALL_CS_TRAJECTORIES[alg_name] = all_cs_sels
                    if is_lowd and len(funnel_data) > 0:
                        FUNNEL_DATA[alg_name] = funnel_data

        RESULTS[alg_name] = {
            "regrets": regrets_all, "rewards": rewards_all, 
            "hits": hits_all, "hits_5": hits_5_all, "hits_10": hits_10_all,
            "selections": selections_all, "losses": losses_all
        }
        
        if DO_COLD_START:
            RESULTS_CS[alg_name] = {
                "regrets": cs_regrets_all, "rewards": cs_rewards_all, "hits": cs_hits_all
            }

    if DO_COLD_START and len(ALL_CS_TRAJECTORIES) > 0:
        print("\n" + "=" * 80)
        print("Post-processing cold-start trajectories: filter LowD-HieRec & pair with baselines")
        print("=" * 80)

        for lowd_alg, base_alg in LOWD_TO_BASELINE.items():
            if lowd_alg not in ALL_CS_TRAJECTORIES:
                continue

            # 1) 仅对 LowD 侧筛选 + 落盘
            lowd_filtered = filter_convergence_trajectories(
                ALL_CS_TRAJECTORIES[lowd_alg], artist_primary_l0,
                min_tail=CS_CONVERGENCE_MIN_TAIL,
                min_prefix_diversity=CS_CONVERGENCE_MIN_PREFIX_DIVERSITY,
            )
            print(f"  [{lowd_alg}] Convergence filter: "
                  f"{len(lowd_filtered)}/{len(cs_users)} users matched "
                  f"(min_tail={CS_CONVERGENCE_MIN_TAIL}, "
                  f"min_prefix_diversity={CS_CONVERGENCE_MIN_PREFIX_DIVERSITY}).")
            save_convergence_trajectories(lowd_filtered, lowd_alg, RESULTS_DIR)

            # 2) 对应 baseline 侧: 用 LowD 的 uid 集合落盘 baseline 自己的轨迹
            if base_alg not in ALL_CS_TRAJECTORIES:
                print(f"  [{base_alg}] Skipped: not found in ALL_CS_TRAJECTORIES (algorithm wasn't run?).")
                continue
            if len(lowd_filtered) == 0:
                print(f"  [{base_alg}] Skipped: no users were matched on {lowd_alg} side.")
                continue

            matched_uids = sorted(lowd_filtered.keys())
            base_trajectories = compute_trajectory_metadata(
                ALL_CS_TRAJECTORIES[base_alg], artist_primary_l0, matched_uids
            )
            print(f"  [{base_alg}] Paired baseline trajectories: "
                  f"{len(base_trajectories)}/{len(matched_uids)} users saved (no filtering).")
            save_convergence_trajectories(base_trajectories, base_alg, RESULTS_DIR)

        print("=" * 80 + "\n")

    if DO_COLD_START and len(FUNNEL_DATA) > 0:
        funnel_path = os.path.join(RESULTS_DIR, "cs_funnel_snapshots.pkl")
        with open(funnel_path, 'wb') as f:
            pickle.dump(FUNNEL_DATA, f)
        total_snaps = sum(len(steps) for alg in FUNNEL_DATA.values() for steps in alg.values())
        print(f"✅ Action-space funnel data saved: {funnel_path}")
        print(f"   {len(FUNNEL_DATA)} alg(s) × {len(next(iter(FUNNEL_DATA.values())))} user(s), "
              f"{total_snaps} total step-snapshots.")

    if DO_COLD_START and RECORD_ALL_CS_TRAJECTORIES and len(ALL_CS_TRAJECTORIES) > 0:
        traj_path = os.path.join(RESULTS_DIR, "cs_all_trajectories.pkl")
        with open(traj_path, 'wb') as f:
            pickle.dump(ALL_CS_TRAJECTORIES, f)
        n_algs  = len(ALL_CS_TRAJECTORIES)
        n_users = sum(len(v) for v in ALL_CS_TRAJECTORIES.values())
        print(f"✅ All cold-start trajectories saved: {traj_path}")
        print(f"   {n_algs} alg(s) × {n_users // max(n_algs, 1)} user(s).")

    print("\nSaving results to Last.fm directories...")
    
    def safe_to_numpy(data_list):
        if not data_list or len(data_list) == 0: return np.array([])
        min_len = min(len(x) for x in data_list)
        return np.array([x[:min_len] for x in data_list])

    def across_run_std(data_matrix):
        """Across-run SD using the population convention reported in the paper."""
        ddof = 0
        return np.std(data_matrix, axis=0, ddof=ddof)

    def save_metric_npy(data_matrix, name_suffix, algo, smoothing=True):
        if len(data_matrix) == 0: return
        if smoothing:
            smoothed_runs = []
            for i in range(data_matrix.shape[0]):
                series = pd.Series(data_matrix[i])
                smoothed = series.rolling(window=500, min_periods=1).mean().values
                smoothed_runs.append(smoothed)
            matrix_to_agg = np.array(smoothed_runs)
        else:
            matrix_to_agg = data_matrix
        
        mean_val = np.mean(matrix_to_agg, axis=0)
        std_val = across_run_std(matrix_to_agg)
        np.save(os.path.join(RESULTS_DIR, f"mean_{name_suffix}_{algo}.npy"), mean_val)
        np.save(os.path.join(RESULTS_DIR, f"std_{name_suffix}_{algo}.npy"), std_val) 

    for algo_name, data in RESULTS.items():
        reg_mat = safe_to_numpy(data['regrets'])
        if len(reg_mat) > 0:
            cumsum_reg = np.cumsum(reg_mat, axis=1)
            np.save(os.path.join(RESULTS_DIR, f"mean_regret_{algo_name}.npy"), np.mean(cumsum_reg, axis=0))
            np.save(os.path.join(RESULTS_DIR, f"std_regret_{algo_name}.npy"), across_run_std(cumsum_reg))
            save_metric_npy(reg_mat, "ma_regret", algo_name)
            
        rew_mat = safe_to_numpy(data['rewards'])
        if len(rew_mat) > 0:
            cumsum_rew = np.cumsum(rew_mat, axis=1)
            np.save(os.path.join(RESULTS_DIR, f"mean_reward_{algo_name}.npy"), np.mean(cumsum_rew, axis=0))
            np.save(os.path.join(RESULTS_DIR, f"std_reward_{algo_name}.npy"), across_run_std(cumsum_rew))
            save_metric_npy(rew_mat, "ma_reward", algo_name)
            
        save_metric_npy(safe_to_numpy(data['hits']), "hr1", algo_name)
        save_metric_npy(safe_to_numpy(data['hits_5']), "hr5", algo_name)
        save_metric_npy(safe_to_numpy(data['hits_10']), "hr10", algo_name)
        
        loss_mat = safe_to_numpy(data['losses'])
        if loss_mat.ndim > 1 and loss_mat.shape[1] > 0:
            save_metric_npy(loss_mat, "loss", algo_name, smoothing=True)

    if DO_COLD_START:
        print("Saving User Cold-Start metrics to numpy...")
        for algo_name, data in RESULTS_CS.items():
            cs_reg_mat = safe_to_numpy(data['regrets'])
            if len(cs_reg_mat) > 0:
                cumsum_cs_reg = np.cumsum(cs_reg_mat, axis=1)
                np.save(os.path.join(RESULTS_DIR, f"mean_cs_cum_regret_{algo_name}.npy"), np.mean(cumsum_cs_reg, axis=0))
                np.save(os.path.join(RESULTS_DIR, f"std_cs_cum_regret_{algo_name}.npy"), across_run_std(cumsum_cs_reg))
                np.save(os.path.join(RESULTS_DIR, f"mean_cs_avg_regret_{algo_name}.npy"), np.mean(cs_reg_mat, axis=0))
                np.save(os.path.join(RESULTS_DIR, f"std_cs_avg_regret_{algo_name}.npy"), across_run_std(cs_reg_mat))
            
            cs_rew_mat = safe_to_numpy(data['rewards'])
            if len(cs_rew_mat) > 0:
                np.save(os.path.join(RESULTS_DIR, f"mean_cs_reward_{algo_name}.npy"), np.mean(cs_rew_mat, axis=0))
                np.save(os.path.join(RESULTS_DIR, f"std_cs_reward_{algo_name}.npy"), across_run_std(cs_rew_mat))
                
            cs_hits_mat = safe_to_numpy(data['hits'])
            if len(cs_hits_mat) > 0:
                np.save(os.path.join(RESULTS_DIR, f"mean_cs_hits_{algo_name}.npy"), np.mean(cs_hits_mat, axis=0))
                np.save(os.path.join(RESULTS_DIR, f"std_cs_hits_{algo_name}.npy"), across_run_std(cs_hits_mat))


if __name__ == "__main__":
    main()
