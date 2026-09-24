import pandas as pd
import numpy as np
import json
import gzip
import os
import pickle
import re
import argparse
from multiprocessing import Pool, cpu_count
from collections import Counter, defaultdict
from functools import partial
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
from item_cluster_bandit import build_item_space_clusters, select_item_space_k_by_kgs

# Try importing SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception as exc:
    SBERT_AVAILABLE = False
    print(f"Warning: sentence-transformers unavailable ({exc}). Item semantic features will be skipped.")

# ================= Configuration =================
# [Global]
TARGET_USER_COUNT = 5000       
FEATURE_DIM = 50               
RANDOM_SEED = 42

# [Filtering]
MIN_USER_INTERACTION_FOR_ITEM = 5  
MIN_FEATURE_COUNT = 5              

MAX_TAGS_PER_ITEM = 5
SEMANTIC_EMB_MODEL = 'sentence-transformers/all-mpnet-base-v2'

L1_SEARCH_RANGE = range(2, 4)     # Level 1 

# [Clustering Algorithm Selection]
# Supported values:
#   'hierarchical'     - Ward hierarchical clustering; original TDGE preprocessing.
#   'kmeans'           - K-Means; fast general-purpose baseline.
#   'minibatch_kmeans' - Mini-Batch K-Means; fastest option for large Amazon tag sets.
#   'gmm'              - Gaussian Mixture Model with diagonal covariance.
#   'spectral'         - Spectral Clustering; expressive but slow for large tag sets.
#   'random'           - Shuffle hierarchical feature/tag cluster assignments; preserves K and cluster sizes.
CLUSTERING_METHOD = 'hierarchical'

L0_SEARCH_RANGE = range(10, 100)  # Level 0 

# [Paths]
DATA_DIR = "./datasets/amazon_data"
METADATA_DIR = "./datasets/Amazon_2023"

# [Outputs]
USER_MAP_FILE = f"{DATA_DIR}/user_map.pkl"
ITEM_MAP_FILE = f"{DATA_DIR}/item_map.pkl"
VALID_ITEM_FILE = f"{DATA_DIR}/valid_items.pkl"
RATING_MATRIX_FILE = f"{DATA_DIR}/rating_matrix.npy"

USER_EMB_FILE = f"{DATA_DIR}/user_features.npy"         
ITEM_EMB_FILE = f"{DATA_DIR}/semantic_item_features.npy"         
TAG_EMB_FILE = f"{DATA_DIR}/semantic_tag_embeddings.npy" 
CLUSTER_EMB_FILE = f"{DATA_DIR}/semantic_cluster_features.npy"   
CLUSTER_L1_EMB_FILE = f"{DATA_DIR}/semantic_cluster_l1_features.npy" 
CLUSTER_DICT_FILE = f"{DATA_DIR}/semantic_cluster_item_dict.pkl"

FEATURE_CSV_FILE = f"{DATA_DIR}/filtered_features.csv"
TAG2CLUSTER_FILE = f"{DATA_DIR}/tag2clusters.csv"
ITEM_CLUSTERS_FILE = f"{DATA_DIR}/item_multi_level_clusters.csv"
ITEM_SPACE_CLUSTER_FILE = f"{DATA_DIR}/item_space_cluster_features.npy"
ITEM_SPACE_CLUSTER_DICT_FILE = f"{DATA_DIR}/item_space_cluster_item_dict.pkl"
ITEM_SPACE_ITEM_CLUSTERS_FILE = f"{DATA_DIR}/item_space_item_clusters.csv"
ITEM_SPACE_KGS_FILE = f"{DATA_DIR}/item_space_kgs_scores.csv"
DENDROGRAM_FILE = f"{DATA_DIR}/dendrogram.png"
KGS_PLOT_FILE = f"{DATA_DIR}/optimal_k_search_kgs.png" 
ITEM_SPACE_KGS_PLOT_FILE = f"{DATA_DIR}/optimal_k_search_item_space_kgs.png"

# [Categories]
CATEGORIES = [
    "Beauty_and_Personal_Care.jsonl.gz" 
]

# [Blacklist Keys]
IGNORE_KEYS = {
    'upc', 'ean', 'isbn', 'asin', 'gtin', 
    'item model number', 'model number', 'manufacturer part number', 'unspsc code',
    'package dimensions', 'product dimensions', 'item dimensions',
    'item weight', 'shipping weight', 'batteries',
    'date first available', 'is discontinued by manufacturer', 
    'best sellers rank', 'customer reviews', 'customer ratings by feature'
}

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def configure_output_paths(data_dir):
    """Update output paths so different clustering methods can be saved separately."""
    global DATA_DIR
    global USER_MAP_FILE, ITEM_MAP_FILE, VALID_ITEM_FILE, RATING_MATRIX_FILE
    global USER_EMB_FILE, ITEM_EMB_FILE, TAG_EMB_FILE, CLUSTER_EMB_FILE, CLUSTER_L1_EMB_FILE, CLUSTER_DICT_FILE
    global FEATURE_CSV_FILE, TAG2CLUSTER_FILE, ITEM_CLUSTERS_FILE
    global ITEM_SPACE_CLUSTER_FILE, ITEM_SPACE_CLUSTER_DICT_FILE
    global ITEM_SPACE_ITEM_CLUSTERS_FILE, ITEM_SPACE_KGS_FILE
    global DENDROGRAM_FILE, KGS_PLOT_FILE, ITEM_SPACE_KGS_PLOT_FILE

    DATA_DIR = data_dir
    os.makedirs(DATA_DIR, exist_ok=True)

    USER_MAP_FILE = f"{DATA_DIR}/user_map.pkl"
    ITEM_MAP_FILE = f"{DATA_DIR}/item_map.pkl"
    VALID_ITEM_FILE = f"{DATA_DIR}/valid_items.pkl"
    RATING_MATRIX_FILE = f"{DATA_DIR}/rating_matrix.npy"

    USER_EMB_FILE = f"{DATA_DIR}/user_features.npy"
    ITEM_EMB_FILE = f"{DATA_DIR}/semantic_item_features.npy"
    TAG_EMB_FILE = f"{DATA_DIR}/semantic_tag_embeddings.npy"
    CLUSTER_EMB_FILE = f"{DATA_DIR}/semantic_cluster_features.npy"
    CLUSTER_L1_EMB_FILE = f"{DATA_DIR}/semantic_cluster_l1_features.npy"
    CLUSTER_DICT_FILE = f"{DATA_DIR}/semantic_cluster_item_dict.pkl"

    FEATURE_CSV_FILE = f"{DATA_DIR}/filtered_features.csv"
    TAG2CLUSTER_FILE = f"{DATA_DIR}/tag2clusters.csv"
    ITEM_CLUSTERS_FILE = f"{DATA_DIR}/item_multi_level_clusters.csv"
    ITEM_SPACE_CLUSTER_FILE = f"{DATA_DIR}/item_space_cluster_features.npy"
    ITEM_SPACE_CLUSTER_DICT_FILE = f"{DATA_DIR}/item_space_cluster_item_dict.pkl"
    ITEM_SPACE_ITEM_CLUSTERS_FILE = f"{DATA_DIR}/item_space_item_clusters.csv"
    ITEM_SPACE_KGS_FILE = f"{DATA_DIR}/item_space_kgs_scores.csv"
    DENDROGRAM_FILE = f"{DATA_DIR}/dendrogram.png"
    KGS_PLOT_FILE = f"{DATA_DIR}/optimal_k_search_kgs.png"
    ITEM_SPACE_KGS_PLOT_FILE = f"{DATA_DIR}/optimal_k_search_item_space_kgs.png"


# ================= Worker Functions =================
def count_users_worker(file_name):
    file_path = os.path.join(METADATA_DIR, file_name)
    u_counter = Counter()
    if not os.path.exists(file_path): 
        print(f"  [Error] File not found: {file_path}")
        return u_counter
    
    print(f"  [Worker] Start counting users in {file_name}...")
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i > 0 and i % 1000000 == 0:
                    print(f"    -> [{file_name}] User counting: Processed {i:,} lines...")
                try:
                    obj = json.loads(line.strip())
                    uid = obj.get('reviewerID', obj.get('user_id'))
                    if uid: u_counter[uid] += 1
                except: continue
    except Exception as e: 
        print(f"  [Error] {e}")
        
    print(f"  [Worker] Finished {file_name}. Found {len(u_counter):,} unique users.")
    return u_counter

def extract_interactions_worker(file_name, valid_users):
    file_path = os.path.join(METADATA_DIR, file_name)
    data = []
    if not os.path.exists(file_path): return data
    
    print(f"  [Worker] Start extracting interactions from {file_name}...")
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i > 0 and i % 1000000 == 0:
                    print(f"    -> [{file_name}] Interaction extracting: Processed {i:,} lines...")
                try:
                    obj = json.loads(line.strip())
                    uid = obj.get('reviewerID', obj.get('user_id'))
                    iid = obj.get('parent_asin')
                    rating = obj.get('rating')
                    if uid in valid_users and iid and rating is not None:
                        data.append((uid, iid, float(rating)))
                except: continue
    except Exception as e: pass
    
    print(f"  [Worker] Finished {file_name}. Kept {len(data):,} interactions.")
    return data

def extract_features_worker(category_file_name, valid_items, metadata_dir):
    features = []
    path_prefix = os.path.join(metadata_dir, f"meta_{category_file_name}")
    path_orig = os.path.join(metadata_dir, category_file_name)
    target_path = path_prefix if os.path.exists(path_prefix) else path_orig
    if not os.path.exists(target_path): return []
    
    print(f"  [Worker] Start extracting features from {target_path}...")
    try:
        with gzip.open(target_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i > 0 and i % 500000 == 0:
                    print(f"    -> [{category_file_name}] Feature extracting: Processed {i:,} lines...")
                try:
                    obj = json.loads(line.strip())
                    iid = obj.get('parent_asin')
                    if iid not in valid_items: continue
                    m_cat = obj.get('main_category')
                    
                    if m_cat:
                        val = m_cat.replace('<br>', ' ').strip()
                        features.append((iid, "Main Category", val))
                    
                    details = obj.get('details')
                    if isinstance(details, dict):
                        for k, v in details.items():
                            if k.lower().strip() in IGNORE_KEYS: continue
                            if isinstance(v, str):
                                v = v.strip()
                                if v and not (v.replace('.','').isdigit() and len(v)>4):
                                    if len(v) > 100: v = v[:100]
                                    features.append((iid, k.strip(), v))
                except: continue
    except: pass
    
    print(f"  [Worker] Finished {category_file_name}. Extracted {len(features):,} raw features.")
    return features

def compute_wcss(embeddings, labels):
    wcss = 0.0
    for label in np.unique(labels):
        pts = embeddings[labels == label]
        if len(pts) > 0:
            wcss += np.sum((pts - np.mean(pts, axis=0))**2)
    return wcss


# --------------------------------------------------------------------------
# Pluggable clustering utilities for clustering-method ablation
# --------------------------------------------------------------------------
SUPPORTED_CLUSTERING_METHODS = {
    'hierarchical',
    'kmeans',
    'minibatch_kmeans',
    'gmm',
    'spectral',
    'random',
}


def _safe_k_range(k_range, n_samples):
    """Clamp a requested K range so every K is valid for the tag set size."""
    if n_samples < 2:
        raise ValueError("Need at least 2 tag embeddings for clustering.")
    safe_start = max(2, min(k_range.start, n_samples))
    safe_stop = min(k_range.stop, n_samples + 1)  # Python range stop is exclusive.
    if safe_start >= safe_stop:
        safe_start = max(2, min(n_samples - 1, k_range.start))
        safe_stop = min(n_samples + 1, safe_start + 1)
    return range(safe_start, safe_stop)


def fit_clusters(embeddings, k, method=None, Z=None):
    """Return 0-indexed integer cluster labels for a given clustering method."""
    method = CLUSTERING_METHOD if method is None else method
    if method not in SUPPORTED_CLUSTERING_METHODS:
        raise ValueError(
            f"Unknown CLUSTERING_METHOD: '{method}'. Choose from: "
            f"{sorted(SUPPORTED_CLUSTERING_METHODS)}."
        )

    k = int(k)
    n = len(embeddings)
    if k < 2 or k > n:
        raise ValueError(f"Invalid k={k}; expected 2 <= k <= n_samples={n}.")

    if method == 'hierarchical':
        if Z is None:
            raise ValueError("Linkage matrix Z is required for hierarchical clustering.")
        labels = fcluster(Z, t=k, criterion='maxclust') - 1
    elif method == 'kmeans':
        model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = model.fit_predict(embeddings)
    elif method == 'minibatch_kmeans':
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=RANDOM_SEED,
            n_init=10,
            batch_size=min(1024, max(128, len(embeddings))),
        )
        labels = model.fit_predict(embeddings)
    elif method == 'gmm':
        # Diagonal covariance is much more stable and faster for high-dimensional
        # SBERT embeddings than full covariance, especially on Last.fm/Amazon.
        model = GaussianMixture(
            n_components=k,
            random_state=RANDOM_SEED,
            covariance_type='diag',
            max_iter=200,
            reg_covar=1e-6,
        )
        model.fit(embeddings)
        labels = model.predict(embeddings)
    elif method == 'spectral':
        n_neighbors = min(15, max(1, len(embeddings) - 1))
        print(f"   [Warning] SpectralClustering on {len(embeddings)} tags may be slow.")
        model = SpectralClustering(
            n_clusters=k,
            random_state=RANDOM_SEED,
            affinity='nearest_neighbors',
            n_neighbors=n_neighbors,
            assign_labels='kmeans',
            n_jobs=-1,
        )
        labels = model.fit_predict(embeddings)
    elif method == 'random':
        raise ValueError(
            "Random clustering is produced by shuffling hierarchical final labels; "
            "call fit_clusters with method='hierarchical' for KGS/final base labels."
        )
    else:
        raise AssertionError("Unreachable clustering branch.")

    return labels.astype(int)


def shuffle_hierarchical_cluster_pairs(fine_labels, coarse_labels, random_state=RANDOM_SEED):
    """Randomize feature/tag cluster assignments while preserving HC cluster sizes."""
    fine_labels = np.asarray(fine_labels, dtype=int)
    coarse_labels = np.asarray(coarse_labels, dtype=int)
    if fine_labels.shape != coarse_labels.shape:
        raise ValueError("Fine and coarse cluster labels must have the same shape.")
    rng = np.random.default_rng(int(random_state) + len(fine_labels) * 9173)
    perm = rng.permutation(len(fine_labels))
    return fine_labels[perm].astype(int), coarse_labels[perm].astype(int)


def auto_find_best_k_kgs(embeddings, k_range, desc="", min_cluster_size=3,
                         method=None, Z=None):
    """Generic KGS search that works for all supported clustering methods."""
    method = CLUSTERING_METHOD if method is None else method
    actual_range = _safe_k_range(k_range, len(embeddings))

    x = []
    wcss_list = []
    valid_flags = []

    print(
        f"   Evaluating KGS Penalty for {desc} [{method}] "
        f"in K range {actual_range.start} to {actual_range.stop - 1}..."
    )
    for k in tqdm(actual_range, desc=f"Eval {desc}", leave=False):
        labels = fit_clusters(embeddings, k, method=method, Z=Z)
        wcss = compute_wcss(embeddings, labels)
        x.append(k)
        wcss_list.append(wcss)

        cluster_sizes = pd.Series(labels).value_counts()
        valid_flags.append(int(cluster_sizes.min()) >= int(min_cluster_size))

    x = np.array(x)
    wcss_list = np.array(wcss_list)
    valid_flags = np.array(valid_flags)

    if len(x) == 0:
        raise RuntimeError(f"Empty KGS search range for {desc}.")

    wcss_norm = (wcss_list - np.min(wcss_list)) / (np.max(wcss_list) - np.min(wcss_list) + 1e-9)
    k_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)
    kgs_scores = wcss_norm + k_norm

    valid_indices = np.where(valid_flags)[0]
    if len(valid_indices) == 0:
        print(f"Warning: No valid K satisfying min_cluster_size={min_cluster_size}. Fallback to the lowest KGS score.")
        best_idx = int(np.argmin(kgs_scores))
    else:
        best_idx = int(valid_indices[np.argmin(kgs_scores[valid_indices])])

    best_k = int(x[best_idx])
    best_score = float(kgs_scores[best_idx])
    print(f"   -> [KGS Selected] Optimal {desc} K: {best_k} (KGS Penalty: {best_score:.4f})")
    return best_k, x, kgs_scores, best_idx, valid_flags


def process_item_space_clusters(item_ids):
    print("\n=== Generating Item-Space Clusters for ItemCluster Ablation ===")

    if not os.path.exists(ITEM_EMB_FILE):
        print("Warning: Semantic item features not found. Skipping item-space clusters.")
        return

    item_features = np.load(ITEM_EMB_FILE)
    num_clusters, k_vals, kgs_scores, best_idx, valid_flags = select_item_space_k_by_kgs(
        item_features,
        k_range=L0_SEARCH_RANGE,
        min_cluster_size=5,
        random_state=RANDOM_SEED,
    )
    pd.DataFrame({
        "k": k_vals.astype(int),
        "kgs_score": kgs_scores,
        "valid_min_cluster_size": valid_flags.astype(int),
        "selected": (np.arange(len(k_vals)) == best_idx).astype(int),
    }).to_csv(ITEM_SPACE_KGS_FILE, index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(k_vals, kgs_scores, marker='o', markersize=3, linestyle='-',
            linewidth=2.0, color='#1f77b4', zorder=2)
    ax.scatter([k_vals[best_idx]], [kgs_scores[best_idx]], color='#d62728',
               s=180, zorder=5, marker='*', edgecolors='none')
    ax.annotate(f'$k^* = {int(k_vals[best_idx])}$',
                xy=(k_vals[best_idx], kgs_scores[best_idx]),
                xytext=(k_vals[best_idx], kgs_scores[best_idx] + 0.12),
                fontsize=14, fontweight='bold', color='#d62728',
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
                zorder=6)
    ax.set_xlabel("Number of Item Clusters (K)", fontsize=18)
    ax.set_ylabel("KGS Penalty Score", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(ITEM_SPACE_KGS_PLOT_FILE, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Item-space KGS curve saved to {ITEM_SPACE_KGS_PLOT_FILE}")

    print(f"Clustering {item_features.shape[0]} items into {num_clusters} item-space clusters...")
    _, cluster_centroids, cluster_item_dict, item_to_clusters = build_item_space_clusters(
        item_features,
        num_clusters=int(num_clusters),
        random_state=RANDOM_SEED,
    )
    np.save(ITEM_SPACE_CLUSTER_FILE, cluster_centroids)
    with open(ITEM_SPACE_CLUSTER_DICT_FILE, "wb") as f:
        pickle.dump(cluster_item_dict, f)

    rows = [
        {
            "item_id": item_ids[item_idx],
            "item_index": int(item_idx),
            "item_cluster": int(item_to_clusters[item_idx][0]),
        }
        for item_idx in range(len(item_ids))
    ]
    pd.DataFrame(rows).to_csv(ITEM_SPACE_ITEM_CLUSTERS_FILE, index=False)

    sizes = pd.Series([item_to_clusters[i][0] for i in range(len(item_ids))]).value_counts().sort_index().tolist()
    print(f"Saved item-space cluster features: {cluster_centroids.shape}")
    print(f"Saved item-space item mapping. Cluster sizes: {sizes}")


# ================= Core Processing Logic =================

def main():
    print(f"=== Amazon Unified Preprocessing Pipeline (Strictly Aligned with Movielens/LastFM KGS) ===")
    num_processes = min(cpu_count(), len(CATEGORIES))
    
    # ---------------------------------------------------------
    # PART 1: Identify Top Users
    # ---------------------------------------------------------
    print("\n>>> [Step 1] Finding Top Users...")
    global_u_counts = Counter()
    with Pool(num_processes) as pool:
        results = pool.map(count_users_worker, CATEGORIES)
    for res in results: global_u_counts.update(res)
    
    top_users_list = global_u_counts.most_common(TARGET_USER_COUNT)
    if not top_users_list: return
    target_users_set = set([u for u, c in top_users_list])
    
    # ---------------------------------------------------------
    # PART 2: Extract Interactions & Filter Items
    # ---------------------------------------------------------
    print("\n>>> [Step 2] Extracting Interactions & Filtering Items...")
    worker_func_inter = partial(extract_interactions_worker, valid_users=target_users_set)
    with Pool(num_processes) as pool:
        data_chunks = pool.map(worker_func_inter, CATEGORIES)
    raw_interactions = [item for sublist in data_chunks for item in sublist]
    
    item_counts = Counter([x[1] for x in raw_interactions])
    valid_items_set = {i for i, c in item_counts.items() if c >= MIN_USER_INTERACTION_FOR_ITEM}
    final_interactions = [x for x in raw_interactions if x[1] in valid_items_set]
    
    sorted_users = sorted(list(target_users_set))
    sorted_items = sorted(list(valid_items_set))
    user_to_idx = {u: i for i, u in enumerate(sorted_users)}
    item_to_idx = {item: i for i, item in enumerate(sorted_items)}
    
    with open(USER_MAP_FILE, 'wb') as f: pickle.dump(user_to_idx, f)
    with open(ITEM_MAP_FILE, 'wb') as f: pickle.dump(item_to_idx, f)
    with open(VALID_ITEM_FILE, 'wb') as f: pickle.dump(valid_items_set, f)

    # ---------------------------------------------------------
    # PART 3: User Feature Extraction (SVD) & Rating Matrix
    # ---------------------------------------------------------
    print("\n>>> [Step 3] Generating User Features & Rating Matrix...")
    row_indices = [user_to_idx[x[0]] for x in final_interactions]
    col_indices = [item_to_idx[x[1]] for x in final_interactions]
    data_values = [x[2] for x in final_interactions]
    
    interaction_matrix = csr_matrix((data_values, (row_indices, col_indices)), shape=(len(sorted_users), len(sorted_items)))
    svd = TruncatedSVD(n_components=FEATURE_DIM, random_state=RANDOM_SEED)
    user_features = svd.fit_transform(interaction_matrix)
    user_features = normalize(user_features, axis=1)
    np.save(USER_EMB_FILE, user_features)

    rating_matrix = np.full((len(sorted_users), len(sorted_items)), np.nan, dtype=np.float32)
    rating_matrix[row_indices, col_indices] = np.array(data_values) / 5.0
    np.save(RATING_MATRIX_FILE, rating_matrix)

    # ---------------------------------------------------------
    # PART 4: Item Feature Extraction & HC + PCA 
    # ---------------------------------------------------------
    if not SBERT_AVAILABLE:
        process_item_space_clusters(sorted_items)
        return

    print("\n>>> [Step 4] Extracting & Filtering Feature Tags...")
    
    worker_func_feat = partial(extract_features_worker, valid_items=valid_items_set, metadata_dir=METADATA_DIR)
    with Pool(num_processes) as pool:
        feat_results = pool.map(worker_func_feat, CATEGORIES)
    
    all_feat_rows = [r for res in feat_results for r in res]
    
    if len(all_feat_rows) == 0:
        print("\n[CRITICAL ERROR] number of features: 0. Please check the data file.")
        return
        
    df_features = pd.DataFrame(all_feat_rows, columns=['item_id', 'dimension', 'feature'])
    df_features['concat_text'] = df_features['dimension'] + ": " + df_features['feature']
    
    feat_counts = df_features['concat_text'].value_counts()
    valid_feats = feat_counts[feat_counts >= MIN_FEATURE_COUNT].index.tolist()
    df_features = df_features[df_features['concat_text'].isin(valid_feats)]
    
    print(f"    Truncating tags to max {MAX_TAGS_PER_ITEM} per item to maintain balance...")
    df_features = df_features.groupby('item_id').head(MAX_TAGS_PER_ITEM).reset_index(drop=True)
    
    all_feats = df_features['concat_text'].unique().tolist()
    print(f"    Valid unique Tags after truncation: {len(all_feats)}")
    
    if len(all_feats) < 2:
        print("\n[CRITICAL ERROR]feature less than 2, can not conduct clustering")
        return
    
    model = SentenceTransformer(SEMANTIC_EMB_MODEL)
    
    print(f"    Encoding {len(all_feats)} Tags with SBERT [{SEMANTIC_EMB_MODEL}]...")
    tag_embs_768 = model.encode(all_feats, show_progress_bar=True)
    tag_embs_768 = normalize(tag_embs_768, axis=1)
    
    print(f"\n    >>> Performing Clustering on Tags [{CLUSTERING_METHOD}] (using raw SBERT embeddings)...")

    kgs_method = 'hierarchical' if CLUSTERING_METHOD == 'random' else CLUSTERING_METHOD
    Z = None
    if kgs_method == 'hierarchical':
        print("   Building Ward linkage matrix...")
        Z = linkage(tag_embs_768, method='ward')
        if CLUSTERING_METHOD == 'random':
            print("   [Random Control] KGS and cluster sizes are inherited from hierarchical clustering; final tag assignments will be shuffled.")

    OPTIMAL_NUM_L1, _, _, _, _ = auto_find_best_k_kgs(
        tag_embs_768,
        L1_SEARCH_RANGE,
        desc="L1 (Macro)",
        min_cluster_size=2,
        method=kgs_method,
        Z=Z,
    )
    OPTIMAL_NUM_L0, k_vals, kgs_vals, best_idx, valid_flags = auto_find_best_k_kgs(
        tag_embs_768,
        L0_SEARCH_RANGE,
        desc="L0 (Micro)",
        min_cluster_size=5,
        method=kgs_method,
        Z=Z,
    )

    cut_threshold_l1 = 0
    cut_threshold_l0 = 0
    if kgs_method == 'hierarchical' and Z is not None:
        if len(Z) >= OPTIMAL_NUM_L1 and OPTIMAL_NUM_L1 > 1:
            cut_threshold_l1 = (Z[-OPTIMAL_NUM_L1, 2] + Z[-(OPTIMAL_NUM_L1 - 1), 2]) / 2.0
        if len(Z) >= OPTIMAL_NUM_L0 and OPTIMAL_NUM_L0 > 1:
            cut_threshold_l0 = (Z[-OPTIMAL_NUM_L0, 2] + Z[-(OPTIMAL_NUM_L0 - 1), 2]) / 2.0

        print(f"Plotting Dendrogram to {DENDROGRAM_FILE}...")
        plt.figure(figsize=(25, 12))
        dendrogram(Z, truncate_mode='level', p=8, no_labels=True, color_threshold=cut_threshold_l0)

        if cut_threshold_l1 > 0:
            plt.axhline(y=cut_threshold_l1, c='red', ls='--', linewidth=3, label=f'L1 Macro Cut (k={OPTIMAL_NUM_L1})')
            plt.text(0, cut_threshold_l1 * 1.02, f' L1 Threshold ({OPTIMAL_NUM_L1} Clusters)', color='red', fontsize=14, fontweight='bold')
        if cut_threshold_l0 > 0:
            plt.axhline(y=cut_threshold_l0, c='green', ls='-.', linewidth=3, label=f'L0 Micro Cut (k={OPTIMAL_NUM_L0})')
            plt.text(0, cut_threshold_l0 * 1.02, f' L0 Threshold ({OPTIMAL_NUM_L0} Clusters)', color='green', fontsize=14, fontweight='bold')

        plt.legend(fontsize=14, loc='upper right')
        plt.title(f"Hierarchical Clustering Dendrogram - Dual Auto Cuts (L1={OPTIMAL_NUM_L1}, L0={OPTIMAL_NUM_L0})", fontsize=18, fontweight='bold')
        plt.ylabel("Ward Cophenetic Distance", fontsize=14)
        plt.tight_layout()
        plt.savefig(DENDROGRAM_FILE, dpi=300)
        plt.close()
    else:
        print(f"   [Info] Dendrogram skipped because CLUSTERING_METHOD='{CLUSTERING_METHOD}'.")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.plot(k_vals, kgs_vals, linestyle='--', linewidth=1.5, color='gray', alpha=0.6, label='All K values')
    valid_indices = np.where(valid_flags)[0]
    if len(valid_indices) > 0:
        ax.plot(k_vals[valid_indices], kgs_vals[valid_indices], marker='o', markersize=4, linestyle='-', linewidth=2.5, color='#1f77b4', label='Valid exploration space')

    opt_k = k_vals[best_idx]
    opt_kgs = kgs_vals[best_idx]
    ax.axvline(x=opt_k, color='#d62728', linestyle='-.', linewidth=2, label=f'Optimal K = {opt_k}')
    ax.scatter([opt_k], [opt_kgs], color='#d62728', s=150, zorder=5, marker='*')
    ax.set_xlabel("Number of Clusters (K)", fontsize=18)
    ax.set_ylabel("KGS Penalty Score", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(fontsize=15, loc='upper center', frameon=True, edgecolor='gray')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(KGS_PLOT_FILE, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✅ KGS Penalty U-Curve Plot saved to {KGS_PLOT_FILE}")

    clusters_fine = fit_clusters(tag_embs_768, OPTIMAL_NUM_L0, method=kgs_method, Z=Z)
    clusters_coarse = fit_clusters(tag_embs_768, OPTIMAL_NUM_L1, method=kgs_method, Z=Z)
    if CLUSTERING_METHOD == 'random':
        clusters_fine, clusters_coarse = shuffle_hierarchical_cluster_pairs(
            clusters_fine,
            clusters_coarse,
            random_state=RANDOM_SEED,
        )
        print("   [Random Control] Shuffled hierarchical tag-to-cluster assignments while preserving K and cluster-size distribution.")

    tag2clusters = pd.DataFrame({
        'tag_id': range(len(all_feats)),
        'tag_index': range(len(all_feats)), 
        'tag': all_feats,
        'clean_tag': all_feats,
        'lvl0_cluster': clusters_fine,   
        'lvl1_cluster': clusters_coarse  
    })
    tag2clusters.to_csv(TAG2CLUSTER_FILE, index=False)
    print("    Saved Tag Hierarchy mapping.")
    
    print("4. Computing Raw Item Embeddings (768 dim) by Aggregation...")
    feat_text_to_idx = {f: i for i, f in enumerate(all_feats)}
    item_groups = df_features.groupby('item_id')['concat_text'].apply(list)
    item_embs_list = []
    
    for iid in tqdm(sorted_items, desc="Aggregating Tags"):
        if iid in item_groups:
            feats = item_groups[iid]
            t_idxs = [feat_text_to_idx[f] for f in feats if f in feat_text_to_idx]
            if t_idxs:
                avg_emb = np.mean(tag_embs_768[t_idxs], axis=0)
            else:
                avg_emb = np.zeros(tag_embs_768.shape[1])
        else:
            avg_emb = np.zeros(tag_embs_768.shape[1])
        item_embs_list.append(avg_emb)
        
    item_embs_high = np.array(item_embs_list)
    
    print(f"5. Reducing Item Embeddings (PCA 768->{FEATURE_DIM})...")
    pca = PCA(n_components=FEATURE_DIM, random_state=RANDOM_SEED)
    
    item_features_20 = pca.fit_transform(item_embs_high)
    item_features_20 = normalize(item_features_20, axis=1)
    np.save(ITEM_EMB_FILE, item_features_20)
    print(f"    Saved Semantic Item Features (Normalized): {item_features_20.shape}")
    
    print("6. Projecting Tags and Computing Cluster Centroids in reduced space...")
    tag_embs_20 = pca.transform(tag_embs_768) 
    tag_embs_20 = normalize(tag_embs_20, axis=1)
    np.save(TAG_EMB_FILE, tag_embs_20)
    
    def compute_centroids(num_clusters, labels, embeddings):
        centers = []
        for cid in range(num_clusters):
            indices = np.where(labels == cid)[0]
            if len(indices) > 0:
                center = np.mean(embeddings[indices], axis=0)
            else:
                center = np.zeros(embeddings.shape[1])
            centers.append(center)
        return normalize(np.array(centers), axis=1)

    cluster_centers_l0 = compute_centroids(OPTIMAL_NUM_L0, clusters_fine, tag_embs_20)
    np.save(CLUSTER_EMB_FILE, cluster_centers_l0)
    
    cluster_centers_l1 = compute_centroids(OPTIMAL_NUM_L1, clusters_coarse, tag_embs_20)
    np.save(CLUSTER_L1_EMB_FILE, cluster_centers_l1)

    print("8. Generating Item-Cluster Mappings...")
    item_cluster_rows = []
    semantic_cluster_item_dict = defaultdict(list)
    
    for iid in tqdm(sorted_items, desc="Mapping"):
        if iid in item_groups:
            feats = item_groups[iid]
            seen_clusters = set()
            
            for f in feats:
                if f in feat_text_to_idx:
                    t_idx = feat_text_to_idx[f]
                    l0 = clusters_fine[t_idx]
                    l1 = clusters_coarse[t_idx]
                    
                    item_cluster_rows.append({
                        'item_id': iid,
                        'tag_id': t_idx,
                        'tag': f,
                        'tag_index': t_idx, 
                        'relevance': 1.0, 
                        'lvl0_cluster': l0, 
                        'lvl1_cluster': l1  
                    })
                    
                    if l0 not in seen_clusters:
                        semantic_cluster_item_dict[l0].append(iid)
                        seen_clusters.add(l0)

    pd.DataFrame(item_cluster_rows).to_csv(ITEM_CLUSTERS_FILE, index=False)
    with open(CLUSTER_DICT_FILE, 'wb') as f:
        pickle.dump(dict(semantic_cluster_item_dict), f)

    process_item_space_clusters(sorted_items)

    print("\n=== All Preprocessing Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data with selectable semantic tag clustering method.")
    parser.add_argument(
        "--clustering_method",
        type=str,
        default=CLUSTERING_METHOD,
        choices=sorted(SUPPORTED_CLUSTERING_METHODS),
        help="Clustering method used to construct lvl0/lvl1 semantic tag clusters.",
    )
    parser.add_argument("--l1_min", type=int, default=L1_SEARCH_RANGE.start)
    parser.add_argument("--l1_max", type=int, default=L1_SEARCH_RANGE.stop - 1)
    parser.add_argument("--l0_min", type=int, default=L0_SEARCH_RANGE.start)
    parser.add_argument("--l0_max", type=int, default=L0_SEARCH_RANGE.stop - 1)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_DIR,
        help="Output directory for generated .npy/.csv/.pkl files. Use method-specific dirs for ablations.",
    )
    parser.add_argument(
        "--semantic_emb_model",
        type=str,
        default=SEMANTIC_EMB_MODEL,
        help="SentenceTransformer checkpoint used to encode semantic tags/features.",
    )
    args = parser.parse_args()

    CLUSTERING_METHOD = args.clustering_method
    SEMANTIC_EMB_MODEL = args.semantic_emb_model
    L1_SEARCH_RANGE = range(args.l1_min, args.l1_max + 1)
    L0_SEARCH_RANGE = range(args.l0_min, args.l0_max + 1)
    configure_output_paths(args.data_dir)
    main()
