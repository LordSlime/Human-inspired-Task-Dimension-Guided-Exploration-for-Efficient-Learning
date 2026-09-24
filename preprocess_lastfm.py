import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from scipy.sparse import csr_matrix
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
import os
import pickle
import re
import argparse
from tqdm import tqdm
from collections import defaultdict
from item_cluster_bandit import build_item_space_clusters, select_item_space_k_by_kgs

# Try importing SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception as exc:
    SBERT_AVAILABLE = False
    print(f"Warning: sentence-transformers unavailable ({exc}). Semantic features will not be generated.")

# ================= Configuration =================
# [Global]
TOP_ARTISTS = 10000
TOP_USERS = 2000
FEATURE_DIM = 50        
RANDOM_SEED = 42
MIN_TAG_FREQ = 5         

# [Baseline Specific]
BASELINE_TAG_DIM = 50 

# [New Algorithm Specific] (Semantic & Hierarchical)
SEMANTIC_EMB_MODEL = 'sentence-transformers/all-mpnet-base-v2' 
ARTIST_TOPK_TAGS_FOR_SEMANTIC = 10 


L1_SEARCH_RANGE = range(2, 10)  # Level 1 

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

# L0_SEARCH_RANGE = range(10, 150)  # Level 0 

# Paths
DATA_DIR = "./datasets/lastfm_data"
FM_DIR = "./datasets/lastfm" 

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# --- Output Files: BASELINE (Variance + SVD) ---
USER_EMB_FILE = f"{DATA_DIR}/user_features.npy"           
ARTIST_EMB_FILE = f"{DATA_DIR}/artist_features.npy"         
RATING_MATRIX_FILE = f"{DATA_DIR}/rating_matrix.npy" 
ARTIST_ID_FILE = f"{DATA_DIR}/artist_ids.npy"
ARTIST_IDX_FILE = f"{DATA_DIR}/artist_idx.pkl"
CLUSTER_FILE_BASELINE = f"{DATA_DIR}/cluster_features.npy" 
CLUSTER_DICT_BASELINE = f"{DATA_DIR}/cluster_artist_dict.pkl"

# --- Output Files: NEW ALGORITHM (Semantic + Hierarchical) ---
SEMANTIC_TAG_EMB_FILE = f"{DATA_DIR}/semantic_tag_embeddings.npy"     
SEMANTIC_ARTIST_EMB_FILE = f"{DATA_DIR}/semantic_artist_features.npy"   
SEMANTIC_USER_EMB_FILE = f"{DATA_DIR}/semantic_user_features.npy"     
TAG2CLUSTER_FILE = f"{DATA_DIR}/tag2clusters.csv"                     
ARTIST_CLUSTERS_FILE = f"{DATA_DIR}/artist_multi_level_clusters.csv"
SEMANTIC_CLUSTER_FILE = f"{DATA_DIR}/semantic_cluster_features.npy"
SEMANTIC_CLUSTER_DICT_FILE = f"{DATA_DIR}/semantic_cluster_artist_dict.pkl"
SEMANTIC_CLUSTER_L1_FILE = f"{DATA_DIR}/semantic_cluster_l1_features.npy"
ITEM_SPACE_CLUSTER_FILE = f"{DATA_DIR}/item_space_cluster_features.npy"
ITEM_SPACE_CLUSTER_DICT_FILE = f"{DATA_DIR}/item_space_cluster_artist_dict.pkl"
ITEM_SPACE_ITEM_CLUSTERS_FILE = f"{DATA_DIR}/item_space_artist_clusters.csv"
ITEM_SPACE_KGS_FILE = f"{DATA_DIR}/item_space_kgs_scores.csv"
DENDROGRAM_FILE = f"{DATA_DIR}/dendrogram.png"
KGS_PLOT_FILE = f"{DATA_DIR}/optimal_k_search_kgs.png" 
ITEM_SPACE_KGS_PLOT_FILE = f"{DATA_DIR}/optimal_k_search_item_space_kgs.png"

np.random.seed(RANDOM_SEED)

def configure_output_paths(data_dir):
    """Update output paths so different clustering methods can be saved separately."""
    global DATA_DIR
    global USER_EMB_FILE, ARTIST_EMB_FILE, RATING_MATRIX_FILE, ARTIST_ID_FILE, ARTIST_IDX_FILE
    global CLUSTER_FILE_BASELINE, CLUSTER_DICT_BASELINE
    global SEMANTIC_TAG_EMB_FILE, SEMANTIC_ARTIST_EMB_FILE, SEMANTIC_USER_EMB_FILE
    global TAG2CLUSTER_FILE, ARTIST_CLUSTERS_FILE, SEMANTIC_CLUSTER_FILE
    global SEMANTIC_CLUSTER_DICT_FILE, SEMANTIC_CLUSTER_L1_FILE
    global ITEM_SPACE_CLUSTER_FILE, ITEM_SPACE_CLUSTER_DICT_FILE
    global ITEM_SPACE_ITEM_CLUSTERS_FILE, ITEM_SPACE_KGS_FILE
    global DENDROGRAM_FILE, KGS_PLOT_FILE, ITEM_SPACE_KGS_PLOT_FILE

    DATA_DIR = data_dir
    os.makedirs(DATA_DIR, exist_ok=True)

    USER_EMB_FILE = f"{DATA_DIR}/user_features.npy"
    ARTIST_EMB_FILE = f"{DATA_DIR}/artist_features.npy"
    RATING_MATRIX_FILE = f"{DATA_DIR}/rating_matrix.npy"
    ARTIST_ID_FILE = f"{DATA_DIR}/artist_ids.npy"
    ARTIST_IDX_FILE = f"{DATA_DIR}/artist_idx.pkl"
    CLUSTER_FILE_BASELINE = f"{DATA_DIR}/cluster_features.npy"
    CLUSTER_DICT_BASELINE = f"{DATA_DIR}/cluster_artist_dict.pkl"

    SEMANTIC_TAG_EMB_FILE = f"{DATA_DIR}/semantic_tag_embeddings.npy"
    SEMANTIC_ARTIST_EMB_FILE = f"{DATA_DIR}/semantic_artist_features.npy"
    SEMANTIC_USER_EMB_FILE = f"{DATA_DIR}/semantic_user_features.npy"
    TAG2CLUSTER_FILE = f"{DATA_DIR}/tag2clusters.csv"
    ARTIST_CLUSTERS_FILE = f"{DATA_DIR}/artist_multi_level_clusters.csv"
    SEMANTIC_CLUSTER_FILE = f"{DATA_DIR}/semantic_cluster_features.npy"
    SEMANTIC_CLUSTER_DICT_FILE = f"{DATA_DIR}/semantic_cluster_artist_dict.pkl"
    SEMANTIC_CLUSTER_L1_FILE = f"{DATA_DIR}/semantic_cluster_l1_features.npy"
    ITEM_SPACE_CLUSTER_FILE = f"{DATA_DIR}/item_space_cluster_features.npy"
    ITEM_SPACE_CLUSTER_DICT_FILE = f"{DATA_DIR}/item_space_cluster_artist_dict.pkl"
    ITEM_SPACE_ITEM_CLUSTERS_FILE = f"{DATA_DIR}/item_space_artist_clusters.csv"
    ITEM_SPACE_KGS_FILE = f"{DATA_DIR}/item_space_kgs_scores.csv"
    DENDROGRAM_FILE = f"{DATA_DIR}/dendrogram.png"
    KGS_PLOT_FILE = f"{DATA_DIR}/optimal_k_search_kgs.png"
    ITEM_SPACE_KGS_PLOT_FILE = f"{DATA_DIR}/optimal_k_search_item_space_kgs.png"


def preprocess_tag(tag):
    if not isinstance(tag, str): return ""
    return re.sub(r'\s+', '_', tag.strip()).lower()

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

def process_baselines(top_artists, top_users, artist_id_map, user_id_map, artists_df, ratings_sample, tag_scores):
    print("\n=== Generating Baseline Features (Variance & SVD) ===")
    
    print("1. Computing Baseline Artist Features (Tag Variance)...")
    tag_scores_sample = tag_scores[tag_scores['artistId'].isin(top_artists)].copy()
    tag_variances = tag_scores_sample.groupby("tagId")["relevance"].var()
    top_20_tags = tag_variances.nlargest(BASELINE_TAG_DIM).index.tolist()
    
    scores_filtered = tag_scores_sample[tag_scores_sample['tagId'].isin(top_20_tags)]
    p_matrix_df = scores_filtered.pivot(index='artistId', columns='tagId', values='relevance')
    p_matrix_df = p_matrix_df.reindex(top_artists, fill_value=0.0)
    artist_features_baseline = p_matrix_df.values 
    
    artist_features_baseline = np.nan_to_num(artist_features_baseline)
    artist_features_baseline = normalize(artist_features_baseline, axis=1)
    
    np.save(ARTIST_EMB_FILE, artist_features_baseline)
    print(f"Saved Baseline Artist Features (Normalized): {artist_features_baseline.shape}")

    print("2. Computing Baseline User Features (SVD)...")
    row = ratings_sample['userId'].map(user_id_map).values
    col = ratings_sample['artistId'].map(artist_id_map).values
    data = ratings_sample['rating'].values
    
    interaction_matrix = csr_matrix((data, (row, col)), shape=(len(top_users), len(top_artists)))
    svd = TruncatedSVD(n_components=FEATURE_DIM, random_state=RANDOM_SEED)
    user_features_baseline = svd.fit_transform(interaction_matrix)
    
    user_features_baseline = normalize(user_features_baseline, axis=1)
    np.save(USER_EMB_FILE, user_features_baseline)
    print(f"Saved Baseline User Features (Normalized): {user_features_baseline.shape}")

    print("3. Computing Baseline Clusters (Pseudo-Genres)...")
    artists_subset = artists_df[artists_df['artistId'].isin(top_artists)].set_index('artistId').reindex(top_artists)
    
    all_genres = set()
    for g_str in artists_subset['genres'].dropna():
        for g in g_str.split('|'):
            all_genres.add(g)
    
    ignore = {'(no genres listed)'}
    target_genres = sorted(list(all_genres - ignore))[:19] 
    genre2id = {g: i for i, g in enumerate(target_genres)}
    
    cluster_ids = []
    cluster_artist_dict = {i: [] for i in range(len(target_genres))}
    
    for aid in top_artists:
        gs = artists_subset.loc[aid, 'genres'].split('|')
        assigned = 0 
        for g in gs:
            if g in genre2id:
                assigned = genre2id[g]
                break
        cluster_ids.append(assigned)
        cluster_artist_dict[assigned].append(aid)
        
    cluster_feats = []
    c_ids_arr = np.array(cluster_ids)
    for cid in range(len(target_genres)):
        idx = np.where(c_ids_arr == cid)[0]
        if len(idx) > 0:
            cluster_feats.append(np.mean(artist_features_baseline[idx], axis=0))
        else:
            cluster_feats.append(np.zeros(BASELINE_TAG_DIM))
    
    cluster_feats = normalize(np.array(cluster_feats), axis=1)
            
    np.save(CLUSTER_FILE_BASELINE, cluster_feats)
    with open(CLUSTER_DICT_BASELINE, "wb") as f:
        pickle.dump(cluster_artist_dict, f)
    print("Saved Baseline Cluster Features (Normalized).")

def process_semantic_hierarchical(top_artists, top_users, artist_id_map, user_id_map, ratings_sample, tag_scores, artists_df, tags_df):
    print("\n=== Generating Semantic & Hierarchical Features (New Algo) ===")
    
    if not SBERT_AVAILABLE:
        print("Error: SBERT not available. Skipping.")
        return

    print("1. Cleaning Tags...")
    tags_df['clean_tag'] = tags_df['tag'].apply(preprocess_tag)
    
    print(f"2. Encoding Tags with SBERT [{SEMANTIC_EMB_MODEL}]...")
    model = SentenceTransformer(SEMANTIC_EMB_MODEL)

    # prompt_template = "This is a tag made by users about music artists and their music: {}"
    prompt_template = "{}"
    tag_texts = [prompt_template.format(tag) for tag in tags_df['clean_tag']]

    tag_embs_768 = model.encode(tag_texts, show_progress_bar=True)
    tag_embs_768 = normalize(tag_embs_768, axis=1)
    
    print(f"3. Performing Clustering on Tags [{CLUSTERING_METHOD}] (using raw SBERT embeddings)...")

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
        'tag_id': tags_df['tagId'],
        'tag_index': range(len(tags_df)), 
        'tag': tags_df['tag'],
        'clean_tag': tags_df['clean_tag'],
        'lvl0_cluster': clusters_fine,   
        'lvl1_cluster': clusters_coarse  
    })
    tag2clusters.to_csv(TAG2CLUSTER_FILE, index=False)
    
    print("4. Computing Raw Artist Embeddings (768 dim) by Aggregation...")
    scores_filtered = tag_scores[tag_scores['artistId'].isin(top_artists)]
    grouped = scores_filtered.groupby('artistId')
    
    artist_embs_768_list = []
    tag_id_to_idx = {tid: i for i, tid in enumerate(tags_df['tagId'])}
    tag_id_to_info = tag2clusters.set_index('tag_id').to_dict('index')
    sorted_aids = sorted(artist_id_map.keys(), key=lambda k: artist_id_map[k])
    
    artist_title_map = dict(zip(artists_df['artistId'], artists_df['title']))

    for aid in tqdm(sorted_aids, desc="Aggregating Tags"):
        if aid in grouped.groups:
            group = grouped.get_group(aid)
            top_k_emb = group.nlargest(50, 'relevance')
            
            t_idxs = [tag_id_to_idx[tid] for tid in top_k_emb['tagId']]
            rels = top_k_emb['relevance'].values
            
            if rels.sum() > 0:
                weights = rels / rels.sum()
                avg_emb = np.average(tag_embs_768[t_idxs], axis=0, weights=weights)
            else:
                avg_emb = np.zeros(tag_embs_768.shape[1])
        else:
            avg_emb = np.zeros(tag_embs_768.shape[1])
            
        artist_embs_768_list.append(avg_emb)
        
    artist_embs_768 = np.array(artist_embs_768_list)
    
    print(f"5. Reducing Artist Embeddings (PCA 768->{FEATURE_DIM})...")
    pca = PCA(n_components=FEATURE_DIM, random_state=RANDOM_SEED)
    artist_features_20 = pca.fit_transform(artist_embs_768)
    artist_features_20 = normalize(artist_features_20, axis=1)
    np.save(SEMANTIC_ARTIST_EMB_FILE, artist_features_20)
    
    print("6. Projecting Tags and Computing Cluster Centroids in reduced space...")
    tag_embs_20 = pca.transform(tag_embs_768)
    tag_embs_20 = normalize(tag_embs_20, axis=1)
    np.save(SEMANTIC_TAG_EMB_FILE, tag_embs_20)
    
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
    np.save(SEMANTIC_CLUSTER_FILE, cluster_centers_l0)
    
    cluster_centers_l1 = compute_centroids(OPTIMAL_NUM_L1, clusters_coarse, tag_embs_20)
    np.save(SEMANTIC_CLUSTER_L1_FILE, cluster_centers_l1)

    print("8. Generating Artist-Cluster Mappings...")
    artist_cluster_rows = []
    semantic_cluster_artist_dict = defaultdict(list)
    
    for aid in tqdm(sorted_aids, desc="Mapping"):
        if aid in grouped.groups:
            group = grouped.get_group(aid)
            top_k_map = group.nlargest(ARTIST_TOPK_TAGS_FOR_SEMANTIC, 'relevance')
            
            seen_clusters = set()
            for _, row in top_k_map.iterrows():
                tid = int(row['tagId'])
                if tid in tag_id_to_info:
                    info = tag_id_to_info[tid]
                    lvl0 = info['lvl0_cluster']
                    
                    artist_cluster_rows.append({
                        'artistId': aid,
                        'title': artist_title_map.get(aid, "Unknown"),
                        'tag_id': tid,
                        'tag': info['clean_tag'],
                        'tag_index': info['tag_index'],
                        'relevance': row['relevance'],
                        'lvl0_cluster': lvl0,
                        'lvl1_cluster': info['lvl1_cluster']
                    })
                    
                    if lvl0 not in seen_clusters:
                        semantic_cluster_artist_dict[lvl0].append(aid)
                        seen_clusters.add(lvl0)
                        
    pd.DataFrame(artist_cluster_rows).to_csv(ARTIST_CLUSTERS_FILE, index=False)
    with open(SEMANTIC_CLUSTER_DICT_FILE, "wb") as f:
        pickle.dump(dict(semantic_cluster_artist_dict), f)


def process_item_space_clusters(artist_ids):
    print("\n=== Generating Artist/Item-Space Clusters for ItemCluster Ablation ===")

    if not os.path.exists(SEMANTIC_ARTIST_EMB_FILE):
        print("Warning: Semantic artist features not found. Skipping item-space clusters.")
        return

    artist_features = np.load(SEMANTIC_ARTIST_EMB_FILE)
    num_clusters, k_vals, kgs_scores, best_idx, valid_flags = select_item_space_k_by_kgs(
        artist_features,
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

    print(f"Clustering {artist_features.shape[0]} artists into {num_clusters} item-space clusters...")
    _, cluster_centroids, cluster_artist_dict, artist_to_clusters = build_item_space_clusters(
        artist_features,
        num_clusters=int(num_clusters),
        random_state=RANDOM_SEED,
    )
    np.save(ITEM_SPACE_CLUSTER_FILE, cluster_centroids)
    with open(ITEM_SPACE_CLUSTER_DICT_FILE, "wb") as f:
        pickle.dump(cluster_artist_dict, f)

    rows = [
        {
            "artistId": int(artist_ids[item_idx]),
            "item_index": int(item_idx),
            "item_cluster": int(artist_to_clusters[item_idx][0]),
        }
        for item_idx in range(len(artist_ids))
    ]
    pd.DataFrame(rows).to_csv(ITEM_SPACE_ITEM_CLUSTERS_FILE, index=False)

    sizes = pd.Series([artist_to_clusters[i][0] for i in range(len(artist_ids))]).value_counts().sort_index().tolist()
    print(f"Saved item-space cluster features: {cluster_centroids.shape}")
    print(f"Saved item-space artist mapping. Cluster sizes: {sizes}")


def main():
    print("--- Starting Dual-Track Preprocessing for Last.fm ---")

    print("Loading Raw Data...")
    artists = pd.read_csv(f"{FM_DIR}/artists.dat", sep='\t', usecols=['id', 'name'])
    artists.rename(columns={'id': 'artistId', 'name': 'title'}, inplace=True)
    
    ratings = pd.read_csv(f"{FM_DIR}/user_artists.dat", sep='\t')
    ratings.rename(columns={'userID': 'userId', 'artistID': 'artistId', 'weight': 'rating'}, inplace=True)
    
    print("Normalizing play counts per user...")
    ratings['rating'] = np.log1p(ratings['rating']) 
    
    user_max_ratings = ratings.groupby('userId')['rating'].transform('max')
    ratings['rating'] = (ratings['rating'] / user_max_ratings.replace(0, 1)) * 0.9 + 0.1
    
    tags_df = pd.read_csv(f"{FM_DIR}/tags.dat", sep='\t', encoding='latin-1', usecols=['tagID', 'tagValue'])
    tags_df.rename(columns={'tagID': 'tagId', 'tagValue': 'tag'}, inplace=True)
    
    user_taggedartists = pd.read_csv(f"{FM_DIR}/user_taggedartists.dat", sep='\t')
    user_taggedartists.rename(columns={'userID': 'userId', 'artistID': 'artistId', 'tagID': 'tagId'}, inplace=True)
    
    print(f"Filtering rare tags (frequency <= {MIN_TAG_FREQ})...")
    tag_freq = user_taggedartists['tagId'].value_counts()
    valid_tags = tag_freq[tag_freq > MIN_TAG_FREQ].index
    
    user_taggedartists = user_taggedartists[user_taggedartists['tagId'].isin(valid_tags)].copy()
    tags_df = tags_df[tags_df['tagId'].isin(valid_tags)].copy()
    print(f"Tags remaining after filter: {len(valid_tags)} / {len(tag_freq)}")

    print("Simulating genome-scores from user_taggedartists...")
    tag_counts = user_taggedartists.groupby(['artistId', 'tagId']).size().reset_index(name='count')
    tag_counts['relevance'] = 1.0  
    tag_scores = tag_counts[['artistId', 'tagId', 'relevance']]
    
    print("Generating pseudo-genres for baseline clustering...")
    top_global_tags_idx = user_taggedartists['tagId'].value_counts().nlargest(19).index
    tag_id_to_name = dict(zip(tags_df['tagId'], tags_df['tag']))
    
    artist_genres = {}
    for aid, group in user_taggedartists.groupby('artistId'):
        artist_tag_counts = group['tagId'].value_counts()
        assigned_genres = [tag_id_to_name.get(tid, str(tid)) for tid in artist_tag_counts.index if tid in top_global_tags_idx]
        if assigned_genres:
            artist_genres[aid] = "|".join(assigned_genres[:3])
        else:
            artist_genres[aid] = "(no genres listed)"
    artists['genres'] = artists['artistId'].map(artist_genres).fillna("(no genres listed)")

    print(f"Filtering Top {TOP_ARTISTS} Artists & {TOP_USERS} Users...")
    
    artists_with_tags = set(tag_scores['artistId'].unique())
    artist_counts = ratings.groupby("artistId")['rating'].count()
    valid_artist_counts = artist_counts[artist_counts.index.isin(artists_with_tags)]
    top_artists_list = valid_artist_counts.nlargest(TOP_ARTISTS).index.tolist()
    
    top_users_list = ratings.groupby("userId")['rating'].count().nlargest(TOP_USERS).index.tolist()
    
    ratings_sample = ratings[
        (ratings['artistId'].isin(top_artists_list)) & 
        (ratings['userId'].isin(top_users_list))
    ].copy()
    
    artist_ids = np.array(top_artists_list)
    artist_id_map = {aid: i for i, aid in enumerate(artist_ids)}
    user_id_map = {uid: i for i, uid in enumerate(top_users_list)}
    
    np.save(ARTIST_ID_FILE, artist_ids)
    with open(ARTIST_IDX_FILE, "wb") as f:
        pickle.dump(artist_id_map, f)
        
    print("Creating Rating Matrix...")
    row = ratings_sample['userId'].map(user_id_map).values
    col = ratings_sample['artistId'].map(artist_id_map).values
    data = ratings_sample['rating'].values
    R_replay = np.full((len(top_users_list), len(top_artists_list)), np.nan, dtype=np.float32)
    R_replay[row, col] = data 
    np.save(RATING_MATRIX_FILE, R_replay)

    process_baselines(top_artists_list, top_users_list, artist_id_map, user_id_map, artists, ratings_sample, tag_scores)
    process_semantic_hierarchical(top_artists_list, top_users_list, artist_id_map, user_id_map, ratings_sample, tag_scores, artists, tags_df)
    process_item_space_clusters(artist_ids)

    print("\nAll Preprocessing Complete.")

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
