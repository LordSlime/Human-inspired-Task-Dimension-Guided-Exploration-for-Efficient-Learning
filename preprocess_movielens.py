import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, AgglomerativeClustering
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
TOP_MOVIES = 10000   
TOP_USERS = 5000
FEATURE_DIM = 50         
RANDOM_SEED = 42

# [Baseline Specific]
BASELINE_TAG_DIM = 50 # Original paper uses top 20 tags by variance

# [New Algorithm Specific] (Semantic & Clustering)
SEMANTIC_EMB_MODEL = 'sentence-transformers/all-mpnet-base-v2'
MOVIE_TOPK_TAGS_FOR_SEMANTIC = 10 

L1_SEARCH_RANGE = range(2, 4)     # Level 1 
L0_SEARCH_RANGE = range(10, 100)  # Level 0 

# [Clustering Algorithm Selection]
# Supported values:
#   'hierarchical'     - Agglomerative / Ward linkage (original, also plots dendrogram)
#   'kmeans'           - K-Means (fast, general purpose)
#   'minibatch_kmeans' - Mini-Batch K-Means (much faster for large tag sets)
#   'gmm'              - Gaussian Mixture Model (soft clustering via EM)
#   'spectral'         - Spectral Clustering (good for non-convex clusters; slow on large data)
#   'random'           - Shuffle hierarchical feature/tag cluster assignments; preserves K and cluster sizes.
CLUSTERING_METHOD = 'hierarchical'

# Paths
DATA_DIR = "./datasets/ml-20m_data"
PLOT_DIR = "./paper_figures"
ML_DIR = "./datasets/ml-20m"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Output Files: BASELINE (Variance + SVD) ---
USER_EMB_FILE = f"{DATA_DIR}/user_features.npy"           
MOVIE_EMB_FILE = f"{DATA_DIR}/movie_features.npy"         
RATING_MATRIX_FILE = f"{DATA_DIR}/rating_matrix.npy" 
MOVIE_ID_FILE = f"{DATA_DIR}/movie_ids.npy"
MOVIE_IDX_FILE = f"{DATA_DIR}/movie_idx.pkl"
CLUSTER_FILE_BASELINE = f"{DATA_DIR}/cluster_features.npy" 
CLUSTER_DICT_BASELINE = f"{DATA_DIR}/cluster_movie_dict.pkl"

# --- Output Files: NEW ALGORITHM (Semantic + Hierarchical) ---
SEMANTIC_TAG_EMB_FILE = f"{DATA_DIR}/semantic_tag_embeddings.npy"     
SEMANTIC_MOVIE_EMB_FILE = f"{DATA_DIR}/semantic_movie_features.npy"   
SEMANTIC_USER_EMB_FILE = f"{DATA_DIR}/semantic_user_features.npy"     
TAG2CLUSTER_FILE = f"{DATA_DIR}/tag2clusters.csv"                     
MOVIE_CLUSTERS_FILE = f"{DATA_DIR}/movie_multi_level_clusters.csv"
SEMANTIC_CLUSTER_FILE = f"{DATA_DIR}/semantic_cluster_features.npy"
SEMANTIC_CLUSTER_DICT_FILE = f"{DATA_DIR}/semantic_cluster_movie_dict.pkl"
SEMANTIC_CLUSTER_L1_FILE = f"{DATA_DIR}/semantic_cluster_l1_features.npy"
SEMANTIC_DIMENSION_HIERARCHY_FILE = f"{DATA_DIR}/semantic_dimension_hierarchy.pkl"
ITEM_SPACE_CLUSTER_FILE = f"{DATA_DIR}/item_space_cluster_features.npy"
ITEM_SPACE_CLUSTER_DICT_FILE = f"{DATA_DIR}/item_space_cluster_movie_dict.pkl"
ITEM_SPACE_MOVIE_CLUSTERS_FILE = f"{DATA_DIR}/item_space_movie_clusters.csv"
ITEM_SPACE_KGS_FILE = f"{DATA_DIR}/item_space_kgs_scores.csv"
DENDROGRAM_FILE = f"{DATA_DIR}/dendrogram.png"
DENDROGRAM_PARTIAL_FILE = f"{PLOT_DIR}/partial_dendrogram.png"
KGS_PLOT_FILE = f"{PLOT_DIR}/optimal_k_search_kgs.png"
ITEM_SPACE_KGS_PLOT_FILE = f"{PLOT_DIR}/optimal_k_search_item_space_kgs.png"

# 🌟 Font scale for publication figures
FONT_SCALE = 2.0

np.random.seed(RANDOM_SEED)


SUPPORTED_CLUSTERING_METHODS = {
    'hierarchical',
    'kmeans',
    'minibatch_kmeans',
    'gmm',
    'spectral',
    'random',
}


def configure_output_paths(data_dir):
    """Update output paths so clustering-method ablations can be saved separately."""
    global DATA_DIR
    global USER_EMB_FILE, MOVIE_EMB_FILE, RATING_MATRIX_FILE, MOVIE_ID_FILE, MOVIE_IDX_FILE
    global CLUSTER_FILE_BASELINE, CLUSTER_DICT_BASELINE
    global SEMANTIC_TAG_EMB_FILE, SEMANTIC_MOVIE_EMB_FILE, SEMANTIC_USER_EMB_FILE
    global TAG2CLUSTER_FILE, MOVIE_CLUSTERS_FILE, SEMANTIC_CLUSTER_FILE
    global SEMANTIC_CLUSTER_DICT_FILE, SEMANTIC_CLUSTER_L1_FILE
    global SEMANTIC_DIMENSION_HIERARCHY_FILE
    global ITEM_SPACE_CLUSTER_FILE, ITEM_SPACE_CLUSTER_DICT_FILE
    global ITEM_SPACE_MOVIE_CLUSTERS_FILE, ITEM_SPACE_KGS_FILE
    global DENDROGRAM_FILE, KGS_PLOT_FILE, ITEM_SPACE_KGS_PLOT_FILE

    DATA_DIR = data_dir
    os.makedirs(DATA_DIR, exist_ok=True)

    USER_EMB_FILE = f"{DATA_DIR}/user_features.npy"
    MOVIE_EMB_FILE = f"{DATA_DIR}/movie_features.npy"
    RATING_MATRIX_FILE = f"{DATA_DIR}/rating_matrix.npy"
    MOVIE_ID_FILE = f"{DATA_DIR}/movie_ids.npy"
    MOVIE_IDX_FILE = f"{DATA_DIR}/movie_idx.pkl"
    CLUSTER_FILE_BASELINE = f"{DATA_DIR}/cluster_features.npy"
    CLUSTER_DICT_BASELINE = f"{DATA_DIR}/cluster_movie_dict.pkl"

    SEMANTIC_TAG_EMB_FILE = f"{DATA_DIR}/semantic_tag_embeddings.npy"
    SEMANTIC_MOVIE_EMB_FILE = f"{DATA_DIR}/semantic_movie_features.npy"
    SEMANTIC_USER_EMB_FILE = f"{DATA_DIR}/semantic_user_features.npy"
    TAG2CLUSTER_FILE = f"{DATA_DIR}/tag2clusters.csv"
    MOVIE_CLUSTERS_FILE = f"{DATA_DIR}/movie_multi_level_clusters.csv"
    SEMANTIC_CLUSTER_FILE = f"{DATA_DIR}/semantic_cluster_features.npy"
    SEMANTIC_CLUSTER_DICT_FILE = f"{DATA_DIR}/semantic_cluster_movie_dict.pkl"
    SEMANTIC_CLUSTER_L1_FILE = f"{DATA_DIR}/semantic_cluster_l1_features.npy"
    SEMANTIC_DIMENSION_HIERARCHY_FILE = f"{DATA_DIR}/semantic_dimension_hierarchy.pkl"
    ITEM_SPACE_CLUSTER_FILE = f"{DATA_DIR}/item_space_cluster_features.npy"
    ITEM_SPACE_CLUSTER_DICT_FILE = f"{DATA_DIR}/item_space_cluster_movie_dict.pkl"
    ITEM_SPACE_MOVIE_CLUSTERS_FILE = f"{DATA_DIR}/item_space_movie_clusters.csv"
    ITEM_SPACE_KGS_FILE = f"{DATA_DIR}/item_space_kgs_scores.csv"
    DENDROGRAM_FILE = f"{DATA_DIR}/dendrogram.png"
    KGS_PLOT_FILE = f"{DATA_DIR}/optimal_k_search_kgs.png"
    ITEM_SPACE_KGS_PLOT_FILE = f"{DATA_DIR}/optimal_k_search_item_space_kgs.png"


def preprocess_tag(tag):
    if not isinstance(tag, str):
        return ""
    tag_clean = re.sub(r'\s+', '_', tag.strip()).lower()
    return tag_clean

def compute_wcss(embeddings, labels):
    wcss = 0.0
    for label in np.unique(labels):
        pts = embeddings[labels == label]
        if len(pts) > 0:
            wcss += np.sum((pts - np.mean(pts, axis=0))**2)
    return wcss


def shuffle_hierarchical_cluster_pairs(fine_labels, coarse_labels, random_state=RANDOM_SEED):
    """Randomize feature/tag cluster assignments while preserving HC cluster sizes."""
    fine_labels = np.asarray(fine_labels, dtype=int)
    coarse_labels = np.asarray(coarse_labels, dtype=int)
    if fine_labels.shape != coarse_labels.shape:
        raise ValueError("Fine and coarse cluster labels must have the same shape.")
    rng = np.random.default_rng(int(random_state) + len(fine_labels) * 9173)
    perm = rng.permutation(len(fine_labels))
    return fine_labels[perm].astype(int), coarse_labels[perm].astype(int)


def save_dimension_hierarchy(fine_labels, coarse_labels, output_path):
    """Persist the coarse-to-fine dimension mapping for depth ablations."""
    fine_labels = np.asarray(fine_labels, dtype=int)
    coarse_labels = np.asarray(coarse_labels, dtype=int)
    if fine_labels.shape != coarse_labels.shape:
        raise ValueError("Fine and coarse cluster labels must have the same shape.")

    parent_to_children = defaultdict(set)
    child_to_parents = defaultdict(set)
    for fine_idx, coarse_idx in zip(fine_labels, coarse_labels):
        parent_to_children[int(coarse_idx)].add(int(fine_idx))
        child_to_parents[int(fine_idx)].add(int(coarse_idx))

    strict_tree = all(len(parents) == 1 for parents in child_to_parents.values())
    hierarchy = {
        "level_names": ["lvl1_cluster", "lvl0_cluster"],
        "num_dimensions": [
            int(len(np.unique(coarse_labels))),
            int(len(np.unique(fine_labels))),
        ],
        "parent_to_children": {
            int(parent): sorted(int(child) for child in children)
            for parent, children in parent_to_children.items()
        },
        "child_to_parent": {
            int(child): int(next(iter(parents)))
            for child, parents in child_to_parents.items()
            if len(parents) == 1
        },
        "strict_tree": bool(strict_tree),
    }
    with open(output_path, "wb") as handle:
        pickle.dump(hierarchy, handle)

    relation = "strict tree" if strict_tree else "multi-parent hierarchy"
    print(
        f"Saved Semantic Dimension Hierarchy ({relation}): "
        f"{hierarchy['num_dimensions'][0]} -> {hierarchy['num_dimensions'][1]} "
        f"dimensions to {output_path}"
    )
    return hierarchy

def process_baselines(top_movies, top_users, movie_id_map, user_id_map, movies_df, ratings_sample, genome_scores):
    print("\n=== Generating Baseline Features (Variance & SVD) ===")
    
    print("1. Computing Baseline Movie Features (Genome Variance)...")
    genome_sample = genome_scores[genome_scores['movieId'].isin(top_movies)].copy()
    tag_variances = genome_sample.groupby("tagId")["relevance"].var()
    top_20_tags = tag_variances.nlargest(BASELINE_TAG_DIM).index.tolist()
    
    genome_filtered = genome_sample[genome_sample['tagId'].isin(top_20_tags)]
    p_matrix_df = genome_filtered.pivot(index='movieId', columns='tagId', values='relevance')
    p_matrix_df = p_matrix_df.reindex(top_movies, fill_value=0.0)
    movie_features_baseline = p_matrix_df.values 
    
    movie_features_baseline = np.nan_to_num(movie_features_baseline)
    movie_features_baseline = normalize(movie_features_baseline, axis=1)
    
    np.save(MOVIE_EMB_FILE, movie_features_baseline)
    print(f"Saved Baseline Movie Features (Normalized): {movie_features_baseline.shape}")

    print("2. Computing Baseline User Features (SVD)...")
    row = ratings_sample['userId'].map(user_id_map).values
    col = ratings_sample['movieId'].map(movie_id_map).values
    data = ratings_sample['rating'].values
    
    interaction_matrix = csr_matrix((data, (row, col)), shape=(len(top_users), len(top_movies)))
    svd = TruncatedSVD(n_components=FEATURE_DIM, random_state=RANDOM_SEED)
    user_features_baseline = svd.fit_transform(interaction_matrix)
    
    user_features_baseline = normalize(user_features_baseline, axis=1)
    
    np.save(USER_EMB_FILE, user_features_baseline)
    print(f"Saved Baseline User Features (Normalized): {user_features_baseline.shape}")

    print("3. Computing Baseline Clusters (Genres)...")
    movies_subset = movies_df[movies_df['movieId'].isin(top_movies)].set_index('movieId').reindex(top_movies)
    
    all_genres = set()
    for g_str in movies_subset['genres'].dropna():
        for g in g_str.split('|'):
            all_genres.add(g)
    
    ignore = {'(no genres listed)', 'IMAX'}
    target_genres = sorted(list(all_genres - ignore))[:19] 
    genre2id = {g: i for i, g in enumerate(target_genres)}
    
    cluster_ids = []
    cluster_movie_dict = {i: [] for i in range(len(target_genres))}
    
    for mid in top_movies:
        gs = movies_subset.loc[mid, 'genres'].split('|')
        assigned = 0 
        for g in gs:
            if g in genre2id:
                assigned = genre2id[g]
                break
        cluster_ids.append(assigned)
        cluster_movie_dict[assigned].append(mid)
        
    cluster_feats = []
    c_ids_arr = np.array(cluster_ids)
    for cid in range(len(target_genres)):
        idx = np.where(c_ids_arr == cid)[0]
        if len(idx) > 0:
            cluster_feats.append(np.mean(movie_features_baseline[idx], axis=0))
        else:
            cluster_feats.append(np.zeros(BASELINE_TAG_DIM))
    
    cluster_feats = normalize(np.array(cluster_feats), axis=1)
            
    np.save(CLUSTER_FILE_BASELINE, cluster_feats)
    with open(CLUSTER_DICT_BASELINE, "wb") as f:
        pickle.dump(cluster_movie_dict, f)
    print("Saved Baseline Cluster Features (Normalized).")

def process_semantic_hierarchical(top_movies, top_users, movie_id_map, user_id_map, ratings_sample, genome_scores, movies_df):
    print("\n=== Generating Semantic & Hierarchical Features (New Algo) ===")
    
    if not SBERT_AVAILABLE:
        print("Error: SBERT not available. Skipping.")
        return

    print("1. Loading and Cleaning Tags...")
    tags_df = pd.read_csv(f"{ML_DIR}/genome-tags.csv") 
    tags_df['clean_tag'] = tags_df['tag'].apply(preprocess_tag)
    
    print(f"2. Encoding Tags with SBERT [{SEMANTIC_EMB_MODEL}]...")
    model = SentenceTransformer(SEMANTIC_EMB_MODEL)
    
    # prompt_template = "This is a movie tag made by users: {}"
    prompt_template = "{}"
    tag_texts = [prompt_template.format(tag) for tag in tags_df['clean_tag']]

    tag_embs_768 = model.encode(tag_texts, show_progress_bar=True)
    tag_embs_768 = normalize(tag_embs_768, axis=1)
    
    print(f"3. Performing Clustering on Tags [{CLUSTERING_METHOD}] (using raw SBERT embeddings)...")

    # ------------------------------------------------------------------ #
    #  Pluggable Clustering Engine                                         #
    #  Supported: hierarchical | kmeans | minibatch_kmeans | gmm | spectral#
    # ------------------------------------------------------------------ #

    # -- Build linkage matrix once (also used by the random-control setting) -------
    kgs_method = 'hierarchical' if CLUSTERING_METHOD == 'random' else CLUSTERING_METHOD
    Z = None
    if kgs_method == 'hierarchical':
        print("   Building Ward linkage matrix...")
        Z = linkage(tag_embs_768, method='ward')
        if CLUSTERING_METHOD == 'random':
            print("   [Random Control] KGS and cluster sizes are inherited from hierarchical clustering; final tag assignments will be shuffled.")

    def fit_clusters(embeddings, k, method, Z=None):
        """Return 0-indexed integer cluster labels for the given embeddings and k."""
        if method not in SUPPORTED_CLUSTERING_METHODS:
            raise ValueError(
                f"Unknown CLUSTERING_METHOD: '{method}'. "
                f"Choose from: {sorted(SUPPORTED_CLUSTERING_METHODS)}."
            )
        if method == 'hierarchical':
            if Z is None:
                raise ValueError("Linkage matrix Z is required for hierarchical clustering.")
            labels = fcluster(Z, t=k, criterion='maxclust') - 1   # fcluster is 1-indexed
        elif method == 'kmeans':
            model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
            labels = model.fit_predict(embeddings)
        elif method == 'minibatch_kmeans':
            model = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_SEED,
                                    n_init=10, batch_size=1024)
            labels = model.fit_predict(embeddings)
        elif method == 'gmm':
            model = GaussianMixture(n_components=k, random_state=RANDOM_SEED,
                                    covariance_type='full', max_iter=200)
            model.fit(embeddings)
            labels = model.predict(embeddings)
        elif method == 'spectral':
            print(f"   [Warning] SpectralClustering on {len(embeddings)} points may be slow.")
            model = SpectralClustering(n_clusters=k, random_state=RANDOM_SEED,
                                       affinity='nearest_neighbors', n_neighbors=15,
                                       assign_labels='kmeans', n_jobs=-1)
            labels = model.fit_predict(embeddings)
        elif method == 'random':
            raise ValueError(
                "Random clustering is produced by shuffling hierarchical final labels; "
                "call fit_clusters with method='hierarchical' for KGS/final base labels."
            )
        else:
            raise AssertionError("Unreachable clustering branch.")
        return labels.astype(int)

    # -- Generic KGS search (works for every method) ----------------------
    def auto_find_best_k_kgs(embeddings, k_range, desc="", min_cluster_size=3,
                             method=CLUSTERING_METHOD, Z=None):
        x          = []
        wcss_list  = []
        valid_flags = []

        print(f"   Evaluating KGS Penalty for {desc} in K range "
              f"{k_range.start} to {k_range.stop-1}...")
        for k in tqdm(k_range, desc=f"Eval {desc}", leave=False):
            labels = fit_clusters(embeddings, k, method, Z)
            wcss   = compute_wcss(embeddings, labels)
            x.append(k)
            wcss_list.append(wcss)
            cluster_sizes = pd.Series(labels).value_counts()
            valid_flags.append(int(cluster_sizes.min()) >= min_cluster_size)

        x           = np.array(x)
        wcss_list   = np.array(wcss_list)
        valid_flags = np.array(valid_flags)

        wcss_norm = (wcss_list - np.min(wcss_list)) / (np.max(wcss_list) - np.min(wcss_list) + 1e-9)
        k_norm    = (x - np.min(x))              / (np.max(x)          - np.min(x)          + 1e-9)
        kgs_scores = wcss_norm + k_norm

        valid_indices = np.where(valid_flags)[0]
        if len(valid_indices) == 0:
            print(f"Warning: No valid K satisfying min_cluster_size={min_cluster_size}. Fallback.")
            best_idx = 0
        else:
            best_idx = valid_indices[np.argmin(kgs_scores[valid_indices])]

        best_k     = x[best_idx]
        best_score = kgs_scores[best_idx]
        print(f"   -> [KGS Selected] Optimal {desc} K: {best_k} (KGS Penalty: {best_score:.4f})")
        return best_k, x, kgs_scores, best_idx, valid_flags

    OPTIMAL_NUM_L1, _, _, _, _ = auto_find_best_k_kgs(
        tag_embs_768, L1_SEARCH_RANGE, desc="L1 (Macro)",
        min_cluster_size=2, method=kgs_method, Z=Z)

    OPTIMAL_NUM_L0, k_vals, kgs_vals, best_idx, valid_flags = auto_find_best_k_kgs(
        tag_embs_768, L0_SEARCH_RANGE, desc="L0 (Micro)",
        min_cluster_size=5, method=kgs_method, Z=Z)

    # -- HC-only: compute cutting distances and plot dendrogram -----------
    cut_distance_l0 = 0
    if kgs_method == 'hierarchical' and Z is not None:
        if len(Z) >= OPTIMAL_NUM_L0 and OPTIMAL_NUM_L0 > 1:
            cut_distance_l0 = (Z[-OPTIMAL_NUM_L0, 2] + Z[-(OPTIMAL_NUM_L0-1), 2]) / 2.0
        cut_threshold_l1 = (Z[-OPTIMAL_NUM_L1, 2] + Z[-(OPTIMAL_NUM_L1-1), 2]) / 2.0 \
                           if len(Z) >= OPTIMAL_NUM_L1 and OPTIMAL_NUM_L1 > 1 else 0
        cut_threshold_l0 = cut_distance_l0

        print(f"Plotting Dendrogram to {DENDROGRAM_FILE}...")
        plt.figure(figsize=(25, 12))
        dendrogram(Z, truncate_mode='level', p=8, no_labels=True,
                   color_threshold=cut_threshold_l0)
        if cut_threshold_l1 > 0:
            plt.axhline(y=cut_threshold_l1, c='red', ls='--', linewidth=3,
                        label=f'L1 Macro Cut (k={OPTIMAL_NUM_L1})')
            plt.text(0, cut_threshold_l1 * 1.02,
                     f' L1 Threshold ({OPTIMAL_NUM_L1} Clusters)',
                     color='red', fontsize=14, fontweight='bold')
        if cut_threshold_l0 > 0:
            plt.axhline(y=cut_threshold_l0, c='green', ls='-.', linewidth=3,
                        label=f'L0 Micro Cut (k={OPTIMAL_NUM_L0})')
            plt.text(0, cut_threshold_l0 * 1.02,
                     f' L0 Threshold ({OPTIMAL_NUM_L0} Clusters)',
                     color='green', fontsize=14, fontweight='bold')
        plt.legend(fontsize=14, loc='upper right')
        plt.title(
            f"Hierarchical Clustering Dendrogram - Dual Cuts "
            f"(L1={OPTIMAL_NUM_L1}, L0={OPTIMAL_NUM_L0})",
            fontsize=18, fontweight='bold')
        plt.ylabel("Ward Cophenetic Distance (Cosine)", fontsize=14)
        plt.tight_layout()
        plt.savefig(DENDROGRAM_FILE, dpi=300)
        plt.close()
    else:
        print(f"   [Info] Dendrogram skipped (not applicable for '{CLUSTERING_METHOD}').")

    # -- KGS Penalty Curve (all methods) ----------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(k_vals, kgs_vals, marker='o', markersize=3, linestyle='-',
            linewidth=2.0, color='#1f77b4', zorder=2)
    opt_k   = k_vals[best_idx]
    opt_kgs = kgs_vals[best_idx]
    ax.scatter([opt_k], [opt_kgs], color='#d62728', s=180, zorder=5,
               marker='*', edgecolors='none')
    annot_extra = f'\n(distance = {cut_distance_l0:.2f})' \
                  if kgs_method == 'hierarchical' and cut_distance_l0 > 0 else ''
    ax.annotate(f'$k^* = {opt_k}$' + annot_extra,
                xy=(opt_k, opt_kgs), xytext=(opt_k, opt_kgs + 0.12),
                fontsize=10*FONT_SCALE, fontweight='bold', color='#d62728',
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
                zorder=6)
    ax.set_xlabel("Number of Clusters ($k$)", fontsize=12*FONT_SCALE)
    ax.set_ylabel("KGS Penalty Score",         fontsize=12*FONT_SCALE)
    ax.tick_params(axis='both', which='major', labelsize=10*FONT_SCALE)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(KGS_PLOT_FILE, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✅ KGS Penalty Curve saved to {KGS_PLOT_FILE}")

    # -- Final cluster label assignment -----------------------------------
    clusters_fine = fit_clusters(tag_embs_768, OPTIMAL_NUM_L0, kgs_method, Z)
    clusters_coarse = fit_clusters(tag_embs_768, OPTIMAL_NUM_L1, kgs_method, Z)
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
    print("Saved Tag Hierarchy mapping.")
    save_dimension_hierarchy(
        clusters_fine,
        clusters_coarse,
        SEMANTIC_DIMENSION_HIERARCHY_FILE,
    )

    print("4. Computing Raw Movie Embeddings (768 dim) by Aggregation...")
    scores_filtered = genome_scores[genome_scores['movieId'].isin(top_movies)]
    grouped = scores_filtered.groupby('movieId')
    
    movie_embs_768_list = []
    tag_id_to_idx = {tid: i for i, tid in enumerate(tags_df['tagId'])}
    tag_id_to_info = tag2clusters.set_index('tag_id').to_dict('index')
    sorted_mids = sorted(movie_id_map.keys(), key=lambda k: movie_id_map[k])
    
    if 'title' in movies_df.columns:
        movie_title_map = dict(zip(movies_df['movieId'], movies_df['title']))
    else:
        movie_title_map = {mid: "Unknown" for mid in movie_id_map.keys()}

    for mid in tqdm(sorted_mids, desc="Aggregating Tags"):
        if mid in grouped.groups:
            group = grouped.get_group(mid)
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
            
        movie_embs_768_list.append(avg_emb)
        
    movie_embs_768 = np.array(movie_embs_768_list)
    
    print(f"5. Reducing Movie Embeddings (PCA 768->{FEATURE_DIM})...")
    pca = PCA(n_components=FEATURE_DIM, random_state=RANDOM_SEED)
    movie_features_20 = pca.fit_transform(movie_embs_768)
    movie_features_20 = normalize(movie_features_20, axis=1)
    np.save(SEMANTIC_MOVIE_EMB_FILE, movie_features_20)
    print(f"Saved Semantic Movie Features (Normalized): {movie_features_20.shape}")
    
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
        centers = np.array(centers)
        return normalize(centers, axis=1)

    cluster_centers_l0 = compute_centroids(OPTIMAL_NUM_L0, clusters_fine, tag_embs_20)
    np.save(SEMANTIC_CLUSTER_FILE, cluster_centers_l0)
    print(f"Saved Semantic Cluster Features (Lvl0): {cluster_centers_l0.shape}")
    
    cluster_centers_l1 = compute_centroids(OPTIMAL_NUM_L1, clusters_coarse, tag_embs_20)
    np.save(SEMANTIC_CLUSTER_L1_FILE, cluster_centers_l1)
    print(f"Saved Semantic Cluster Features (Lvl1): {cluster_centers_l1.shape}")

    print("8. Generating Movie-Cluster Mappings...")
    movie_cluster_rows = []
    semantic_cluster_movie_dict = defaultdict(list)
    
    for mid in tqdm(sorted_mids, desc="Mapping"):
        if mid in grouped.groups:
            group = grouped.get_group(mid)
            top_k_map = group.nlargest(MOVIE_TOPK_TAGS_FOR_SEMANTIC, 'relevance')
            
            seen_clusters = set()
            for _, row in top_k_map.iterrows():
                tid = int(row['tagId'])
                if tid in tag_id_to_info:
                    info = tag_id_to_info[tid]
                    lvl0 = info['lvl0_cluster']
                    
                    movie_cluster_rows.append({
                        'movieId': mid,
                        'title': movie_title_map.get(mid, "Unknown"),
                        'tag_id': tid,
                        'tag': info['clean_tag'],
                        'tag_index': info['tag_index'],
                        'relevance': row['relevance'],
                        'lvl0_cluster': lvl0,
                        'lvl1_cluster': info['lvl1_cluster']
                    })
                    
                    if lvl0 not in seen_clusters:
                        semantic_cluster_movie_dict[lvl0].append(mid)
                        seen_clusters.add(lvl0)
                        
    pd.DataFrame(movie_cluster_rows).to_csv(MOVIE_CLUSTERS_FILE, index=False)
    print("Saved Movie-Tag-Cluster mapping.")
    
    with open(SEMANTIC_CLUSTER_DICT_FILE, "wb") as f:
        pickle.dump(dict(semantic_cluster_movie_dict), f)
    print("Saved Semantic Cluster-Movie Dict.")

    # =====================================================================
    # 🌟 Partial Dendrogram: only meaningful for hierarchical clustering
    # =====================================================================
    if CLUSTERING_METHOD == 'hierarchical' and Z is not None:
        from scipy.cluster.hierarchy import cophenet
        from scipy.spatial.distance import squareform

        print(f"Plotting Partial Dendrogram (from full Z) to {DENDROGRAM_PARTIAL_FILE}...")

        rep_labels = []
        rep_indices = []   # global tag indices (into the full Z)

        # 1. Pick 10 largest L0 clusters, 4 representative tags each
        l0_counts = pd.Series(clusters_fine).value_counts()
        target_l0_cids = l0_counts.nlargest(10).index.tolist()
        TAGS_PER_CLUSTER = 4

        for cid in target_l0_cids:
            idx_in_cluster = np.where(clusters_fine == cid)[0]
            if len(idx_in_cluster) >= TAGS_PER_CLUSTER:
                sims = np.dot(tag_embs_20[idx_in_cluster], cluster_centers_l0[cid])
                top_idx_local = np.argsort(sims)[-TAGS_PER_CLUSTER:][::-1]
                top_idx_global = idx_in_cluster[top_idx_local]
                for g_idx in top_idx_global:
                    tag_text = tags_df.iloc[g_idx]['tag']
                    rep_labels.append(tag_text)
                    rep_indices.append(g_idx)

        rep_indices = np.array(rep_indices)

        # 2. Extract cophenetic distances from the full Z for these tags
        coph_condensed_full = cophenet(Z)
        coph_full = squareform(coph_condensed_full)
        coph_sub = coph_full[np.ix_(rep_indices, rep_indices)]
        coph_condensed = squareform(coph_sub)

        # 3. Complete linkage on cophenetic distances preserves original merge heights
        Z_sub = linkage(coph_condensed, method='complete')

        # 4. Plot
        plt.figure(figsize=(20, 10))
        dendro = dendrogram(
            Z_sub,
            labels=rep_labels,
            leaf_rotation=60,
            leaf_font_size=10*FONT_SCALE,
            color_threshold=cut_distance_l0,
            above_threshold_color='gray'
        )
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if cut_distance_l0 > 0:
            ax.axhline(y=cut_distance_l0, color='#d62728', linestyle='--',
                       linewidth=2.0, zorder=10, alpha=0.85)
            ax.text(ax.get_xlim()[1] * 0.98, cut_distance_l0 + 0.03,
                    f'Cutting height ($k^*\\!={OPTIMAL_NUM_L0}$)',
                    ha='right', va='bottom', fontsize=10*FONT_SCALE,
                    fontweight='bold', color='#d62728')
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=60, ha='right', rotation_mode="anchor",
            fontsize=10*FONT_SCALE, fontweight='medium'
        )
        plt.subplots_adjust(bottom=0.3)
        plt.ylabel("Distance", fontsize=15*FONT_SCALE)
        ax.tick_params(axis='y', labelsize=10*FONT_SCALE)
        plt.savefig(DENDROGRAM_PARTIAL_FILE, dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Partial Dendrogram (full-Z distances) saved.")
    else:
        print(f"   [Info] Partial dendrogram skipped (not applicable for '{CLUSTERING_METHOD}').")


def process_item_space_clusters(movie_ids):
    print("\n=== Generating Item-Space Clusters for ItemCluster Ablation ===")

    if not os.path.exists(SEMANTIC_MOVIE_EMB_FILE):
        print("Warning: Semantic movie features not found. Skipping item-space clusters.")
        return

    movie_features = np.load(SEMANTIC_MOVIE_EMB_FILE)
    num_clusters, k_vals, kgs_scores, best_idx, valid_flags = select_item_space_k_by_kgs(
        movie_features,
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
                fontsize=10*FONT_SCALE, fontweight='bold', color='#d62728',
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
                zorder=6)
    ax.set_xlabel("Number of Item Clusters ($k$)", fontsize=12*FONT_SCALE)
    ax.set_ylabel("KGS Penalty Score", fontsize=12*FONT_SCALE)
    ax.tick_params(axis='both', which='major', labelsize=10*FONT_SCALE)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(ITEM_SPACE_KGS_PLOT_FILE, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Item-space KGS curve saved to {ITEM_SPACE_KGS_PLOT_FILE}")

    print(f"Clustering {movie_features.shape[0]} movies into {num_clusters} item-space clusters...")
    _, cluster_centroids, cluster_movie_dict, movie_to_clusters = build_item_space_clusters(
        movie_features,
        num_clusters=int(num_clusters),
        random_state=RANDOM_SEED,
    )
    np.save(ITEM_SPACE_CLUSTER_FILE, cluster_centroids)
    with open(ITEM_SPACE_CLUSTER_DICT_FILE, "wb") as f:
        pickle.dump(cluster_movie_dict, f)

    rows = [
        {
            "movieId": int(movie_ids[env_idx]),
            "movie_index": int(env_idx),
            "item_cluster": int(movie_to_clusters[env_idx][0]),
        }
        for env_idx in range(len(movie_ids))
    ]
    pd.DataFrame(rows).to_csv(ITEM_SPACE_MOVIE_CLUSTERS_FILE, index=False)

    sizes = pd.Series([movie_to_clusters[i][0] for i in range(len(movie_ids))]).value_counts().sort_index().tolist()
    print(f"Saved item-space cluster features: {cluster_centroids.shape}")
    print(f"Saved item-space cluster mapping. Cluster sizes: {sizes}")


def main():
    print("--- Starting Dual-Track Preprocessing ---")

    print("Loading Raw Data...")
    movies = pd.read_csv(f"{ML_DIR}/movies.csv")
    ratings = pd.read_csv(f"{ML_DIR}/ratings.csv")
    genome_scores = pd.read_csv(f"{ML_DIR}/genome-scores.csv")
    
    print(f"Filtering Top {TOP_MOVIES} Movies & {TOP_USERS} Users...")
    
    movies_with_tags = set(genome_scores['movieId'].unique())
    print(f"Total movies in genome-scores: {len(movies_with_tags)}")
    
    movie_counts = ratings.groupby("movieId")['rating'].count()
    valid_movie_counts = movie_counts[movie_counts.index.isin(movies_with_tags)]
    top_movies = valid_movie_counts.nlargest(TOP_MOVIES).index.tolist()
    print(f"Selected Top-{len(top_movies)} movies (Guaranteed to have Genome Tags).")

    top_users = ratings.groupby("userId")['rating'].count().nlargest(TOP_USERS).index.tolist()
    
    ratings_sample = ratings[
        (ratings['movieId'].isin(top_movies)) & 
        (ratings['userId'].isin(top_users))
    ].copy()
    
    movie_ids = np.array(top_movies)
    movie_id_map = {mid: i for i, mid in enumerate(movie_ids)}
    user_id_map = {uid: i for i, uid in enumerate(top_users)}
    
    np.save(MOVIE_ID_FILE, movie_ids)
    with open(MOVIE_IDX_FILE, "wb") as f:
        pickle.dump(movie_id_map, f)
        
    print("Creating Rating Matrix...")
    row = ratings_sample['userId'].map(user_id_map).values
    col = ratings_sample['movieId'].map(movie_id_map).values
    data = ratings_sample['rating'].values
    R_replay = np.full((len(top_users), len(top_movies)), np.nan, dtype=np.float32)
    R_replay[row, col] = data / 5.0 
    np.save(RATING_MATRIX_FILE, R_replay)

    process_baselines(top_movies, top_users, movie_id_map, user_id_map, movies, ratings_sample, genome_scores)
    process_semantic_hierarchical(top_movies, top_users, movie_id_map, user_id_map, ratings_sample, genome_scores, movies)
    process_item_space_clusters(movie_ids)

    print("\nAll Preprocessing Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MovieLens with selectable semantic tag clustering method.")
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
    parser.add_argument(
        "--export_hierarchy_only",
        action="store_true",
        help=(
            "Build semantic_dimension_hierarchy.pkl from an existing "
            "tag2clusters.csv without rerunning semantic preprocessing."
        ),
    )
    args = parser.parse_args()

    CLUSTERING_METHOD = args.clustering_method
    SEMANTIC_EMB_MODEL = args.semantic_emb_model
    L1_SEARCH_RANGE = range(args.l1_min, args.l1_max + 1)
    L0_SEARCH_RANGE = range(args.l0_min, args.l0_max + 1)
    configure_output_paths(args.data_dir)
    if args.export_hierarchy_only:
        if not os.path.exists(TAG2CLUSTER_FILE):
            raise FileNotFoundError(
                f"Cannot export hierarchy: {TAG2CLUSTER_FILE} does not exist."
            )
        existing_hierarchy = pd.read_csv(TAG2CLUSTER_FILE)
        required_columns = {"lvl0_cluster", "lvl1_cluster"}
        missing_columns = required_columns - set(existing_hierarchy.columns)
        if missing_columns:
            raise ValueError(
                f"Missing hierarchy columns in {TAG2CLUSTER_FILE}: "
                f"{sorted(missing_columns)}"
            )
        save_dimension_hierarchy(
            existing_hierarchy["lvl0_cluster"].to_numpy(),
            existing_hierarchy["lvl1_cluster"].to_numpy(),
            SEMANTIC_DIMENSION_HIERARCHY_FILE,
        )
    else:
        main()
