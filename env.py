import numpy as np

class Movielens:
    def __init__(self, user_features, movie_features, ratings_matrix, movie_ids, cluster_features=None, cluster_movie_dict=None):
        self.user_features = user_features
        self.movie_features = movie_features
        self.rating_matrix = ratings_matrix  
        self.movie_ids = movie_ids
        
        self.movie_idx_map = {mid: i for i, mid in enumerate(movie_ids)}
        self.num_users = user_features.shape[0]
        self.num_movies = movie_features.shape[0]
        
        # Hierarchical Settings
        self.cluster_features = cluster_features
        self.cluster_movie_dict = cluster_movie_dict
        self.num_clusters = cluster_features.shape[0] if cluster_features is not None else 0
        
        self.current_user = None

    def reset(self, user_idx=None):
        if user_idx is None:
            self.current_user = np.random.randint(0, self.num_users)
        else:
            self.current_user = user_idx
        return self.user_features[self.current_user]

    def get_user_valid_items(self):
        if self.current_user is None: raise ValueError("Reset first")
        user_ratings = self.rating_matrix[self.current_user]
        return np.where(~np.isnan(user_ratings))[0]

    def _construct_paper_context(self, user_emb, item_embs):
        if item_embs.ndim == 1:
            item_embs = item_embs.reshape(1, -1)
            
        N = item_embs.shape[0]
        user_part = np.tile(user_emb, (N, 1))
        
        x_tilde = np.concatenate([user_part, item_embs], axis=1)
        
        constant_col = np.ones((N, 1)) * 0.01
        x_with_bias = np.concatenate([x_tilde, constant_col], axis=1)
        
        norms = np.linalg.norm(x_with_bias, axis=1, keepdims=True)
        return x_with_bias / (norms + 1e-9)

    def step_cluster(self):
        if self.current_user is None: raise ValueError("Reset first")
        if self.cluster_features is None: raise ValueError("Cluster features not initialized")
        
        user_emb = self.user_features[self.current_user]
        return self._construct_paper_context(user_emb, self.cluster_features)

    def step_movie(self, movie_indices=None):
        if self.current_user is None: raise ValueError("Reset first")
        
        if movie_indices is None or len(movie_indices) == 0: 
            return np.array([]), np.array([])
            
        item_embs = self.movie_features[movie_indices]
        context = self._construct_paper_context(self.user_features[self.current_user], item_embs)
        rewards = self.rating_matrix[self.current_user, movie_indices]
        
        return context, rewards