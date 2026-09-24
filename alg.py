import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy as sp 
import scipy.linalg

# Global device definition
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class RandomAlg:
    def __init__(self, d=None):
        pass
        
    def take_action(self, context, beta=None):
        # context: (K, d)
        K = context.shape[0]
        scores = np.random.rand(K)
        return np.argmax(scores), scores
        
    def update(self, context, action, reward, sample_weight=1.0):
        pass


class LinUCB(RandomAlg):
    def __init__(self, d, beta=0.1, lamb=1.0):
        """
        d: Context dimension (e.g., 41)
        """
        self.d = d
        self.sigma_inv = (1.0 / lamb) * np.eye(d)
        self.b = np.zeros((d, 1))
        self.beta = beta
        
    def take_action(self, context, beta=None):
        """
        [Modified] Added beta parameter for evaluation override
        """
        use_beta = beta if beta is not None else self.beta

        # theta = A^-1 b
        theta = self.sigma_inv @ self.b  # (d, 1)
        
        # Mean prediction: X theta
        mean = context @ theta # (K, 1)
        
        # Variance: x^T A^-1 x
        tmp = context @ self.sigma_inv
        matrix_var = np.sum(tmp * context, axis=1, keepdims=True)
        confidence = np.sqrt(matrix_var)
        
        # UCB Score
        ucb_score = mean + use_beta * confidence
        
        return np.argmax(ucb_score), ucb_score.flatten()

    def update(self, context, action, reward, sample_weight=1.0):
        """
        [Modified] Weighted LinUCB Update (Weighted Sherman-Morrison)
        w: sample_weight (relevance)
        A_new = A_old + w * x * x.T
        b_new = b_old + w * r * x
        """
        x = context[action].reshape(-1, 1) # (d, 1)
        w = float(sample_weight)
        
        # 1. Sherman-Morrison formula for (A + wxx^T)^-1
        # Ax = A^-1 * x
        Ax = self.sigma_inv @ x 
        
        # Denominator = 1 + w * x^T * A^-1 * x
        denom = 1.0 + w * float(x.T @ Ax)
        
        # Numerator = w * (A^-1 x) (A^-1 x)^T
        numerator = (Ax @ Ax.T) * w
        
        # Update Inverse Matrix
        self.sigma_inv -= numerator / denom
        
        # 2. Update Bias Vector b
        self.b += x * reward * w

# ==========================================
# Hierarchical LinUCB Components
# ==========================================

class LinUCB_cluster(LinUCB):
    def __init__(self, d, num_actions, beta=1.0, beta_r=0.0, lamb=1.0):
        super().__init__(d, beta, lamb)
        self.beta_r = beta_r 
        
    def take_action(self, context, r_std=None, beta=None):
        use_beta = beta if beta is not None else self.beta

        theta = self.sigma_inv @ self.b
        mean = context @ theta
        
        tmp = context @ self.sigma_inv
        confidence = np.sqrt(np.sum(tmp * context, axis=1, keepdims=True))
        
        score = mean + use_beta * confidence
        
        if r_std is not None and self.beta_r > 0:
            score += self.beta_r * r_std
            
        best_idx = np.argmax(score)
        return [best_idx], confidence

class LinUCB_movie(LinUCB):
    def __init__(self, d, beta=0.1, beta_c=0.1, beta_d=0.0, lamb=1.0):
        super().__init__(d, beta, lamb)
        self.beta_c = beta_c 
        self.beta_d = beta_d 
        
    def take_action(self, context, cluster_ucb=0.0, distance=None, beta=None):
        use_beta = beta if beta is not None else self.beta

        theta = self.sigma_inv @ self.b
        mean = context @ theta
        
        tmp = context @ self.sigma_inv
        confidence = np.sqrt(np.sum(tmp * context, axis=1, keepdims=True))
        
        score = mean + use_beta * confidence
        
        # Add Cluster Uncertainty Bonus
        score += self.beta_c * cluster_ucb
        
        if distance is not None and self.beta_d != 0:
            score += self.beta_d * distance
            
        return np.argmax(score), score.flatten()

# ==========================================
# TagNeuralUCB (For LowD_HieRec Cluster Selection)
# ==========================================
class TagNeuralUCB:
    def __init__(self, d, beta=1.0, lamb=1.0, lr=1e-3, hidden=64, buffer_size=10000, batch_size=128, train_epochs=10):
        """
        Neural UCB variant for selecting Tag Clusters.
        Based on Neural Linear UCB.
        """
        self.d = d
        self.beta = beta
        self.lamb = lamb
        self.lr = lr
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.buffer_size = buffer_size
        
        # Neural Network: Feature Extractor + Linear Head
        # Semantic Input (d) -> Hidden -> Hidden -> 1
        self.func = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        ).to(device)
        
        self.optimizer = optim.Adam(self.func.parameters(), lr=lr)
        
        # Replay Buffer
        self.buffer_context = []
        self.buffer_reward = []
        self.buffer_weight = [] # [New] Store sample weights
        
        # Feature Extractor (All layers except last Linear)
        self.feature_extractor = self.func[:-1]
        
        # Matrix for Uncertainty (A_inv)
        # Using Full Matrix as hidden dim (64) is small enough for exact calculation
        self.A_inv = (1.0 / lamb) * torch.eye(hidden).to(device)
        
    def _get_features(self, x):
        # Extract features from the second to last layer
        return self.feature_extractor(x)

    def take_action(self, context, beta=None):
        """
        Predict scores for all arms (clusters) in context.
        context: (Num_Clusters, d)
        """
        use_beta = beta if beta is not None else self.beta
        
        self.func.eval()
        x_tensor = torch.FloatTensor(context).to(device)
        
        with torch.no_grad():
            features = self._get_features(x_tensor) # (K, hidden)
            mean = self.func(x_tensor).squeeze(-1)  # (K,)
            
        # UCB Calculation: mean + beta * sqrt(phi^T A_inv phi)
        tmp = features @ self.A_inv
        # Variance = sum((phi @ A_inv) * phi, dim=1)
        variance = torch.sum(tmp * features, dim=1)
        std = torch.sqrt(torch.clamp(variance, min=1e-6))
        
        ucb_score = mean + use_beta * std
        
        return torch.argmax(ucb_score).item(), ucb_score.detach().cpu().numpy()

    def update(self, context, action, reward, sample_weight=1.0):
        """
        Update NN and Matrix with Weighted Loss and Weighted Sherman-Morrison.
        """
        # Add to buffer (Store the chosen arm's context)
        chosen_context = context[action] # (d,)
        self.buffer_context.append(chosen_context)
        self.buffer_reward.append(reward)
        self.buffer_weight.append(sample_weight)
        
        if len(self.buffer_context) > self.buffer_size:
            self.buffer_context.pop(0)
            self.buffer_reward.pop(0)
            self.buffer_weight.pop(0)
            
        # 1. Train NN with Weighted MSE
        self.func.train()
        buffer_len = len(self.buffer_context)
        total_loss = 0.0
        
        # Mini-batch Training
        for _ in range(self.train_epochs):
            if buffer_len < self.batch_size:
                indices = range(buffer_len)
            else:
                indices = np.random.choice(buffer_len, self.batch_size, replace=False)
            
            batch_x = torch.FloatTensor(np.array([self.buffer_context[i] for i in indices])).to(device)
            batch_y = torch.FloatTensor(np.array([self.buffer_reward[i] for i in indices])).unsqueeze(1).to(device)
            batch_w = torch.FloatTensor(np.array([self.buffer_weight[i] for i in indices])).unsqueeze(1).to(device)
            
            self.optimizer.zero_grad()
            pred = self.func(batch_x)
            
            # Weighted MSE Loss calculation
            # Loss = mean( weight * (y_pred - y_true)^2 )
            loss = torch.mean(batch_w * (pred - batch_y) ** 2)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        # 2. Update Matrix A_inv (Weighted Sherman-Morrison with new data point)
        self.func.eval()
        x_new = torch.FloatTensor(chosen_context).unsqueeze(0).to(device)
        w_val = float(sample_weight)
        
        with torch.no_grad():
            feat = self._get_features(x_new).squeeze(0) # (hidden,)
            
        # A_inv = A_inv - (w * A_inv u u^T A_inv) / (1 + w * u^T A_inv u)
        u = feat.unsqueeze(1) # (hidden, 1)
        Au = self.A_inv @ u
        
        denom = 1.0 + w_val * float(u.T @ Au)
        numerator = (Au @ Au.T) * w_val
        
        self.A_inv -= numerator / denom
        
        return total_loss / self.train_epochs

class TagNeuralTS(TagNeuralUCB):
    def __init__(self, d, beta=1.0, lamb=1.0, lr=1e-3, hidden=64, buffer_size=10000, batch_size=128, train_epochs=10):
        super().__init__(d, beta, lamb, lr, hidden, buffer_size, batch_size, train_epochs)
        
    def take_action(self, context, beta=None):
        use_beta = beta if beta is not None else self.beta
        
        self.func.eval()
        x_tensor = torch.FloatTensor(context).to(device)
        
        with torch.no_grad():
            features = self._get_features(x_tensor) # (K, hidden)
            mean = self.func(x_tensor).squeeze(-1)  # (K,)
            
        # Thompson Sampling
        tmp = features @ self.A_inv
        variance = torch.sum(tmp * features, dim=1)
        std = torch.sqrt(torch.clamp(variance, min=1e-6))
        
        # Sample score ~ N(mean, beta * std)
        sample = torch.normal(mean, use_beta * std)
        
        return torch.argmax(sample).item(), sample.detach().cpu().numpy()
    
