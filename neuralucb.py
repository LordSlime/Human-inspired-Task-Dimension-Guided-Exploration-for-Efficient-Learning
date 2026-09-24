import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# Helper Class from Source 3 (NeuralUCB.py)
# =============================================================================
class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

class NeuralUCBDiag:
    def __init__(self, d, num_arm=1, beta=1.0, lamb=1.0, lr=1e-3, hidden=64, buffer_size=10000, batch_size=64, train_epochs=10):
        self.d = d
        self.beta = beta
        self.lamb = lamb
        self.lr = lr
        self.batch_size = batch_size
        self.train_epochs = train_epochs 
        
        # Neural Network
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
        self.buffer_weight = []
        self.buffer_size = buffer_size

        self.use_neural_linear = True
        self.last_layer_dim = hidden
        
        # Inverse Covariance Matrix
        self.A_inv = (1.0 / lamb) * torch.eye(self.last_layer_dim).to(device)
        self.feature_extractor = self.func[:-1]

    def _get_features(self, x_tensor):
        return self.feature_extractor(x_tensor)

    def take_action(self, context, beta=None):
        use_beta = beta if beta is not None else self.beta
        
        self.func.eval()
        x_tensor = torch.FloatTensor(context).to(device)
        
        if self.use_neural_linear:
            with torch.no_grad():
                features = self._get_features(x_tensor)
                mean = self.func(x_tensor).squeeze(-1)
            
            tmp = features @ self.A_inv
            var = torch.sum(tmp * features, dim=1)
            std = torch.sqrt(torch.clamp(var, min=1e-6))
            
            ucb = mean + use_beta * std
            return torch.argmax(ucb).item(), ucb.detach().cpu().numpy()
        else:
            with torch.no_grad():
                pred = self.func(x_tensor).squeeze(-1)
            return torch.argmax(pred).item(), pred.detach().cpu().numpy()

    def update(self, context, action, reward, sample_weight=1.0):
        self.buffer_context.append(context[action])
        self.buffer_reward.append(reward)
        self.buffer_weight.append(sample_weight)
        
        if len(self.buffer_context) > self.buffer_size:
            self.buffer_context.pop(0)
            self.buffer_reward.pop(0)
            self.buffer_weight.pop(0)
            
        self.func.train()
        current_size = len(self.buffer_context)
        total_loss = 0.0
        
        for _ in range(self.train_epochs):
            if current_size < self.batch_size:
                idx = range(current_size)
            else:
                idx = np.random.choice(current_size, self.batch_size, replace=False)
            
            batch_x = torch.FloatTensor(np.array([self.buffer_context[i] for i in idx])).to(device)
            batch_y = torch.FloatTensor(np.array([self.buffer_reward[i] for i in idx])).unsqueeze(1).to(device)
            batch_w = torch.FloatTensor(np.array([self.buffer_weight[i] for i in idx])).unsqueeze(1).to(device)
            
            self.optimizer.zero_grad()
            pred = self.func(batch_x)
            
            loss = torch.mean(batch_w * (pred - batch_y)**2)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        self.func.eval()
        x_chosen = torch.FloatTensor(context[action]).unsqueeze(0).to(device)
        w_val = float(sample_weight)
        
        with torch.no_grad():
            feat = self._get_features(x_chosen).squeeze(0).unsqueeze(1)
        
        Au = self.A_inv @ feat
        denom = 1.0 + w_val * float(feat.T @ Au)
        numerator = (Au @ Au.T) * w_val
        
        self.A_inv -= numerator / denom
        
        return total_loss / self.train_epochs

class NeuralTSDiag(NeuralUCBDiag):
    def __init__(self, d, num_arm=1, beta=1.0, lamb=1.0, lr=1e-3, hidden=64, buffer_size=10000, batch_size=64, train_epochs=10):
        super().__init__(d, num_arm, beta, lamb, lr, hidden, buffer_size, batch_size, train_epochs)
    
    def take_action(self, context, beta=None):
        use_beta = beta if beta is not None else self.beta
        self.func.eval()
        x_tensor = torch.FloatTensor(context).to(device)
        
        with torch.no_grad():
            features = self._get_features(x_tensor)
            mean = self.func(x_tensor).squeeze(-1)
        
        tmp = features @ self.A_inv
        var = torch.sum(tmp * features, dim=1)
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        
        sample = torch.normal(mean, use_beta * std)
        
        return torch.argmax(sample).item(), sample.detach().cpu().numpy()