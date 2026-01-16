import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
    
class TD3Controller(nn.Module):
    def __init__(self, input=2, output=2):
        super().__init__()
        self.linear1 = nn.Linear(input, 16)
        self.linear2 = nn.Linear(16, 16)
        self.fc= nn.Linear(16, output)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.fc(x))

        return x
    
    def test_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action, _ = self.forward(state)
        return action.detach().numpy()[0]
    

class ControlValue(nn.Module):
    def __init__(self, state_dim=2, action_dim=2):
        super().__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, 16)
        self.linear2 = nn.Linear(16, 8)
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.fc(x)

        return x
    

class SACLookaheadPlanner(nn.Module):
    def __init__(self, input=10):
        super().__init__()
        self.linear1 = nn.Linear(input, 16)
        self.linear2 = nn.Linear(16, 16)
        self.mean_layer = nn.Linear(16, 1)
        self.std_layer = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        mean = self.mean_layer(x)
        log_std = self.std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()

        normal = D.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)# .sum(-1, keepdim=True)

        return action, log_prob, torch.tanh(mean)

class LDValue(nn.Module):
    def __init__(self, input=11, output=1):
        super().__init__()
        self.linear1 = nn.Linear(input, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 32)
        self.fc = nn.Linear(32, output)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.fc(x)

        return x
    
    
class EndToEndPolicy(nn.Module):
    def __init__(self, input=42, output=2):
        super().__init__()
        self.linear1 = nn.Linear(input, 32)
        self.linear2 = nn.Linear(32, 32)
        self.fc = nn.Linear(32, output)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear2(x))
        x = self.fc(x)

        return x
    