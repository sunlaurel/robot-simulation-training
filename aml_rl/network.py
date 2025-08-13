import torch
from torch.distributions import Independent
from torch.distributions import Normal
import torch.nn.functional as F
import torch.nn as nn

policy_size = 32
value_size = 128

# Policy network - Predicts actions distribution from the current state
class Policy(nn.Module):
  def __init__(self, state_size, action_size):
    super(Policy, self).__init__()
    self.fc1 = nn.Linear(state_size, policy_size)
    self.fc2 = nn.Linear(policy_size, policy_size)
    self.fc3 = nn.Linear(policy_size, action_size)
    self.std_dev_v = nn.Parameter(torch.ones(action_size)) # Trainable standard deviation (will be softplus transformed to ensure positivity)

  # Returns action distribution
  def forward(self, x):
    x = torch.tanh(self.fc1(x))
    x = torch.tanh(self.fc2(x))
    x = self.fc3(x)
    x = torch.tanh(x)
    action_mean = x
    action_distrib = Independent(Normal(action_mean, F.softplus(self.std_dev_v)), 1)
    return action_distrib


# Value network - Predicts state value from the current state
class Value(nn.Module):
  def __init__(self, state_size):
    super(Value, self).__init__()
    self.fc1 = nn.Linear(state_size, value_size)
    self.fc2 = nn.Linear(value_size, value_size)
    self.fc3 = nn.Linear(value_size, 1)

  def forward(self, x):
    x = torch.tanh(self.fc1(x))
    x = torch.tanh(self.fc2(x))
    x = self.fc3(x)
    return x