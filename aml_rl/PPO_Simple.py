# Simplified Reinforcement Learning with Proximal Policy Optimization (PPO) in Python
# Contact: sjguy@umn.edu

import torch
import torch.nn.functional as F

from collections import deque

def _mean_or_nan(xs):
    return float(sum(xs) / len(xs)) if xs else float('nan')

def _std_or_nan(xs):
  mean = _mean_or_nan(xs)
  return float((sum((x - mean)**2 for x in xs) / len(xs))**0.5) if xs else float('nan')
    
class PPO():
    """Proximal Policy Optimization (See https://arxiv.org/abs/1707.06347)

    Args:
        policy (torch.nn.Module): Policy model to train (behavior policy)
        value (torch.nn.Module): Value function model to train (state-value function)
        optimizer (torch.optim.Optimizer): Optimizer used to train the model
        gpu (int): GPU device id if not None nor negative
        gamma (float): Discount factor [0, 1] (discount factor for returns)
        lambd (float): Lambda-return factor [0, 1] (discount factor for GAE)
        value_func_coef (float): Weight coefficient for loss of
            value function (0, inf)
        entropy_coef (float): Weight coefficient for entropy bonus [0, inf)
        update_interval (int): Model update interval in step
        minibatch_size (int): Minibatch size
        epochs (int): Training epochs in an update
        clip_eps (float): Epsilon for pessimistic clipping of likelihood ratio
            to update policy
        standardize_advantages (bool): Use standardized advantages on updates.
    """

    def __init__(
      self,
      policy,
      value,
      optimizer,
      gamma=0.99,
      lambd=0.95,
      value_func_coef=1.0,
      entropy_coef=0.01,
      update_interval=2048,
      minibatch_size=64,
      epochs=10,
      clip_eps=0.2,
      standardize_advantages=True,
      reward_normalization=True,
      value_stats_window=1000,
      value_loss_stats_window=100,
      policy_loss_stats_window=100,
    ):
      self.policy = policy
      self.value = value
      self.optimizer = optimizer

      self.device = torch.device("cpu")

      self.gamma = gamma
      self.lambd = lambd
      self.value_func_coef = value_func_coef
      self.entropy_coef = entropy_coef
      self.update_interval = update_interval
      self.minibatch_size = minibatch_size
      self.epochs = epochs
      self.clip_eps = clip_eps
      self.standardize_advantages = standardize_advantages
      self.reward_normalization = reward_normalization
      self.value_normalization = True

      # Contains episodes used for next update iteration
      self.memory = []

      # Contains transitions of the last episode not moved to self.memory yet
      self.last_episode = []
      self.last_state = None
      self.last_action = None

      # Batch versions of last_episode, last_state, and last_action
      self.batch_last_episode = None
      self.batch_last_state = None
      self.batch_last_action = None

      # Records of losses
      self.value_record = deque(maxlen=value_stats_window)
      # self.entropy_record = deque(maxlen=entropy_stats_window)
      self.value_loss_record = deque(maxlen=value_loss_stats_window)
      self.policy_loss_record = deque(maxlen=policy_loss_stats_window)
      self.rewards_record = deque(maxlen=10000)

      self.return_mean = 0
      self.reward_var = 1
      self.return_std = 1



      self.n_updates = 0


    def act(self, state):
      with torch.no_grad():
        state_torch = torch.tensor(state, dtype=torch.float32, device=self.device)
        action_distrib = self.policy(state_torch)
        action = action_distrib.sample().cpu().numpy()
        value = float(self.value(state_torch).cpu().detach().numpy())
        value_unnormalized = value*self.return_std + self.return_mean
        self.value_record.append(value_unnormalized)
        
        
      self.last_state = state  
      self.last_action = action
      return action
      
    def observe(self, state, action, reward, done, reset):
      if self.last_state is not None:
        # if self.reward_normalization:
        #   alpha = 0.999
        #   self.return_mean = alpha*self.return_mean + (1-alpha)*reward
        #   self.reward_var = alpha*self.reward_var + (1-alpha)*(reward - self.return_mean)**2
        #   self.return_std = self.reward_var**0.5
        #   reward = (reward - self.return_mean) / (self.return_std + 1e-8)
        self.last_episode.append((self.last_state, self.last_action, reward, state, done))
      self.last_state = state
      self.last_action = action
      if done or reset:
        assert(len(self.last_episode) > 0)
        self.memory.append(self.last_episode)
        self.last_episode = []
        self.last_state = None
        self.last_action = None

      assert((len(self.memory) == 0) or (len(self.memory[0]) != 0))

      num_steps = sum(len(episode) for episode in self.memory) + len(self.last_episode) + (1 if self.last_state is not None else 0)
      if num_steps >= self.update_interval:
        self.update()
        self.memory = []
        self.last_episode = []
        self.last_state = None
        self.last_action = None
    
    def update(self):
      # print("updating...")
      self.n_updates += 1
      
      # Prepare data
      batch_state = []
      batch_action = []
      # batch_return = []
      batch_advantage = []
      batch_log_prob = []
      batch_teacher_value = []
      for episode in self.memory:
        states = [transition[0] for transition in episode]
        actions = [transition[1] for transition in episode]
        rewards = [transition[2] for transition in episode]
        dones = [transition[4] for transition in episode]
        values = self.value(torch.tensor(states, dtype=torch.float32, device=self.device)).cpu().detach().numpy()
        next_value = self.value(torch.tensor([episode[-1][3]], dtype=torch.float32, device=self.device)).cpu().detach().numpy()
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        # returns = self.compute_returns(rewards_tensor, dones, next_value)
        advantages, teacher_values = self.compute_gae(rewards_tensor, dones, values, next_value)
        action_distrib = self.policy(torch.tensor(states, dtype=torch.float32, device=self.device))
        actions_pt = torch.tensor(actions, dtype=torch.float32, device=self.device)
        log_probs = action_distrib.log_prob(actions_pt)
        # print("log_probs: ", log_probs)
        batch_state.extend(states)
        batch_action.extend(actions)
        batch_advantage.extend(advantages)
        # batch_return.extend(returns)
        batch_log_prob.extend(log_probs)
        batch_teacher_value.extend(teacher_values)

      batch_state = torch.tensor(batch_state, dtype=torch.float32, device=self.device)
      batch_action = torch.tensor(batch_action, dtype=torch.float32, device=self.device)
      # batch_return = torch.tensor(batch_return, dtype=torch.float32, device=self.device)
      batch_advantage = torch.tensor(batch_advantage, dtype=torch.float32, device=self.device)
      batch_log_prob = torch.tensor(batch_log_prob, dtype=torch.float32, device=self.device)
      batch_teacher_value = torch.tensor(batch_teacher_value, dtype=torch.float32, device=self.device)

      # Normalize advantages
      if self.standardize_advantages:
        batch_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)
      if self.value_normalization:
        self.return_mean = batch_teacher_value.mean().item()
        self.return_std = batch_teacher_value.std().item()
        batch_teacher_value = (batch_teacher_value - self.return_mean) / (self.return_std + 1e-8)

      # Train policy
      for _ in range(self.epochs):
        action_distrib = self.policy(batch_state)
        log_prob = action_distrib.log_prob(batch_action)
        ratio = torch.exp(log_prob - batch_log_prob)
        policy_loss = -torch.min(
          ratio * batch_advantage,
          torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantage
        ).mean()
        entropy = action_distrib.entropy().mean()
        # value_loss = F.mse_loss(self.value(batch_state).squeeze(), batch_return)
        pred_vals = self.value(batch_state).squeeze()
        true_vals = batch_teacher_value
        value_loss = F.mse_loss(pred_vals, batch_teacher_value)
        loss = policy_loss - self.entropy_coef * entropy + self.value_func_coef * value_loss
        self.explained_variance = 1 - (torch.var(true_vals-pred_vals) / torch.var(true_vals)).item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.value_loss_record.append(float(value_loss))
        self.policy_loss_record.append(float(loss))


    # def compute_returns(self, rewards, dones, next_value):
    #   returns = torch.zeros_like(rewards, dtype=torch.float32)
    #   returns[-1] = rewards[-1] + self.gamma * next_value * (1 - dones[-1])
    #   for i in reversed(range(len(rewards)-1)):
    #     returns[i] = rewards[i] + self.gamma * returns[i+1] * (1 - dones[i])
    #   return returns

    def compute_gae(self, rewards, dones, values, next_value):
      advantages = torch.zeros_like(rewards, dtype=torch.float32)
      teacher_values = torch.zeros_like(rewards, dtype=torch.float32)
      gae = 0
      for i in reversed(range(len(rewards))):
        not_done = 1 - dones[i]
        delta = rewards[i] + self.gamma * next_value * not_done - values[i] # TD error
        gae = delta + self.gamma * self.lambd * not_done * gae
        advantages[i] = gae
        next_value = values[i]
        teacher_values[i] = gae + values[i] # return = value_function + advantage
      return advantages, teacher_values
    
    def get_statistics(self):
      return {
        "value": _mean_or_nan(self.value_record),
        "value_loss": _mean_or_nan(self.value_loss_record),
        "policy_loss": _mean_or_nan(self.policy_loss_record),
        "explained_variance": self.explained_variance,
        "n_updates": self.n_updates,
        "return_mean": self.return_mean,
        "return_std": self.return_std,
      }
