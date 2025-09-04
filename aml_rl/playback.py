
import single_agent_dd_env as env

import numpy as np
import pygame
import torch
import torch.nn as nn

policy_size = 32
arena_size = 1.8
agent_radius = 0.11
goal_radius = 0.15
action_scale = [0.7, 2.0] # Maximum linear and angular velocity
fps = 60
sim_dt = 1/fps

state_size, action_size = env.init(arena_size, agent_radius, goal_radius, goal_radius, action_scale)

agent_pos = np.array([0, 0])
agent_ori = 0
goal_pos = np.array([0.5, 0.5])

class Policy(nn.Module):
  def __init__(self):
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
    action = torch.tanh(x)
    return action
  
# Load policy
policy = Policy()
policy.load_state_dict(torch.load('policy_latest.pt'))

# Initialize environment
state = env.reset()
done = False

# Initialize pygame 
pygame.init()
screen_width = 500
screen_height = 500
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

# Main loop
running = True
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False

  # Get observation: relative goal position rotated to agent's frame
  rel_goal_pos = goal_pos - agent_pos
  sin_ori = np.sin(agent_ori)
  cos_ori = np.cos(agent_ori)
  rel_goal_x = cos_ori*rel_goal_pos[0] + sin_ori*rel_goal_pos[1]
  rel_goal_y = -sin_ori*rel_goal_pos[0] + cos_ori*rel_goal_pos[1]
  rel_goal_pos_rot = np.array([rel_goal_x, rel_goal_y])
  observation_pt = torch.tensor(rel_goal_pos_rot, dtype=torch.float32)
  action = policy(observation_pt).detach().numpy()
  action = np.clip(action, -1, 1) * action_scale

  # Simulate environment
  state = np.array([agent_pos[0], agent_pos[1], agent_ori, goal_pos[0], goal_pos[1]])
  state, _, _ = env.step(state, action, sim_dt)
  agent_pos = state[:2]
  agent_ori = state[2]
  if np.linalg.norm(agent_pos - goal_pos) < goal_radius:
    goal_pos[0] = np.random.uniform(low=-arena_size/2, high=arena_size/2)
    goal_pos[1] = np.random.uniform(low=-arena_size/2, high=arena_size/2)

  # Render
  env.render(state, screen, screen_width, screen_height)

  pygame.display.flip()
  clock.tick(fps)

  # If escape is pressed, exit
  keys = pygame.key.get_pressed()
  if keys[pygame.K_ESCAPE]:
    running = False

pygame.quit()