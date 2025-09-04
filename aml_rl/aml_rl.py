# Example harness for PPO_Simple.py
# Contact: sjguy@umn.edu

import torch
import torch.nn as nn
import torch.optim as optim
from PPO_Simple import PPO
import numpy as np
import pygame
from tqdm import tqdm
from network import Policy, Value

# import single_agent_env as env
# import single_agent_dd_env as env
import single_agent_dd_env as env
# import two_agent_env as env
# import two_agent_collision_env as env
# import n_agent_env as env

pygame.init()
screen_width = 800 # was previously 500
screen_height = 800
screen = pygame.display.set_mode((screen_width, screen_height))

clock = pygame.time.Clock()


arena_size = 3 # was previously 1.8
agent_radius = 0.11
goal_radius = 0.2

state_size, action_size = env.init(arena_size, agent_radius, goal_radius/2, goal_radius, [0.7, 2.0])


class Branched(torch.nn.Module):
  """Calls forward functions of child modules in parallel."""

  def __init__(self, *modules):
      super().__init__()
      self.child_modules = torch.nn.ModuleList(modules)

  def forward(self, *args, **kwargs):
      return tuple(mod(*args, **kwargs) for mod in self.child_modules)

# Call render(), but save the result to a buffer instead of displaying it
def render_to_buffer(state, surface, screen_width, screen_height):
  # Call render() to draw on the new surface
  env.render(state, surface, screen_width, screen_height)

import cv2
def render_video(video_name, render_fps = 30):
  max_num_evals = 10
  max_episode_render_step = 300
  num_evals = 0
  num_steps = 0
  episode_step = 0
  running = True
  state = env.reset()
  buffers = []
  while num_evals < max_num_evals and num_steps < max_episode_render_step:
    # env.goal_radius = agent_radius # Force a large goal radius for the video

    action_dist = policy(torch.tensor(env.sense(state), dtype=torch.float32))
    # action = action_dist.sample().cpu().detach().numpy()*v_max
    action = action_dist.mean.detach().numpy()*v_max
    state, reward, reached_goal = env.step(state, action, dt)
    num_steps += 1
    episode_step += 1

    # env.render(state, screen, screen_width, screen_height)
    surface = pygame.Surface((screen_width, screen_height))
    render_to_buffer(state, surface, screen_width, screen_height)
    buffers.append(surface)

    if reached_goal or episode_step >= max_episode_steps_eval:
      num_evals += 1
      episode_step = 0
      state = env.reset()

  # Save the buffers to a video file
  out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), render_fps, (screen_width, screen_height))
  for buffer in buffers:
    img = pygame.surfarray.array3d(buffer)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out.write(img)
  out.release()


def render():
  render_fps = 30
  max_num_evals = 10
  max_episode_render_step = 200
  num_evals = 0
  num_steps = 0
  episode_step = 0
  running = True
  state = env.reset()
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
    # quit on escape
    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
      running = False

    action_dist = policy(torch.tensor(env.sense(state), dtype=torch.float32))
    action = action_dist.sample().cpu().detach().numpy()*v_max
    state, reward, reached_goal = env.step(state, action, dt)
    num_steps += 1
    episode_step += 1

    # breakpoint()
    env.render(state, screen, screen_width, screen_height)

    if reached_goal or episode_step >= max_episode_steps_eval:
      num_evals += 1
      episode_step = 0
      state = env.reset()

    if num_evals >= max_num_evals or num_steps >= max_episode_render_step:
      running = False

    pygame.display.flip()
    clock.tick(render_fps)


policy = Policy(state_size, action_size)
policy.std_dev_v = nn.Parameter(torch.tensor([1.0]*action_size, dtype=torch.float32)) # Set initial standard deviation (will be softplus transformed to ensure positivity)
vf = Value(state_size)

# Small initialization of policy in last layer
nn.init.uniform_(policy.fc3.weight, -3e-3, 3e-3)

# Orthogonal initialization of vf
for name, param in vf.named_parameters():
  if "weight" in name:
    nn.init.orthogonal_(param)
  elif "bias" in name:
    nn.init.zeros_(param)

model = Branched(policy, vf)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
ppo = PPO(policy, vf, optimizer, update_interval=2000, minibatch_size=2000, epochs=10, standardize_advantages=True, clip_eps=0.2, value_func_coef=0.01, entropy_coef=0.0)
max_episode_steps_train = 100
max_episode_steps_eval = 100

def eval_policy(num_evals, policy):
  eval_returns = []
  with torch.no_grad():
    for _ in range(num_evals):
      R = 0
      state = env.reset()
      for _ in range(max_episode_steps_eval):
        action_dist = policy(torch.tensor(env.sense(state), dtype=torch.float32))
        action = action_dist.sample().cpu().detach().numpy()*v_max
        state, reward, reached_goal = env.step(state, action, dt)
        R += reward
        if reached_goal:
          break
      eval_returns.append(R)
  print("eval_returns ", np.mean(eval_returns) , np.std(eval_returns), "[",np.min(eval_returns), "...", np.max(eval_returns), "]")


dt = 0.1
v_max = 1.0
avg_return = -np.inf
R = 0
done = True

train_steps  = 5000000
render_steps = 500000
stats_steps  = 50000
eval_steps   = 100000

for i in tqdm(range(train_steps)):
    if done:
        state = env.reset()
        done = False
        episode_step = 0
        if avg_return == -np.inf:
            avg_return = R
        avg_return = 0.01 * R + 0.99 * avg_return
        R = 0
    action = ppo.act(env.sense(state)) * v_max
    episode_step += 1
    state, reward, reached_goal = env.step(state, action, dt)
    if reached_goal or episode_step >= max_episode_steps_train:
        done = True
    R += reward
    ppo.observe(env.sense(state), action, reward, reached_goal, done)

    if (i + 1) % stats_steps == 0:
        # print stats
        stats = ppo.get_statistics()
        print("stats: ", stats)

        # print("policy.std_dev: ", policy.std_dev.detach().numpy())

        print("avg_return: ", avg_return)

    if (i + 1) % eval_steps == 0:
        # Evaluate policy
        eval_policy(100, policy)

    if (i + 1) % render_steps == 0:
        # breakpoint()
        render()
        # render_video(f"videos/video_{i+1}.mp4", render_fps=15)

        # Save model 
        # torch.save(policy.state_dict(), f"models/policy_{i+1}.pt")
        # torch.save(policy.state_dict(), f"policy_latest.pt")

        # changing the save path
        torch.save(policy.state_dict(), f"../weights/best-weights-rl-vel/no-reward-goal-vel.pt")

print("done")
