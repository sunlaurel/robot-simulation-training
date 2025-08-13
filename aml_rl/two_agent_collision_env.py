import numpy as np
import pygame

arena_size = 1
agent_radius = 0.1
goal_radius_min = 0.1
goal_radius_max = 0.2
goal_radius = goal_radius_max

agent_1_collided = False
agent_2_collided = False

def init(_arena_size, _agent_radius, _goal_radius):
  global arena_size, agent_radius, goal_radius_min, goal_radius_max
  arena_size = _arena_size
  agent_radius = _agent_radius
  goal_radius_max = _goal_radius
  goal_radius_min = _goal_radius/2
  state_size = 8
  action_size = 4
  return state_size, action_size

def sense(state):
  return state

def reset():
  global goal_radius, agent_1_collided, agent_2_collided, at_goal_1, at_goal_2
  state_pos_1 = np.random.uniform(low=-arena_size/2, high=arena_size/2, size=2)
  state_goal_1 = np.random.uniform(low=-arena_size/2, high=arena_size/2, size=2)
  state_pos_2 = np.random.uniform(low=-arena_size/2, high=arena_size/2, size=2)
  state_goal_2 = np.random.uniform(low=-arena_size/2, high=arena_size/2, size=2)
  state = np.concatenate((state_pos_1, state_goal_1, state_pos_2, state_goal_2))
  goal_radius = np.random.uniform(low=goal_radius_min, high=goal_radius_max)
  agent_1_collided = False
  agent_2_collided = False
  at_goal_1 = False
  at_goal_2 = False
  return state

def step_old(state, action, dt):
  state_pos_1 = state[:2]
  state_goal_1 = state[2:4]
  state_pos_2 = state[4:6]
  state_goal_2 = state[6:]

  last_dist_1 = np.linalg.norm(state_pos_1 - state_goal_1)
  last_dist_2 = np.linalg.norm(state_pos_2 - state_goal_2)

  state_pos_1 = state_pos_1 + action[:2]*dt
  state_pos_2 = state_pos_2 + action[2:]*dt
  distance_1 = np.linalg.norm(state_pos_1 - state_goal_1)
  distance_2 = np.linalg.norm(state_pos_2 - state_goal_2)
  state = np.concatenate((state_pos_1, state_goal_1, state_pos_2, state_goal_2))
  reward = (last_dist_1-distance_1) + (last_dist_2 - distance_2) - 2*dt

  agent_1_colliding = np.linalg.norm(state_pos_1 - state_pos_2) < 2*agent_radius
  agent_2_colliding = np.linalg.norm(state_pos_1 - state_pos_2) < 2*agent_radius
  if agent_1_colliding or agent_2_colliding: reward -= 2

  global agent_1_collided, agent_2_collided
  agent_1_collided = agent_1_colliding or agent_1_collided
  agent_2_collided = agent_2_colliding or agent_2_collided

  reached_goal = (distance_1 < goal_radius) and (distance_2 < goal_radius)
  if reached_goal: 
    reward += 20
    if agent_1_collided or agent_2_collided: reward -= 18

  return state, reward, reached_goal

last_dist_1 = 0
last_dist_2 = 0
at_goal_1 = False
at_goal_2 = False

def step(state, action, dt):
  global last_dist_1, last_dist_2, at_goal_1, at_goal_2

  state_pos_1 = state[:2]
  state_goal_1 = state[2:4]
  state_pos_2 = state[4:6]
  state_goal_2 = state[6:]

  if not at_goal_1:
    state_pos_1 = state_pos_1 + action[:2]*dt
  if not at_goal_2:
    state_pos_2 = state_pos_2 + action[2:]*dt
  distance_1 = np.linalg.norm(state_pos_1 - state_goal_1)
  distance_2 = np.linalg.norm(state_pos_2 - state_goal_2)
  state = np.concatenate((state_pos_1, state_goal_1, state_pos_2, state_goal_2))
  reward = (last_dist_1-distance_1) + (last_dist_2 - distance_2) - 2*dt

  agent_1_colliding = np.linalg.norm(state_pos_1 - state_pos_2) < 2*agent_radius
  agent_2_colliding = np.linalg.norm(state_pos_1 - state_pos_2) < 2*agent_radius
  if agent_1_colliding or agent_2_colliding: reward -= 2

  global agent_1_collided, agent_2_collided
  agent_1_collided = agent_1_colliding or agent_1_collided
  agent_2_collided = agent_2_colliding or agent_2_collided

  if (distance_1 < goal_radius) and not at_goal_1: 
    reward += 10
    if agent_1_collided: reward -= 9
  
  if (distance_2 < goal_radius) and not at_goal_2:
    reward += 10
    if agent_2_collided: reward -= 9

  # Cache values for next step
  at_goal_1 = distance_1 < goal_radius
  at_goal_2 = distance_2 < goal_radius
  last_dist_1 = distance_1
  last_dist_2 = distance_2
  reached_goal = at_goal_1 and at_goal_2

  return state, reward, reached_goal

def render(state, screen, screen_width, screen_height):
  screen.fill((255, 255, 255))
  agent_1_x = int((state[0] + arena_size/2) / arena_size * screen_width)
  agent_1_y = int((state[1] + arena_size/2) / arena_size * screen_height)
  agent_2_x = int((state[4] + arena_size/2) / arena_size * screen_width)
  agent_2_y = int((state[5] + arena_size/2) / arena_size * screen_height)
  radius_pixels = int(agent_radius / arena_size * screen_width)
  pygame.draw.circle(screen, (0, 0, 255), (agent_1_x, agent_1_y), radius_pixels)
  pygame.draw.circle(screen, (0, 255, 0), (agent_2_x, agent_2_y), radius_pixels)
  goal_1_x = int((state[2] + arena_size/2) / arena_size * screen_width)
  goal_1_y = int((state[3] + arena_size/2) / arena_size * screen_height)
  goal_2_x = int((state[6] + arena_size/2) / arena_size * screen_width)
  goal_2_y = int((state[7] + arena_size/2) / arena_size * screen_height)
  goal_radius_pixels = int(goal_radius / arena_size * screen_width)
  pygame.draw.circle(screen, (0, 0, 100), (goal_1_x, goal_1_y), goal_radius_pixels, 1)
  pygame.draw.circle(screen, (0, 100, 0), (goal_2_x, goal_2_y), goal_radius_pixels, 1)
  pygame.display.flip()