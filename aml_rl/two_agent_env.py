import numpy as np
import pygame

arena_size = 1
agent_radius = 0.1
goal_radius_min = 0.1
goal_radius_max = 0.2
goal_radius = goal_radius_max

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
  global goal_radius
  state_pos_1 = np.random.uniform(low=-arena_size/2, high=arena_size/2, size=2)
  state_goal_1 = np.random.uniform(low=-arena_size/2, high=arena_size/2, size=2)
  state_pos_2 = np.random.uniform(low=-arena_size/2, high=arena_size/2, size=2)
  state_goal_2 = np.random.uniform(low=-arena_size/2, high=arena_size/2, size=2)
  state = np.concatenate((state_pos_1, state_goal_1, state_pos_2, state_goal_2))
  goal_radius = np.random.uniform(low=goal_radius_min, high=goal_radius_max)
  return state

def step(state, action, dt):
  state_pos_1 = state[:2]
  state_goal_1 = state[2:4]
  state_pos_2 = state[4:6]
  state_goal_2 = state[6:]
  state_pos_1 = state_pos_1 + action[:2]*dt
  state_pos_2 = state_pos_2 + action[2:]*dt
  distance_1 = np.linalg.norm(state_pos_1 - state_goal_1)
  distance_2 = np.linalg.norm(state_pos_2 - state_goal_2)
  state = np.concatenate((state_pos_1, state_goal_1, state_pos_2, state_goal_2))
  reward = -distance_1 - distance_2
  reached_goal = (distance_1 < goal_radius) and (distance_2 < goal_radius)
  if reached_goal: reward += 20
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