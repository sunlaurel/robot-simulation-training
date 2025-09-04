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
  obs_size = 4
  action_size = 2
  return obs_size, action_size

def sense(state):
  return state

def reset():
  global goal_radius
  state_pos = np.random.uniform(low=-arena_size/2, high=arena_size/2, size=2)
  state_goal = np.random.uniform(low=-arena_size/2, high=arena_size/2, size=2)
  state = np.concatenate((state_pos, state_goal))
  goal_radius = np.random.uniform(low=goal_radius_min, high=goal_radius_max)
  return state

def step(state, action, dt):
  state_pos = state[:2]
  state_goal = state[2:4]
  state_pos = state_pos + action[:2]*dt
  distance = np.linalg.norm(state_pos - state_goal)
  state = np.concatenate((state_pos, state_goal))
  reward = -distance
  reached_goal = (distance < goal_radius) 
  if reached_goal: reward += 10
  return state, reward, reached_goal

def render(state, screen, screen_width, screen_height):
  screen.fill((255, 255, 255))
  agent_x = int((state[0] + arena_size/2) / arena_size * screen_width)
  agent_y = int((state[1] + arena_size/2) / arena_size * screen_height)
  radius_pixels = int(agent_radius / arena_size * screen_width)
  pygame.draw.circle(screen, (0, 0, 255), (agent_x, agent_y), radius_pixels)
  goal_radius_pixels = int(goal_radius / arena_size * screen_width)
  goal_x = int((state[2] + arena_size/2) / arena_size * screen_width)
  goal_y = int((state[3] + arena_size/2) / arena_size * screen_height)
  pygame.draw.circle(screen, (0, 0, 100), (goal_x, goal_y), goal_radius_pixels, 1)
  pygame.display.flip()