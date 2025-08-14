import numpy as np
import pygame

num_agents = 2
max_vel = 1.0
arena_size = 1
agent_radius = 0.1
goal_radius_min = 0.1
goal_radius_max = 0.2
goal_radius = goal_radius_max

agent_collided = [False] * num_agents

def init(_arena_size, _agent_radius, _goal_radius):
  global arena_size, agent_radius, goal_radius_min, goal_radius_max, num_agents
  arena_size = _arena_size
  agent_radius = _agent_radius
  goal_radius_max = _goal_radius
  goal_radius_min = _goal_radius
  state_size = 4 * num_agents
  action_size = 2 * num_agents
  return state_size, action_size

def sense(state):
  return state

def reset():
  global goal_radius, agent_collided, at_goal, last_goal_dist
  any_agent_colliding = True

  #TODO: Don't reset all agents if only one is colliding, just the colliding one
  while any_agent_colliding:
    state_pos = np.random.uniform(low=-arena_size/2, high=arena_size/2, size=num_agents*2)
    state_goal = np.random.uniform(low=-arena_size/2, high=arena_size/2, size=num_agents*2)
    state = np.concatenate((state_pos, state_goal))

    any_agent_colliding = False
    for i in range(num_agents):
      for j in range(i+1, num_agents):
        if np.linalg.norm(state_pos[i*2:i*2+2] - state_pos[j*2:j*2+2]) < 2*agent_radius:
          any_agent_colliding = True
          break
      if any_agent_colliding: break
    if any_agent_colliding: continue

    for i in range(num_agents):
      last_goal_dist[i] = np.linalg.norm(state_pos[i*2:i*2+2] - state_goal[i*2:i*2+2])
      if last_goal_dist[i] < goal_radius:
        any_agent_colliding = True
        break
    if any_agent_colliding: continue

  goal_radius = np.random.uniform(low=goal_radius_min, high=goal_radius_max)
  agent_collided = [False] * num_agents
  at_goal = [False] * num_agents
  return state


last_goal_dist = [0] * num_agents

def step(state, action, dt):
  global last_goal_dist, at_goal

  state_pos = state[:num_agents*2]
  state_goal = state[num_agents*2:]

  # if not at_goal_1:
  #   state_pos_1 = state_pos_1 + action[:2]*dt
  # if not at_goal_2:
  #   state_pos_2 = state_pos_2 + action[2:]*dt


  # Update agent positions
  for i in range(num_agents):
    if not at_goal[i]:
      state_pos[i*2:i*2+2] = state_pos[i*2:i*2+2] + action[i*2:i*2+2]*dt

  # Check for collisions between agents
  agents_colliding = [False] * num_agents
  for i in range(num_agents):
    for j in range(i+1, num_agents):
      if np.linalg.norm(state_pos[i*2:i*2+2] - state_pos[j*2:j*2+2]) < 2*agent_radius:
        agents_colliding[i] = True
        agents_colliding[j] = True
        agent_collided[i] = True
        agent_collided[j] = True

  state = np.concatenate((state_pos, state_goal))

  reward = 0
  for i in range(num_agents):
    if at_goal[i]: continue
    goal_distance = np.linalg.norm(state_pos[i*2:i*2+2] - state_goal[i*2:i*2+2])
    reward += (last_goal_dist[i] - goal_distance) - max_vel*dt
    if (goal_distance < goal_radius) and not at_goal[i]: 
      reward += 10
      if agent_collided[i]: reward -= 9
      at_goal[i] = True
    last_goal_dist[i] = goal_distance
    # Collision penalty
    if agents_colliding[i]: reward -= 2.0

  reached_all_goals = np.all(at_goal)
  normalized_reward = reward / (num_agents)
  return state, normalized_reward, reached_all_goals


# HSV to RGB
def hsv_to_rgb(h, s, v):
  if s == 0.0: v*=255; return (v, v, v)
  i = int(h*6.) 
  f = (h*6.)-i
  p,q,t = int(255*(v*(1.-s))), int(255*(v*(1.-s*f))), int(255*(v*(1.-s*(1.-f))))
  v*=255
  i%=6
  if i == 0: return (v, t, p)
  if i == 1: return (q, v, p)
  if i == 2: return (p, v, t)
  if i == 3: return (p, q, v)
  if i == 4: return (t, p, v)
  if i == 5: return (v, p, q)
  
# Index to color
def idx_color(i, scale=1.0):
  r,g,b = hsv_to_rgb(i/num_agents, 1.0, 1.0)
  return int(r*scale), int(g*scale), int(b*scale)

def render(state, screen, screen_width, screen_height):
  screen.fill((255, 255, 255))
  for a in range(num_agents):
    agent_x = int((state[a*2] + arena_size/2) / arena_size * screen_width)
    agent_y = int((state[a*2+1] + arena_size/2) / arena_size * screen_height)
    radius_pixels = int(agent_radius / arena_size * screen_width)
    pygame.draw.circle(screen, idx_color(a), (agent_x, agent_y), radius_pixels)

    goal_radius_pixels = int(goal_radius / arena_size * screen_width)
    goal_x = int((state[num_agents*2+a*2] + arena_size/2) / arena_size * screen_width)
    goal_y = int((state[num_agents*2+a*2+1] + arena_size/2) / arena_size * screen_height)
    pygame.draw.circle(screen, idx_color(a,0.5), (goal_x, goal_y), goal_radius_pixels, 2)

  pygame.display.flip()

