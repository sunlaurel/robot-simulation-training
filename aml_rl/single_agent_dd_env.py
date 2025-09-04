import numpy as np
import pygame

# Single agent environment, with differential drive dynamics

arena_size = 1
agent_radius = 0.1
goal_radius_min = 0.1
goal_radius_max = 0.2
goal_radius = goal_radius_max
action_scale = [0.2, 2.0] # Maximum linear and angular velocity

# state --> pos, ori, vel, goal, goal_vel

def init(_arena_size, _agent_radius, _goal_radius_min, _goal_radius_max, _action_scale):
  global arena_size, agent_radius, goal_radius_min, goal_radius_max, action_scale
  arena_size = _arena_size
  agent_radius = _agent_radius
  goal_radius_max = _goal_radius_max
  goal_radius_min = _goal_radius_min
  action_scale = _action_scale
  obs_size = 2 # 2 for position, 2 for orientation, 2 for goal position
  action_size = 2
  return obs_size, action_size

def sense(state):
  pos = state[:2]
  ori = state[2]
  # goal = state[3:]n
  goal = state[5:7]
  ori_cos = np.cos(ori)
  ori_sin = np.sin(ori)
  #relative goal position (rotated to agent frame)
  rel_goal = np.array([goal[0] - pos[0], goal[1] - pos[1]])
  rel_goal_rot = np.array([ori_cos*rel_goal[0] + ori_sin*rel_goal[1], -ori_sin*rel_goal[0] + ori_cos*rel_goal[1]])
  return rel_goal_rot

def reset():
    global goal_radius
    state_pos = np.random.uniform(low=-arena_size/2, high=arena_size/2, size=2)
    state_ori = np.random.uniform(low=-np.pi, high=np.pi)
    state_goal = np.random.uniform(low=-arena_size/2, high=arena_size/2, size=2)

    # generating random starting and goal velocities
    state_vel_x = np.random.uniform(low=0, high=action_scale[0])
    state_vel_y = np.random.uniform(
        low=0,
        high=np.sqrt(
            np.linalg.norm(action_scale[0]) - np.linalg.norm(state_vel_x)
        ),
    )
    state_goal_vel_x = np.random.uniform(low=0, high=action_scale[0])
    state_goal_vel_y = np.random.uniform(
        low=0,
        high=np.sqrt(
            np.linalg.norm(action_scale[0]) - np.linalg.norm(state_goal_vel_x)
        ),
    )

    # creating the starting state
    state = np.concatenate(
        (state_pos, [state_ori], [state_vel_x, state_vel_y], state_goal, [state_goal_vel_x, state_goal_vel_y])
    )

    goal_radius = np.random.uniform(low=goal_radius_min, high=goal_radius_max)
    return state

def step(state, action, dt):
    # breakpoint()
    v_max = action_scale[0]
    w_max = action_scale[1]
    state_pos = state[:2]
    state_ori = state[2]
    state_vel = state[3:5]
    state_goal = state[5:7]
    state_goal_vel = state[7:]

    lin_vel = action[0]*v_max
    ang_vel = action[1]*w_max

    last_distance = np.linalg.norm(state_pos - state_goal)

    state_pos = state_pos + lin_vel * np.array([np.cos(state_ori), np.sin(state_ori)]) * dt
    state_ori = state_ori + ang_vel * dt
    state_ori = (state_ori + np.pi) % (2 * np.pi) - np.pi

    # TODO: update goal position + goal velocity
    state_goal += (state_goal_vel * dt) # for training only

    distance = np.linalg.norm(state_pos - state_goal)

    # updating the state
    state = np.concatenate(
        (
            state_pos,
            [state_ori],
            lin_vel * np.array([np.cos(state_ori), np.sin(state_ori)]),
            state_goal,
            state_goal_vel,
        )
    )
    # reward = -distance
    reward = last_distance - distance - dt * v_max

    # penalizing if going backwards
    if lin_vel < 0:
        reward -= 2

    # penalize more if the distance increases from last time
    if distance > last_distance:
        reward -= 2

    reached_goal = (distance < goal_radius) 
    if reached_goal: 
        reward += 10
        # reward -= np.linalg.norm(state_goal_vel - lin_vel) # penalizing if not same velocity

    # # have it always try to match the velocity
    # reward -= (state_goal_vel[0] - lin_vel * np.cos(state_ori))
    # reward -= (state_goal_vel[1] - lin_vel * np.sin(state_ori))
    # # else:
    # #   reward -= abs(action_scale[0] - np.linalg.norm(lin_vel))  # penalizing if not reached target and the velocity isn't speeding up to catch up

    return state, reward, reached_goal

def render(state, screen, screen_width, screen_height):
    screen.fill((255, 255, 255))
    agent_x = int((state[0] + arena_size/2) / arena_size * screen_width)
    agent_y = int((state[1] + arena_size/2) / arena_size * screen_height)
    radius_pixels = int(agent_radius / arena_size * screen_width)
    pygame.draw.circle(screen, (0, 0, 0), (agent_x, agent_y), radius_pixels, 1)
    pygame.draw.circle(screen, (100, 100, 255), (agent_x, agent_y), radius_pixels)

    # Draw orientation
    ori_x = agent_x + int(radius_pixels * np.cos(state[2]))
    ori_y = agent_y + int(radius_pixels * np.sin(state[2]))
    pygame.draw.line(screen, (0, 0, 0), (agent_x, agent_y), (ori_x, ori_y), 2)

    # Draw goal
    goal_radius_pixels = int(goal_radius / arena_size * screen_width)
    goal_x = int((state[3] + arena_size / 2) / arena_size * screen_width)
    goal_y = int((state[4] + arena_size / 2) / arena_size * screen_height)
    pygame.draw.circle(screen, (0, 0, 100), (goal_x, goal_y), goal_radius_pixels, 3)
    pygame.display.flip()
