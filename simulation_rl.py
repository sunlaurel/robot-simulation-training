import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pygame
import sys
from tqdm import tqdm
from aml_rl.network import Policy
import aml_rl.single_agent_dd_env as env


# initializing the screen
pygame.init()
screen_width = 500
screen_height = screen_width
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()
dt = 0.1
v_max = 1.0
state = env.reset()


# setting the size of the arena in meters
arena_size = 1.8
agent_radius = 0.11
goal_radius = 0.2


# variables of simulation
RESET = True
RUNNING = True
CURSOR_XY = (screen_width / 2, screen_height / 2)  # cursor position is in pixels


state_size, action_size = env.init(
    arena_size, agent_radius, goal_radius / 2, goal_radius, [0.7, 2.0]
)


# initializing the policy network
save_path = "./weights/best-weights-rl/policy_latest.pt"
policy = Policy(state_size, action_size)
policy.load_state_dict(
    torch.load(save_path, map_location=torch.device("cpu"))
)  # use this for non-gpu machines


# renders one env frame
def render(screen, state):
    # breakpoint()
    env.render(state, screen, screen_width, screen_height)


def pixels_to_meters(pixel):
    return pixel * arena_size / screen_width


def meters_to_pixels(meter):
    return meter * screen_width / arena_size


def set_goal(state, new_goal):
    state[-2] = new_goal[0]
    state[-1] = new_goal[1]
    return state


if __name__ == "__main__":
    while RUNNING:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                RUNNING = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    RESET = True
                elif event.key == pygame.K_q:
                    RUNNING = False
                elif event.key == pygame.K_m:
                    CURSOR_XY = (screen_width / 2, screen_height / 2)

        if RESET:
            env.reset()
            RESET = False

        # TODO: set the env goal here
        state = set_goal(state, np.vectorize(pixels_to_meters)(CURSOR_XY))

        action_dist = policy(
            torch.tensor(env.sense(state), dtype=torch.float32)
        )  # policy returns action distribution
        action = (
            action_dist.sample().cpu().detach().numpy() * v_max
        )  # samples from the action distribution

        # breakpoint()

        new_state, _, _ = env.step(state, action, dt)
        render(screen, new_state)

        pygame.display.flip()
        clock.tick(15)

    pygame.quit()
    sys.exit()

# RESET = True  # make a key callback so 'r' sets to true
# RUNNING = True  # make a key callback so 'q' sets to false
# CURSOR_XY = (x, y)  # have some event set this too
# pygame.init()
# screen = ...  # fill in

# while RUNNING:
#     # check pygame keys here
#     if RESET:
#         env.reset()
#         RESET = False
#     env.set_goal(CURSOR_XY)
#     state = (
#         env.get_observation()
#     )  # state was updated by the cursor, so you need to get it explicitly
#     action = agent(state)
#     new_state = env.step()
#     render(screen, new_state)
