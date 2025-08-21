import torch
import numpy as np
import pygame
import sys
import aml_rl.single_agent_dd_env as env
from aml_rl.network import Policy
from simulation_helper import *
from agent import Agent
from utils import *

# initializing the screen
pygame.init()
screen_width = 800
screen_height = screen_width
screen = pygame.display.set_mode((screen_width, screen_height))
font = pygame.font.SysFont("Arial", 24)
clock = pygame.time.Clock()
dt = 0.12
v_max = 1
state = env.reset()

# setting the size of the arena in meters
arena_size = 5
agent_radius = 0.11
goal_radius = 0.2


# variables of simulation
RESET = True
RUNNING = True
CURSOR_XY = (screen_width / 2, screen_height / 2)  # cursor position is in pixels
RADIUS = 0.75

state_size, action_size = env.init(
    arena_size, agent_radius, goal_radius / 2, goal_radius, [0.7, 2.0]
)


# initializing the policy network
save_path = "./weights/best-weights-rl/policy_latest(2).pt"
policy = Policy(state_size, action_size)
policy.load_state_dict(
    torch.load(save_path, map_location=torch.device("cpu"))
)  # use this for non-gpu machines


def render_text(agent, screen):
    # calculating the speed of the agent
    speeds = (
        np.sum(
            (agent.past_trajectory[:, :-1] - agent.past_trajectory[:, 1:]) ** 2,
            axis=0,
        )
        ** 0.5
    )

    # rendering the speed and position on screen
    median_speed = np.median(speeds / 0.12)
    speed_text = f"Median speed(m/s): {median_speed:.5f}"
    pos_text = f"Position(m): ({agent.pos[0]:.2f}, {agent.pos[1]:.2f})"

    text_surface_speed = font.render(speed_text, True, (0, 0, 0))
    text_surface_pos = font.render(pos_text, True, (0, 0, 0))

    text_rect_speed = text_surface_speed.get_rect(
        bottomright=(
            screen_width - 30,
            screen_height - 20,
        )
    )
    text_rect_pos = text_surface_pos.get_rect(
        bottomright=(
            screen_width - 30,
            screen_height - 50,
        )
    )

    screen.blits(
        ((text_surface_speed, text_rect_speed), (text_surface_pos, text_rect_pos))
    )


# renders one env frame
def render(screen, state, agent):
    screen.fill((255, 255, 255))
    # agent_x = int((state[0] + arena_size / 2) / arena_size * screen_width)
    # agent_y = int((state[1] + arena_size / 2) / arena_size * screen_height)
    agent_x = meters_to_pixels(state[0])
    agent_y = meters_to_pixels(state[1])
    radius_pixels = int(agent_radius / arena_size * screen_width)
    pygame.draw.circle(screen, (0, 0, 0), (agent_x, agent_y), radius_pixels, 1)
    pygame.draw.circle(screen, (100, 100, 255), (agent_x, agent_y), radius_pixels)
    print("curr pos calc:", agent_x, ",", agent_y)

    # Draw orientation
    ori_x = agent_x + int(radius_pixels * np.cos(state[2]))
    ori_y = agent_y + int(radius_pixels * np.sin(state[2]))
    pygame.draw.line(screen, (0, 0, 0), (agent_x, agent_y), (ori_x, ori_y), 2)

    # Draw goal
    goal_radius_pixels = int(goal_radius / arena_size * screen_width)
    # goal_x = int((state[3] + arena_size / 2) / arena_size * screen_width)
    # goal_y = int((state[4] + arena_size / 2) / arena_size * screen_height)
    goal_x = meters_to_pixels(state[3])
    goal_y = meters_to_pixels(state[4])
    pygame.draw.circle(screen, (0, 0, 100), (goal_x, goal_y), goal_radius_pixels, 3)

    # breakpoint()
    render_text(agent, screen)
    agent.draw(screen)
    pygame.display.flip()


def pixels_to_meters(pixel):
    # making it so the center is (0, 0), assuming that it's not already centered
    return pixel * arena_size / screen_width - arena_size / 2


def meters_to_pixels(meter):
    # shifting coordinates to match original screen in pygame
    return int((meter + arena_size / 2) * screen_width / arena_size)


def set_goal(state, new_goal):
    # offsetting so that the new goal is RADIUS away from the last future predicted position on the right
    # breakpoint()
    offset_goal = np.array([-new_goal[0], new_goal[1]])
    offset_goal /= np.linalg.norm(offset_goal)
    offset_goal *= RADIUS
    offset_goal += new_goal

    state[-2] = offset_goal[0]
    state[-1] = offset_goal[1]

    # state[-2] = new_goal[0]
    # state[-1] = new_goal[1]

    return state

# TODO #1:
# For now, comment out your future motion in your environment (to save on processing time),
# go back to the static goal, and let's update your reward function to try and get this behavior.
# Stephen suggested just trying to add a new penalty based on distance to the goal.
# You can try something simple like every step check the final distance, train the agent,
# and then try again in the interactive simulator to see how it worked

# TODO #2:
# I wonder if it's to do with the control update not being correctly handled (relating to dt)
# in Stephen's or your code

if __name__ == "__main__":
    last_sample_time = 0
    agent = Agent(
        display_flag=True,
        draw_circle=False,
        save_path="./weights/best-weights/best_weight.pth",
        meters_to_pixels=meters_to_pixels,
        pixels_to_meters=pixels_to_meters,
    )

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
            state = env.reset()
            pygame.mouse.set_pos(
                meters_to_pixels(state[-2]), meters_to_pixels(state[-1])
            )
            # TODO: reset the agent's past positions
            RESET = False

        CURSOR_XY = pygame.mouse.get_pos()  # setting the cursor pos to the mouse pos

        # updating the agent's past positions
        current_time = pygame.time.get_ticks()
        if current_time - last_sample_time >= dt:
            agent.update(pixels_to_meters(CURSOR_XY[0]), pixels_to_meters(CURSOR_XY[1]))
            last_sample_time = current_time

            state = set_goal(
                state, agent.future_trajectory[:, 0]
            )  # setting the env goal

            # getting the action pair and new state
            action_dist = policy(torch.tensor(env.sense(state), dtype=torch.float32))
            action = action_dist.sample().cpu().detach().numpy() * v_max
            new_state, _, _ = env.step(state, action, dt)

            # breakpoint()
            render(screen, new_state, agent)
            state = new_state

        clock.tick()

    pygame.quit()
    sys.exit()
