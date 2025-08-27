import torch
import numpy as np
import pygame
import sys
import aml_rl.single_agent_dd_env as env
from aml_rl.network import Policy
from simulation_helper import *
from agent import Agent
from robot import Robot
from utils import *

# TODO: fix the problem/bug with rendering - the median speed of the cursor
#       is displaying 0 even when it's clearly moving

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
FPS = 30

# setting the size of the arena in meters
arena_size = 2.88
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
save_path = "./weights/best-weights-rl-vel/policy_latest(penalizing-not-same-vel-if-reached-goal).pt"
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
    agent_x = meters_to_pixels(state[0])
    agent_y = meters_to_pixels(state[1])
    radius_pixels = int(agent_radius / arena_size * screen_width)
    pygame.draw.circle(screen, (0, 0, 0), (agent_x, agent_y), radius_pixels, 1)
    pygame.draw.circle(screen, (100, 100, 255), (agent_x, agent_y), radius_pixels)

    # Draw orientation
    ori_x = agent_x + int(radius_pixels * np.cos(state[2]))
    ori_y = agent_y + int(radius_pixels * np.sin(state[2]))
    pygame.draw.line(screen, (0, 0, 0), (agent_x, agent_y), (ori_x, ori_y), 2)

    # Draw goal
    goal_radius_pixels = int(goal_radius / arena_size * screen_width)
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
    state[3] = new_goal[0]
    state[4] = new_goal[1]
    return state


def set_goal_vel(state, new_goal_vel):
    state[5] = new_goal_vel[0]
    state[6] = new_goal_vel[1]
    return state


if __name__ == "__main__":
    last_sample_time = 0
    agent = Agent(
        display_flag=False,
        draw_circle=False,
        save_path="./weights/best-weights/best_weight.pth",
        meters_to_pixels=meters_to_pixels,
        pixels_to_meters=pixels_to_meters,
    )

    # robot = Robot(
    #     meters_to_pixels=meters_to_pixels,
    #     pixels_to_meters=pixels_to_meters,
    #     dt=dt,
    #     display_body=False,
    #     display_target=False,
    # )

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
        agent.pos[0] = pixels_to_meters(CURSOR_XY[0])
        agent.pos[1] = pixels_to_meters(CURSOR_XY[1])

        # updating the agent's past positions
        current_time = pygame.time.get_ticks()
        if current_time - last_sample_time >= dt:
            agent.update(pixels_to_meters(CURSOR_XY[0]), pixels_to_meters(CURSOR_XY[1]))
            last_sample_time = current_time

            # setting the state goal
            state = set_goal(state, agent.pos)  # setting the env goal to be the cursor
            state = set_goal_vel(
                state, agent.past_trajectory[:, -1] - agent.past_trajectory[:, -2]
            )  # setting the target velocity to be the most velocity

            # getting the action pair and new state
            action_dist = policy(torch.tensor(env.sense(state), dtype=torch.float32))
            action = action_dist.sample().cpu().detach().numpy() * v_max
            state, _, _ = env.step(state, action, dt)

            # render(screen, new_state, agent)
            # state = new_state

        # clock.tick(FPS)
        print("state:", state)
        render(screen, state, agent)
        clock.tick_busy_loop(FPS)

    pygame.quit()
    sys.exit()
