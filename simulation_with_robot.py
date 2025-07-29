import pygame
import sys
import math
import pandas as pd
from agent import Agent
from robot import Robot, RADIUS, MOVE_RADIUS
from utils import *
from simulation_helper import *

""" Constants """
WIDTH, HEIGHT = 10, 8
BG_COLOR = (255, 255, 255)
FPS = 30
# SAMPLING_INTERVAL_MS = 8.33 / 1000  # ~8.33 samples/sec
SAMPLING_INTERVAL_MS = 0.12

########### Initializing constants for the CSV file ###########
id = 0
csv_data = []

""" Initializing the game + setup """
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode(
        (meters_to_pixels(WIDTH), meters_to_pixels(HEIGHT))
    )
    pygame.display.set_caption("Trajectory Prediction Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)
    last_sample_time = 0
    agent = Agent(x=2, y=5, radius=0.5)
    robot = Robot(
        x=2,
        y=7,
        target_x=agent.pos[0],
        target_y=agent.pos[1],
        theta=-math.pi / 2,
        width=0.7,
        height=0.9,
        dt=SAMPLING_INTERVAL_MS,
    )

    """ Main Loop """
    running = True
    while running:
        screen.fill(BG_COLOR)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            agent.handle_event(event)
            # robot.handle_event(event)

        # sampling positions at 0.12 sec/sample
        current_time = pygame.time.get_ticks()
        if current_time - last_sample_time >= SAMPLING_INTERVAL_MS:
            agent.update(agent.pos[0], agent.pos[1])
            u_next = robot.policy(
                agent.past_trajectory,
            )  # passing in the last two positions of predicted trajectory
            csv_data.append(
                {
                    "id": id,
                    "agent_past_pos": agent.past_trajectory,
                    "robot_past_pos": robot.past_trajectory,
                    "predicted_target_pos": robot.target_pos,
                }
            )  # adding to the csv file
            id += 1
            robot.update(u_next)
            last_sample_time = current_time

        # Draws a circle the radius of what the distance is when the agent is standing still
        draw_transparent_circle(
            screen,
            np.vectorize(meters_to_pixels)(agent.pos),
            int(meters_to_pixels(RADIUS)),
        )

        draw_transparent_circle(
            screen,
            np.vectorize(meters_to_pixels)(robot.pos),
            int(meters_to_pixels(MOVE_RADIUS)),
        )

        agent.draw(screen)
        robot.draw(screen)
        display_text(agent, robot, font, screen, SAMPLING_INTERVAL_MS, WIDTH, HEIGHT)

        pygame.display.flip()
        clock.tick(FPS)

    df = pd.DataFrame(csv_data)
    df.to_csv('./simulation-data/sim1.csv', index=False)
    pygame.quit()
    sys.exit()
