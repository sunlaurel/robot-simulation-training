import pygame
import sys
from simulation_agent import Agent
from utils import *
from simulation_helper import *

""" Constants """
WIDTH, HEIGHT = 10, 10
BG_COLOR = (255, 255, 255)
FPS = 30


""" Initializing the game + setup """
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((meters_to_pixels(WIDTH), meters_to_pixels(HEIGHT)))
    pygame.display.set_caption("Trajectory Prediction Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)
    agent = Agent(x=2, y=5)
    sampling_interval_ms = 8.33 / 1000  # sampling at ~8.33 samples/sec
    last_sample_time = 0

    """ Main Loop """
    running = True
    while running:
        screen.fill(BG_COLOR)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            agent.handle_event(event)

        # sampling positions at 0.12 sec/sample
        current_time = pygame.time.get_ticks()
        if current_time - last_sample_time >= sampling_interval_ms:
            agent.update(agent.pos[0], agent.pos[1])
            last_sample_time = current_time

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
            bottomright=(meters_to_pixels(WIDTH) - 30, meters_to_pixels(HEIGHT) - 20)
        )
        text_rect_pos = text_surface_pos.get_rect(
            bottomright=(meters_to_pixels(WIDTH) - 30, meters_to_pixels(HEIGHT) - 50)
        )

        screen.blits(
            ((text_surface_speed, text_rect_speed), (text_surface_pos, text_rect_pos))
        )

        agent.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()
