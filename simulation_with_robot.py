import pygame
import sys
import math
from agent import Agent
from robot import Robot, STAND_RADIUS
from utils import *
from simulation_helper import *

""" Constants """
WIDTH, HEIGHT = 15, 8
BG_COLOR = (255, 255, 255)
FPS = 30
# SAMPLING_INTERVAL_MS = 8.33 / 1000  # ~8.33 samples/sec
SAMPLING_INTERVAL_MS = 0.12

# Displaying text on screen
def display_text(agent):
    # calculating the speed of the agent
    speeds = np.linalg.norm(
        agent.past_trajectory[:, :-1] - agent.past_trajectory[:, 1:],
        axis=0,
    ) / SAMPLING_INTERVAL_MS

    # rendering the speed and position on screen
    median_speed = np.median(speeds)
    speed_text = f"Median speed(m/s): {median_speed:.5f}"
    pos_text = f"Position(m): ({agent.pos[0]:.2f}, {agent.pos[1]:.2f})"
    robot_speed_text = f"Robot speed(m/s): {np.linalg.norm(robot.v):.5f}"
    robot_v_text = f"Robot linear vel(m/s): ({robot.v[0]:.3f}, {robot.v[1]:.3f})"
    robot_w_text = f"Robot angular vel(rad/s): {robot.w:.3f}"
    robot_pos_text = f"Robot pos(m): ({robot.pos[0]:.2f}, {robot.pos[1]:.2f})"
    robot_theta_text = f"Robot angle(rad): {robot.theta:.2f}"

    text_surface_speed = font.render(speed_text, True, (0, 0, 0))
    text_surface_pos = font.render(pos_text, True, (0, 0, 0))
    text_surface_robot_speed = font.render(robot_speed_text, True, (0, 0, 0))
    text_surface_robot_pos = font.render(robot_pos_text, True, (0, 0, 0))
    text_surface_robot_v = font.render(robot_v_text, True, (0, 0, 0))
    text_surface_robot_w = font.render(robot_w_text, True, (0, 0, 0))
    text_surface_robot_theta = font.render(robot_theta_text, True, (0, 0, 0))

    text_rect_speed = text_surface_speed.get_rect(
        bottomright=(meters_to_pixels(WIDTH) - 30, meters_to_pixels(HEIGHT) - 20)
    )
    text_rect_pos = text_surface_pos.get_rect(
        bottomright=(meters_to_pixels(WIDTH) - 30, meters_to_pixels(HEIGHT) - 50)
    )
    text_rect_robot_speed = text_surface_robot_speed.get_rect(
        bottomleft=(30, meters_to_pixels(HEIGHT) - 20)
    )
    text_rect_robot_pos = text_surface_robot_pos.get_rect(
        bottomleft=(30, meters_to_pixels(HEIGHT) - 50)
    )
    text_rect_robot_v = text_surface_robot_v.get_rect(
        bottomleft=(30, meters_to_pixels(HEIGHT) - 80)
    )
    text_rect_robot_w = text_surface_robot_w.get_rect(
        bottomleft=(30, meters_to_pixels(HEIGHT) - 110)
    )
    text_rect_robot_theta = text_surface_robot_theta.get_rect(
        bottomleft=(30, meters_to_pixels(HEIGHT) - 140)
    )

    screen.blits(
        (
            (text_surface_speed, text_rect_speed),
            (text_surface_pos, text_rect_pos),
            (text_surface_robot_theta, text_rect_robot_theta),
            (text_surface_robot_pos, text_rect_robot_pos),
            (text_surface_robot_v, text_rect_robot_v),
            (text_surface_robot_w, text_rect_robot_w),
            (text_surface_robot_speed, text_rect_robot_speed),
        )
    )


# Drawing a circle radius around the agent when standing still
def draw_transparent_circle(surface, center, radius):
    target_rect = pygame.Rect(center, (0, 0)).inflate((radius * 2, radius * 2))
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.circle(shape_surf, (255, 0, 0, 75), (radius, radius), radius)
    surface.blit(shape_surf, target_rect)


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
                agent.future_trajectory[:, -2:],
                agent.past_trajectory[:, -2:],
                screen
            )  # passing in the last two positions of predicted trajectory
            robot.update(u_next)
            last_sample_time = current_time

        # Draws a circle the radius of what the distance is when the agent is standing still
        draw_transparent_circle(
            screen,
            np.vectorize(meters_to_pixels)(agent.pos),
            int(meters_to_pixels(STAND_RADIUS)),
        )
        agent.draw(screen)
        robot.draw(screen)
        display_text(agent)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()
