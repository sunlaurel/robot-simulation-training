import pygame
import sys
import numpy as np
from utils import models

""" Constants """
WIDTH, HEIGHT = 10, 10
BG_COLOR = (255, 255, 255)
LINE_COLOR = (255, 0, 0)
FPS = 30


""" Functions """
# TODO: put these in a different script later on
def meters_to_pixels(m):
    """Converts from meters to pixels to display on screen"""
    return m * 100


def pixels_to_meters(p):
    """Converts from pixels to meters"""
    return p / 100


def convert_to_tuple_list(lst):
    """Converts a 2XN array to a list of tuples"""
    pos_lst = []
    for i in range(len(lst[0])):
        pos_lst.append((lst[0][i], lst[1][i]))

    return pos_lst


""" Person Agent Class """
class Agent:
    def __init__(self, x=2, y=5, radius=0.5, color=(0, 0, 255), N_past=17):
        self.pos = [x, y]
        self.radius = radius
        self.color = color
        self.N_past = N_past  # default sampling the last two seconds
        self.dragging = False
        self.offset = [0, 0]
        self.past_trajectory = np.array(
            (np.full(N_past, x), np.full(N_past, y)), dtype=object
        )

    def draw(self, surface):
        pygame.draw.lines(
            surface,
            color=LINE_COLOR,
            closed=False,
            points=convert_to_tuple_list(
                np.vectorize(meters_to_pixels)(self.past_trajectory)
            ),
        )

        for (x, y) in convert_to_tuple_list(self.past_trajectory):
            pygame.draw.circle(
                surface,
                LINE_COLOR,
                (
                    int(meters_to_pixels(x)),
                    int(meters_to_pixels(y)),
                ),
                radius=5,
            )
            
        pygame.draw.circle(
            surface,
            self.color,
            (int(meters_to_pixels(self.pos[0])), int(meters_to_pixels(self.pos[1]))),
            meters_to_pixels(self.radius),
        )

    def update(self, x, y):
        self.past_trajectory[:, :-1] = self.past_trajectory[:, 1:]
        self.past_trajectory[0][-1] = x
        self.past_trajectory[1][-1] = y

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            dx = mouse_x - meters_to_pixels(self.pos[0])
            dy = mouse_y - meters_to_pixels(self.pos[1])
            if dx**2 + dy**2 <= meters_to_pixels(self.radius) ** 2:
                self.dragging = True
                self.offset = [dx, dy]
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                mouse_x, mouse_y = event.pos
                self.pos = [
                    pixels_to_meters(mouse_x - self.offset[0]),
                    pixels_to_meters(mouse_y - self.offset[1]),
                ]


""" Initializing the game + setup """
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

    # displaying the median speed of the trajectory on the screen
    speeds = (
        np.sum(
            (agent.past_trajectory[:, :-1] - agent.past_trajectory[:, 1:]) ** 2, axis=0
        )
        ** 1
        / 2
    )
    median_speed = np.median(speeds / 0.12)
    speed_text = f"Median speed: {median_speed:.10f} m/s"
    text_surface = font.render(speed_text, True, (0, 0, 0))
    text_rect = text_surface.get_rect(
        bottomright=(meters_to_pixels(WIDTH) - 20, meters_to_pixels(HEIGHT) - 10)
    )
    screen.blit(text_surface, text_rect)

    agent.draw(screen)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
