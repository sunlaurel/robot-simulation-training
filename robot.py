import pygame
import math
from utils import *
from simulation_helper import *


class Robot:

    def __init__(
        self,
        x=2,
        y=5,
        theta=0,
        width=0.7,
        height=0.9,
    ):
        self.pos = [x, y]
        self.theta = theta  # angle that the robot's facing
        self.width = width
        self.height = height
        # self.dragging = False
        # self.offset = [0, 0]

    def draw(self, surface):
        dx = self.width / 2
        dy = self.height / 2
        corners = np.array(
            [
                (-dx, -dy),
                (dx, -dy),
                (dx, dy),
                (-dx, dy),
            ]
        )

        rotated = []
        for x, y in corners:
            rx = x * math.cos(self.theta) - y * math.sin(self.theta)
            ry = x * math.sin(self.theta) + y * math.cos(self.theta)
            rotated.append(
                (
                    int(meters_to_pixels(self.pos[0] + rx)),
                    int(meters_to_pixels(self.pos[1] + ry)),
                )
            )

        # Draws the robot on the screen as a gray rectangle
        pygame.draw.polygon(
            surface,
            (128, 128, 128),
            rotated,
        )

    def update(self, x, y):
        pass

    # Adding event handlers for arrow keys to ajust noise
    def on_arrow_down(self):
        self.sigma = max(self.sigma - self.epsilon, 0)

    def on_arrow_up(self):
        self.sigma = min(self.sigma_max, self.sigma + self.epsilon)

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
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.on_arrow_up()
            elif event.key == pygame.K_DOWN:
                self.on_arrow_down()
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                mouse_x, mouse_y = event.pos
                self.pos = [
                    pixels_to_meters(mouse_x - self.offset[0]),
                    pixels_to_meters(mouse_y - self.offset[1]),
                ]
