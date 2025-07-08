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
        self.theta = theta              # angle that the robot's facing
        self.width = width
        self.height = height
        self.v = np.array([0.0, 0.0])   # initial linear velocity
        self.w = 0                      # initial angular velocity
        # self.dragging = False
        # self.offset = [0, 0]

    def policy(self, target_pos):
        v = np.array([0.0, 0.0])
        w = 0.0
        # breakpoint()

        if self.pos[0] != target_pos[0] or self.pos[1] != target_pos[1]:
            if math.atan(target_pos[1] / target_pos[0]) != self.theta:
                w = 0.3 if self.theta - math.atan(target_pos[1] / target_pos[0]) < 0 else -0.3
            if self.pos[0] != target_pos[0]:
                v[0] = 0.2 if self.pos[0] - target_pos[0] else -0.2
            if self.pos[1] != target_pos[1]:
                v[1] = 0.2 if self.pos[1] - target_pos[1] else -0.2

        return [v, w]

    def update(self, u, dt):
        v, w = u
        self.v += v
        self.w += w
        # breakpoint()
        self.pos = self.pos + self.v * dt
        self.theta = self.theta + self.w * dt

    # Adding event handlers for arrow keys to adjust robot's velocity and position
    def handle_event(self, event, dt):
        # TODO: later, update this to change framerate in main loop
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_d:
                self.update([np.array([0.2, 0.0]), 0.0], dt)
            elif event.key == pygame.K_a:
                self.update([np.array([-0.2, 0.0]), 0.0], dt)
            elif event.key == pygame.K_s:
                self.update([np.array([0.0, 0.2]), 0.0], dt)
            elif event.key == pygame.K_w:
                self.update([np.array([0.0, -0.2]), 0.0], dt)
            elif event.key == pygame.K_e:
                self.update([np.array([0.0, 0]), 0.3], dt)
            elif event.key == pygame.K_q:
                self.update([np.array([0.0, 0.0]), -0.3], dt)
        elif event.type == pygame.KEYUP:
            self.update([-self.v, -self.w], dt)

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