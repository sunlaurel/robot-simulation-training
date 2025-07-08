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
        width=0.9,
        height=0.7,
    ):
        self.pos = [x, y]
        self.theta = theta  # angle that the robot's facing
        self.width = width
        self.height = height
        self.v = np.array([0.1, 0.1])  # initial linear velocity
        self.w = 0.052  # initial angular velocity

    def angle_difference(self, target_angle):
        # forcing the angle to be between -pi to pi
        return (target_angle - self.theta + math.pi) % (2 * math.pi) - math.pi

    def policy(self, target_pos):
        v = np.array([0.0, 0.0])
        w = 0.0
        # breakpoint()

        direction = np.array(target_pos) - self.pos

        if direction[0] ** 2 + direction[1] ** 2 > 0.01:
            target_angle = math.atan2(
                -direction[1], direction[0]
            )  # y is flipped for screen coordinates
            diff = self.angle_difference(target_angle)

            # rotating towards the target
            if abs(diff) > self.w:
                w = self.w if diff > 0 else -self.w
            else:
                self.theta = target_angle  # snap to the right angle if close enough

            # moving toward target if facing close to target
            if abs(diff) < 0.349:
                v = self.v * np.array([math.cos(self.theta), -math.sin(self.theta)])

        return [v, w]

    def update(self, u, dt):
        # breakpoint()
        v, w = u
        self.v += v
        self.w += w
        self.pos = self.pos + self.v * dt
        self.theta = self.theta + self.w * dt
        self.theta %= 2 * math.pi  # normalizing angle to 0 to 2 pi

    # Adding event handlers for arrow keys to adjust robot's velocity and position
    def handle_event(self, event, dt):
        # TODO: later, update this to change frame rate in main loop
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
