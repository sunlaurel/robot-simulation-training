import pygame
import math
from utils import *
from simulation_helper import *

MAX_W = 0.5               # max angular speed (rad/s)
MAX_V = 2               # max linear speed (m/s)
STOP_DISTANCE = 0.1       # threshold distance to stop near target
THETA_TOLERANCE = 0.349   # threshold angle range in radians

class Robot:

    def __init__(
        self,
        x=2,
        y=5,
        theta=0,            # default, the robot is facing right
        width=0.9,
        height=0.6,
    ):
        self.pos = [x, y]
        self.theta = theta  # angle that the robot's facing
        self.width = width
        self.height = height
        self.v = np.array([0.1, 0.1])  # initial linear velocity
        self.w = 0  # initial angular velocity

    def angle_difference(self, target_angle):
        # forcing the angle to be between -pi to pi
        diff = target_angle - self.theta
        return (diff + math.pi) % (2 * math.pi) - math.pi

    def policy(self, target_pos):
        v = np.array([0.0, 0.0])
        w = 0.0

        direction = np.array(target_pos) - self.pos

        if np.linalg.norm(direction) > STOP_DISTANCE:
            target_angle = math.atan2(
                direction[1], direction[0]
            )  # y is flipped for screen coordinates


            diff = self.angle_difference(target_angle)

            # rotating towards the target
            if abs(diff) > MAX_W:
                w = MAX_W if diff > 0 else -MAX_W
            else:
                self.theta = target_angle  # snap to the right angle if close enough

            # moving toward target if facing close to target
            if abs(diff) < THETA_TOLERANCE:
                v = MAX_V * np.array([math.cos(self.theta), math.sin(self.theta)])

        return [v, w]

    def update(self, u, dt):
        v, w = u
        # debugging purposes
        self.v = v
        self.w = w
        self.pos += v * dt
        self.theta += w * dt
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
            self.update([np.array([0.0, 0.0]), 0.0], dt)

    def draw(self, surface):
        dx = self.width / 2
        dy = self.height / 2
        corners = np.array(
            [
                (-dx, -dy),  # top left
                (dx, -dy),   # top right
                (dx, dy),    # bottom right
                (-dx, dy),   # bottom left
            ]
        )

        rotated = []
        for x, y in corners:
            rx = x * math.cos(self.theta) - y * math.sin(self.theta)
            ry = x * math.sin(self.theta) + y * math.cos(self.theta)
            rotated.append(
                (
                    meters_to_pixels(self.pos[0] + rx),
                    meters_to_pixels(self.pos[1] + ry),
                )
            )

        # Draws the robot on the screen as a gray rectangle
        pygame.draw.polygon(
            surface,
            (128, 128, 128),
            rotated,
        )
