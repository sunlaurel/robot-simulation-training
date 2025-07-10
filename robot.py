import pygame
import math
from utils import *
from simulation_helper import *

MAX_W = 10  # max angular speed (rad/s)
MAX_V = 10  # max linear speed (m/s)
STOP_DISTANCE = 0.05  # threshold distance to stop near target
STOP_AND_TURN_RADIUS = 2  # threshold distance for robot to stop and turn to the target
SLOW_AND_TURN_RADIUS = 5  # threshold distance for robot to slow down enough to turn
STAND_RADIUS = 1  # if human is stopped, robot stay 1m away from human
WALK_RADIUS = 1.5  # if human is walking, robot moves 1.5m away from human
THETA_TOLERANCE = 0.5  # threshold angle range in radians


class Robot:

    def __init__(
        self,
        x=2,
        y=5,
        target_x=2,
        target_y=5,
        theta=0,  # default, the robot is facing right
        width=0.9,
        height=0.6,
        dt=0.833 / 1000,
    ):
        self.pos = [x, y]
        self.target_pos = [target_x, target_y]
        self.theta = theta  # angle that the robot is facing
        self.width = width
        self.height = height
        self.dt = dt
        self.v = np.array([0.1, 0.1])  # initial linear velocity
        self.w = 0  # initial angular velocity

    def angle_difference(self, target_angle):
        # forcing the angle to be between -pi to pi
        diff = target_angle - self.theta
        return (diff + math.pi) % (2 * math.pi) - math.pi

    def policy(self, target):
        # TODO: have the robot avoid the person as much as possible
        # TODO: an issue is that the robot needs to slow down when out of range to follow the person again
        # TODO: fix the offset target position to be on person's right
        # TODO: fix the robot to turn where the person is oriented
        # TODO: fix on #84 so that it would more smoothly track the person's position when it's close to it
        w = 0.0
        breakpoint()
        self.target_pos = target[:, -1]

        # calculating the target position based on the future trajectory
        target_v = (target[:, -1] - target[:, 0]) / self.dt
        print("target v:", target_v)
        target_speed = np.linalg.norm(target_v)

        # adjusting offset --> 1.25m to the right if speed > 1 m/s, or 1m if speed < 1m/s (standing still)
        offset = (
            (WALK_RADIUS if target_speed > 1 else STAND_RADIUS)
            * np.array([-target_v[1], target_v[0]])
            / target_speed
        )
        print("offset:", offset)
        self.target_pos += offset

        # calculating the angles and where the robot should move
        direction = np.array(self.target_pos) - self.pos
        distance = np.linalg.norm(direction)
        target_angle = math.atan2(
            direction[1], direction[0]
        )  # y is flipped for screen coordinates

        diff = self.angle_difference(target_angle)

        v = MAX_V * np.array([math.cos(self.theta + w), math.sin(self.theta + w)])

        # checking if the distance is within the stopping distance of the target position
        if distance < STOP_AND_TURN_RADIUS:
            # breakpoint()
            if distance < STOP_DISTANCE or abs(diff) > THETA_TOLERANCE:
                v = np.array(
                    [0.0, 0.0]
                )
        elif (
            distance < SLOW_AND_TURN_RADIUS
        ):  # slows down the robot enough to be able to turn
            v *= 0.5

        if abs(diff) > MAX_W:
            w = MAX_W if diff > 0 else -MAX_W
        else:
            w = diff

        return [v, w]

    def update(self, u):
        v, w = u
        self.v = v  # debugging purposes
        self.w = w  # debugging purposes
        self.theta += w * self.dt
        if self.theta > 2 * math.pi:
            self.theta = self.theta - 2 * math.pi
        elif self.theta < 0:
            self.theta = self.theta + 2 * math.pi
        self.pos += v * self.dt

    # Adding event handlers for arrow keys to adjust robot's velocity and position
    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_d:
                self.update([np.array([0.2, 0.0]), 0.0])
            elif event.key == pygame.K_a:
                self.update([np.array([-0.2, 0.0]), 0.0])
            elif event.key == pygame.K_s:
                self.update([np.array([0.0, 0.2]), 0.0])
            elif event.key == pygame.K_w:
                self.update([np.array([0.0, -0.2]), 0.0])
            elif event.key == pygame.K_e:
                self.update([np.array([0.0, 0]), 0.3])
            elif event.key == pygame.K_q:
                self.update([np.array([0.0, 0.0]), -0.3])
        elif event.type == pygame.KEYUP:
            self.update([np.array([0.0, 0.0]), 0.0])

    def draw(self, surface):
        dx = self.width / 2
        dy = self.height / 2
        corners = np.array(
            [
                (-dx, -dy),  # top left
                (dx, -dy),  # top right
                (dx, dy),  # bottom right
                (-dx, dy),  # bottom left
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

        # Draws the robot's target on the screen as a dark green circle
        # drawing the target on the agent for now
        pygame.draw.circle(
            surface,
            (5, 59, 14),
            (
                int(meters_to_pixels(self.target_pos[0])),
                int(meters_to_pixels(self.target_pos[1])),
            ),
            meters_to_pixels(0.15),
        )
