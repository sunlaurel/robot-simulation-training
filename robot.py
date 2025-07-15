import pygame
import math
from utils import *
from simulation_helper import *

MAX_W = 10  # max angular speed (rad/s)
MAX_V = 1.5  # max linear speed (m/s)
STAND_RADIUS = 1  # if human is stopped, robot stay 1m away from human
WALK_RADIUS = 1.5  # if human is walking, robot moves 1.5m away from human


class Robot:

    def __init__(
        self,
        x=2,
        y=5,
        target_x=2,
        target_y=5,
        theta=0,  # default, the robot is facing right
        width=0.6,
        height=0.9,
        dt=8.33 / 1000,
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

    def policy(self, target, past_pos, screen):
        agent_pos = past_pos[:, -1]
        self.target_pos = target[:, -1]
        alpha = 1
        past_target = self.target_pos

        X = np.array(self.pos) - agent_pos
        # drawing a line for the line from the robot to the agent
        pygame.draw.line(
            screen,
            (0, 255, 0),
            (meters_to_pixels(agent_pos[0]), meters_to_pixels(agent_pos[1])),
            (
                meters_to_pixels(self.pos[0]),
                meters_to_pixels(self.pos[1]),
            ),
            5,
        )
        # breakpoint()

        present_tangent = past_pos[:, -1] - past_pos[:, -2]
        if np.linalg.norm(present_tangent) == 0:
            present_tangent += 1e-02
        # drawing a line for the present tangent line
        pygame.draw.line(
            screen,
            (255, 0, 0),
            (meters_to_pixels(agent_pos[0]), meters_to_pixels(agent_pos[1])),
            (
                meters_to_pixels(agent_pos[0] + present_tangent[0]),
                meters_to_pixels(agent_pos[1] + present_tangent[1]),
            ),
            5,
        )
        future_tangent = target[:, -1] - target[:, -2]
        if np.linalg.norm(future_tangent) == 0:
            future_tangent += 1e-02
        # drawing a line for the future tangent line
        pygame.draw.line(
            screen,
            (255, 0, 0),
            (meters_to_pixels(target[0][-1]), meters_to_pixels(target[1][-1])),
            (
                meters_to_pixels(target[0][-1] + future_tangent[0]),
                meters_to_pixels(target[1][-1] + future_tangent[1]),
            ),
            5,
        )

        present_perp = np.array([-present_tangent[1], present_tangent[0]])
        present_perp /= np.linalg.norm(present_perp)
        # drawing a line for the present perpendicular line
        pygame.draw.line(
            screen,
            (0, 0, 255),
            (meters_to_pixels(agent_pos[0]), meters_to_pixels(agent_pos[1])),
            (
                meters_to_pixels(agent_pos[0] + present_perp[0]),
                meters_to_pixels(agent_pos[1] + present_perp[1]),
            ),
            5,
        )
        future_perp = np.array([-future_tangent[1], future_tangent[0]])
        future_perp /= np.linalg.norm(future_perp)
        # drawing a line for the future perpendicular line
        pygame.draw.line(
            screen,
            (0, 0, 255),
            (meters_to_pixels(target[0][-1]), meters_to_pixels(target[1][-1])),
            (
                meters_to_pixels(target[0][-1] + future_perp[0]),
                meters_to_pixels(target[1][-1] + future_perp[1]),
            ),
            5,
        )

        # breakpoint()
        t = X @ present_perp
        offset = t * future_perp
        offset /= np.linalg.norm(offset)
        offset *= STAND_RADIUS

        self.target_pos += offset

        self.target_pos[0] = alpha * self.target_pos[0] + (1 - alpha) * past_target[0]
        self.target_pos[1] = alpha * self.target_pos[1] + (1 - alpha) * past_target[1]

        # calculating the angles and where the robot should move
        direction = np.array(self.target_pos) - self.pos
        distance = np.linalg.norm(direction)
        target_angle = math.atan2(
            direction[1], direction[0]
        )  # angle is + for cw and - for ccw

        angle_diff = self.angle_difference(target_angle)

        # calculating the angular velocity
        if abs(angle_diff) > MAX_W:
            w = MAX_W if angle_diff > 0 else -MAX_W
        else:
            w = angle_diff

        # calculating the linear velocity
        if distance < MAX_V:
            # if within the stopping range, then moves incrementally closer to the target
            v = distance * np.array(
                [math.cos(self.theta + w * self.dt), math.sin(self.theta + w * self.dt)]
            )

            if np.linalg.norm(v) < 1e-01:
                v = np.array([0.0, 0.0])
                w = 0.0
        else:
            v = MAX_V * np.array(
                [math.cos(self.theta + w * self.dt), math.sin(self.theta + w * self.dt)]
            )

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
        # Draws the robot on the screen as a gray rectangle
        dx = self.height / 2
        dy = self.width / 2
        nose_dx = self.width * 0.3 / 2
        nose_dy = self.height * 0.5 / 2
        corners = np.array(
            [
                (-dx, -dy),  # top left
                (dx, -dy),  # top right
                (dx + nose_dx, -nose_dy),  # top left of nose
                (dx + nose_dx, nose_dy),  # top right of nose
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
