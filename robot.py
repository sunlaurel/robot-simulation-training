import pygame
import math
from utils import *
from simulation_helper import *

MAX_W = 1  # max angular speed (rad/s)
MAX_V = 1.5  # max linear speed (m/s)
STOP_DISTANCE = 0.1       # threshold distance to stop near target
THETA_TOLERANCE = 0.349   # threshold angle range in radians

class Robot:

    def __init__(
        self,
        x=2,
        y=5,
        target_x=2,
        target_y=5,
        theta=0,            # default, the robot is facing right
        width=0.9,
        height=0.6,
        dt=0.833/1000
    ):
        self.pos = [x, y]
        self.target_pos = [target_x, target_y]
        self.theta = theta  # angle that the robot's facing
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
        # robot is constantly trying to get closer to the target
        v = MAX_V * np.array([math.cos(self.theta), math.sin(self.theta)])
        w = 0.0
        self.target_pos = target[:, -1]

        # calculating the target position based on the future trajectory
        target_v = (target[:, -1] - target[:, 0]) / self.dt
        target_speed = np.linalg.norm(target_v)

        # calculating the angles and where the robot should move
        direction = np.array(self.target_pos) - self.pos
        target_angle = math.atan2(
            direction[1], direction[0]
        )  # y is flipped for screen coordinates

        diff = self.angle_difference(target_angle)

        # breakpoint()

        # adjusting robot's target position based on walking speed
        if target_speed > 1:
            # if target's speed is greater than 1 m/s, then offset is 1.25 m to the right
            offset = 1.25 / target_speed * np.array([-target_v[1], target_v[0]])
            self.target_pos += offset
        else:  # if target's speed is less than 1 m/s (essentially standing still), then offset is 1 m to the right
            offset = 1 / target_speed * np.array([-target_v[1], target_v[0]])
            self.target_pos += offset

        # checking if the distance is within the stopping distance of the target position
        if np.linalg.norm(direction) > STOP_DISTANCE:
            # rotating towards the target
            if abs(diff) > MAX_W:
                w = MAX_W if diff > 0 else -MAX_W
            else:
                w = diff

            # # moving toward target if facing close to target
            # if abs(diff) < THETA_TOLERANCE:
            #     v = MAX_V * np.array([math.cos(self.theta), math.sin(self.theta)])
        else:  # if the robot gets to the target position, reorient to face the same way as the person
            print("reached the target position")
            breakpoint()
            v = np.array([0.0, 0.0])
            w = MAX_W if diff > 0 else -MAX_W

        return [v, w]

    def update(self, u):
        v, w = u
        # debugging purposes
        self.v = v
        self.w = w
        self.pos += v * self.dt
        self.theta += w * self.dt
        if self.theta > 2 * math.pi:
            self.theta = self.theta - 2 * math.pi
        elif self.theta < 0:
            self.theta = self.theta + 2 * math.pi
        # self.theta %= 2 * math.pi  # normalizing angle to 0 to 2 pi

    # Adding event handlers for arrow keys to adjust robot's velocity and position
    def handle_event(self, event):
        # TODO: later, update this to change frame rate in main loop
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_d:
                self.update([np.array([0.2, 0.0]), 0.0], self.dt)
            elif event.key == pygame.K_a:
                self.update([np.array([-0.2, 0.0]), 0.0], self.dt)
            elif event.key == pygame.K_s:
                self.update([np.array([0.0, 0.2]), 0.0], self.dt)
            elif event.key == pygame.K_w:
                self.update([np.array([0.0, -0.2]), 0.0], self.dt)
            elif event.key == pygame.K_e:
                self.update([np.array([0.0, 0]), 0.3], self.dt)
            elif event.key == pygame.K_q:
                self.update([np.array([0.0, 0.0]), -0.3], self.dt)
        elif event.type == pygame.KEYUP:
            self.update([np.array([0.0, 0.0]), 0.0], self.dt)

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

        # Draws the robot's target on the screen as a dark green circle
        # drawing the target on the agent for now
        pygame.draw.circle(
            surface,
            (5, 59, 14),
            (int(meters_to_pixels(self.target_pos[0])), int(meters_to_pixels(self.target_pos[1]))),
            meters_to_pixels(0.15),
        )
