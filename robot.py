import pygame
import math
import re
from utils import *
from simulation_helper import *

#################################################################
##   Overview                                                  ##
##   - renders the robot on the screen                         ##
##   - contains the policy of the robot                        ##
##   - the model predicts the future position that the robot   ##
##     should be in the future                                 ##
##   - change save_path to change what model is running in     ##
##     the simulation                                          ##
#################################################################


###########  Initializing constants and global variables  ###########
MAX_W = 6  # max angular speed (rad/s)
MAX_V = 1.5  # max linear speed (m/s)
RADIUS = 1.25  # if human is stopped, robot stay 1m away from human

global v_last  # last significant velocity
v_last = np.array([1.0, 0.0])


##############  Creating the Robot class  #################
class Robot:

    def __init__(
        self,
        x=2,
        y=5,
        target_x=2,
        target_y=5,
        theta=0,  # default, the robot is facing right
        display_body=True,
        display_target=True,
        meters_to_pixels=meters_to_pixels,
        pixels_to_meters=pixels_to_meters,
        save_path="./weights/best-weights-robot/(copy-person)best_weight.pth",
        width=0.6,
        height=0.9,
        dt=8.33 / 1000,
    ):
        self.pos = [x, y]
        self.target_pos = [target_x, target_y]
        self.theta = theta  # angle that the robot is facing
        self.display_body = display_body
        self.display_target = display_target
        self.meters_to_pixels = meters_to_pixels
        self.pixels_to_meters = pixels_to_meters
        self.width = width
        self.height = height
        self.dt = dt
        self.v = np.array([0.1, 0.1])  # initial linear velocity
        self.w = 0  # initial angular velocity

        """ Importing constants from config file """
        with open("./utils/config.json", "r", encoding="utf-8") as config:
            data = json.load(config)

        self.N_future = data["future-steps"]
        self.N_past = data["past-steps"]

        # initializing the model
        # save_path = "./weights/best-weights-robot/(big-room)best_weight.pth"

        # setting offset + scale flag
        self.offset = "offset" in save_path
        self.scale = "scale" in save_path

        # extracting the number of past steps from the file name
        match = re.search(r"\((\d+)-past\)", save_path)
        if match:
            self.N_past = int(match.group(1))

        self.model = models.MultiLayerRobot(4 * self.N_past, 100, 100, 2)
        # self.model.load_state_dict(torch.load(save_path, weights_only=True))
        # self.model.load_state_dict(torch.load(save_path))
        self.model.load_state_dict(
            torch.load(save_path, map_location=torch.device("cpu"))
        )  # use this for non-gpu machines

        self.past_trajectory = np.array(
            (np.full(self.N_past, x), np.full(self.N_past, y)),
            dtype=float,  # storing past trajectories
        )

        self.past_headings = np.array(
            (
                np.full(self.N_past, math.cos(theta)),
                np.full(self.N_past, math.sin(theta)),
            ),
            dtype=float,  # storing past headings
        )

    def angle_difference(self, target_angle):
        # forcing the angle to be between -pi to pi
        diff = target_angle - self.theta
        return (diff + math.pi) % (2 * math.pi) - math.pi

    def predict_goal(self, X_past):
        # INPUT: X_past is the list of past positions of the agent in world space
        # RETURN: returns the projected goal for the robot to go to in world space
        relative_vectors_r, robot_velocities_r, _ = ConvertAbsoluteToRobotFrame(
            X_past,
            self.past_trajectory,
            self.past_headings,
            np.zeros(2),
        )

        input_vectors = ProcessPast(relative_vectors_r, robot_velocities_r)

        with torch.no_grad():
            target = self.model(
                torch.tensor(input_vectors).float().unsqueeze(0)
            ).squeeze()

        target = self.predict_goal(input_vectors)

        self.target_pos = (
            ConvertRobotFrameToAbsolute(
                np.array([math.cos(self.theta), math.sin(self.theta)]), target
            )
            + self.past_trajectory[:, -1]
        )

        return self.target_pos

    def policy(self, X_past):
        # TODO: move everything so that the policy is only being done relative to the robot
        global v_last, target_pos
        # alpha = 0.2
        epsilon = 5e-02

        ##############  Calculating the target position from the model #############
        self.predict_goal(X_past)

        #############  Robot Behavior  ################
        # calculating the angles and where the robot should move
        # breakpoint()
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
        if distance < RADIUS:
            # if within a certain radius, decouple angular velocity and linear velocity
            v = direction
            target_angle = math.atan2(
                v_last[1], v_last[0]
            )  # angle is + for cw and - for ccw

            angle_diff = self.angle_difference(target_angle)

            # calculating the angular velocity
            if abs(angle_diff) > MAX_W:
                w = MAX_W if angle_diff > 0 else -MAX_W
            else:
                w = angle_diff
        elif distance < MAX_V:
            # if within the stopping range, then moves incrementally closer to the target
            v = (
                1.5
                * distance
                * np.array(
                    [
                        math.cos(self.theta + w * self.dt),
                        math.sin(self.theta + w * self.dt),
                    ]
                )
            )

            if np.linalg.norm(v) < 0.5:
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

        """ Updating the target position and headings from the new positions """
        self.past_trajectory[:, :-1] = self.past_trajectory[:, 1:]
        self.past_trajectory[0][-1] = self.pos[0]
        self.past_trajectory[1][-1] = self.pos[1]

        self.past_headings[:, :-1] = self.past_headings[:, 1:]
        self.past_headings[0][-1] = math.cos(self.theta)
        self.past_headings[1][-1] = math.sin(self.theta)

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
        global target_pos  # sanity check

        if self.display_body:
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
                        self.meters_to_pixels(self.pos[0] + rx),
                        self.meters_to_pixels(self.pos[1] + ry),
                    )
                )

            pygame.draw.polygon(
                surface,
                (128, 128, 128),
                rotated,
            )

        if self.display_target:
            # Draws the robot's target on the screen as a dark green circle
            pygame.draw.circle(
                surface,
                (5, 59, 14),
                (
                    int(self.meters_to_pixels(self.target_pos[0])),
                    int(self.meters_to_pixels(self.target_pos[1])),
                ),
                self.meters_to_pixels(0.15),
            )
