import json
import re
import pygame
from utils import *
from simulation_helper import *

########################################################
##  Overview                                          ##
##  - renders the agent (the person) on the screen    ##
##  - if display_flag is on, then will display the    ##
##    model's predicted trajectory on the screen      ##
########################################################


class Agent:

    def __init__(
        self,
        x=2,
        y=5,
        radius=0.5,
        epsilon=0.005,
        sigma_max=0.1,
        display_flag=False,
    ):
        self.pos = [x, y]
        self.radius = radius
        self.dragging = False
        self.offset_pos = [0, 0]
        self.sigma = 0
        self.epsilon = epsilon
        self.sigma_max = sigma_max
        self.display_flag = display_flag  # if display_flag is on, then predicts and displays predictions of the agent's future position

        """ Importing constants from the config file """
        with open("./utils/config.json", "r", encoding="utf-8") as file:
            data = json.load(file)

        self.N_future = data["future-steps"]
        self.N_past = data["past-steps"]

        if display_flag:
            # initializing the model
            save_path = "./weights/best-weights/best_weight.pth"

            # setting offset + scale flag
            self.offset = "offset" in save_path
            self.scale = "scale" in save_path

            # extracting the number of past steps from the file name
            match = re.search(r"\((\d+)-past\)", save_path)
            if match:
                self.N_past = int(match.group(1))

            self.model = models.MultiLayer(2 * self.N_past, 100, 100, self.N_future * 2)
            self.model.load_state_dict(torch.load(save_path, weights_only=True))

        self.past_trajectory = np.array(
            (np.full(self.N_past, x), np.full(self.N_past, y)),
            dtype=float,  # storing past trajectories
        )

        self.future_trajectory = np.array(
            (np.full(self.N_future, x), np.full(self.N_future, y)),
            dtype=float,  # storing future trajectories
        )

    def draw(self, surface):
        # Draws the agent on the screen as a blue circle
        pygame.draw.circle(
            surface,
            (0, 0, 255),
            (int(meters_to_pixels(self.pos[0])), int(meters_to_pixels(self.pos[1]))),
            meters_to_pixels(self.radius),
        )

        # Draws a red line for the agent's past trajectory
        pygame.draw.lines(
            surface,
            color=(255, 0, 0),
            width=2,
            closed=False,
            points=convert_to_tuple_list(
                np.vectorize(meters_to_pixels)(self.past_trajectory)
            ),
        )

        # Draws red dots showing the samples that were taken from the agent's past trajectory
        for x, y in convert_to_tuple_list(self.past_trajectory):
            pygame.draw.circle(
                surface,
                (255, 0, 0),
                (
                    int(meters_to_pixels(x)),
                    int(meters_to_pixels(y)),
                ),
                radius=5,
            )

        # if display flag is true, then will predict agent's future trajectory and display on screen
        if self.display_flag:
            # Draws green dots showing the model's prediction for the agent's future trajectory based on the past trajectory
            for i in range(0, 3):
                pygame.draw.circle(
                    surface,
                    (0, 255, 0),
                    (
                        int(meters_to_pixels(self.future_trajectory[0][i])),
                        int(meters_to_pixels(self.future_trajectory[1][i])),
                    ),
                    radius=5,
                )

            for i in range(3, len(convert_to_tuple_list(self.future_trajectory))):
                pygame.draw.circle(
                    surface,
                    (0, 153, 0),
                    (
                        int(meters_to_pixels(self.future_trajectory[0][i])),
                        int(meters_to_pixels(self.future_trajectory[1][i])),
                    ),
                    radius=5,
                )

            for x, y in convert_to_tuple_list(self.future_trajectory):
                pygame.draw.circle(
                    surface,
                    (0, 255, 0),
                    (
                        int(meters_to_pixels(x)),
                        int(meters_to_pixels(y)),
                    ),
                    radius=5,
                )

    def update(self, x, y):
        """When updating, it updates its past trajectory and then predicts a new path from the trained model

        Args:
            x (int): the new x position (in meters) that the agent is located on the screen
            y (int): the new y position (in meters) of the agent's location on the screen
        """
        # updating the past trajectories
        self.past_trajectory[:, :-1] = self.past_trajectory[:, 1:]
        self.past_trajectory[0][-1] = x
        self.past_trajectory[1][-1] = y

        # adding noise to the past trajectories
        N = np.random.randn(2, self.N_past) * self.sigma
        self.past_trajectory = self.past_trajectory + N

        if self.display_flag:
            # Passing past trajectories to model to predict future trajectories
            X_ego_past = T(
                self.past_trajectory,
                self.pos,
                offset=self.offset,
                scale=self.scale,
                scale_factor=self.model.scale_factor,
            )
            X_ego_future = self.model(torch.tensor(X_ego_past).float().unsqueeze(0))
            self.future_trajectory = T_inv(
                X_ego_future.squeeze(),
                self.pos,
                offset=self.offset,
                scale=self.scale,
                scale_factor=self.model.scale_factor,
            )

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
                self.offset_pos = [dx, dy]
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
                    pixels_to_meters(mouse_x - self.offset_pos[0]),
                    pixels_to_meters(mouse_y - self.offset_pos[1]),
                ]
