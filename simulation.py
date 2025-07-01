import pygame
import sys
import numpy as np
import torch
from utils import models
from training import past_steps, future_steps
# from baseline_models import baseline_model

""" Constants """
WIDTH, HEIGHT = 10, 10
BG_COLOR = (255, 255, 255)
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


def T(X_past, X_current, offset=True, scale=True):
    """Transforms the data"""
    X_update = X_past.copy()
    if offset:
        # TODO: later, replace with more efficient code
        X_update = torch.stack(
            (
                torch.tensor(np.array(X_past[0] - X_current[0], dtype=float)),
                torch.tensor(np.array(X_past[1] - X_current[1], dtype=float)),
            )
        )

    if scale:
        X_update /= 2
        
    return X_update


def T_inv(X_future, X_current, offset=True, scale=True):
    """Reverses the transformation to display on screen"""
    X_update = X_future.detach().numpy().copy()
    if scale:
        X_future *= 2

    if offset:
        # TODO: later, replace with more efficient code
        X_update = torch.stack(
            (
                torch.tensor(X_future[0] + X_current[0]),
                torch.tensor(X_future[1] + X_current[1]),
            )
        )

    return X_update


""" Person Agent Class """
class Agent:

    def __init__(
        self,
        N_past,
        N_future,
        x=2,
        y=5,
        radius=0.5,
        epsilon=0.005,
        sigma_max=0.1,
    ):
        self.pos = [x, y]
        self.radius = radius
        self.N_past = N_past  # default sampling the last two seconds
        self.N_future = N_future  # default predicting two seconds into the future
        self.dragging = False
        self.offset = [0, 0]
        self.past_trajectory = np.array(
            (np.full(N_past, x), np.full(N_past, y)),
            dtype=float,  # storing past trajectories
        )
        self.future_trajectory = np.array(
            (np.full(N_future, x), np.full(N_future, y)),
            dtype=float,  # storing future trajectories
        )

        # initializing the model
        self.model = models.MultiLayer(2 * N_past, 100, 100, N_future * 2)
        save_path = "./best-weights/best_weight_noise_scale_offset.pth"
        self.model.load_state_dict(torch.load(save_path, weights_only=True))

        # defining the parameters for adding noise to the past trajectory
        self.sigma = 0
        self.epsilon = epsilon
        self.sigma_max = sigma_max


    def draw(self, surface):
        """Draws the agent on the screen, with a blue circle for the agent, a red line for the agent's past trajectory, 
        and a green line for the agent's predicted trajectory based on the trained model

        Args:
            surface (surface): the surface that the components will be drawn on
        """

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

        # for x, y in convert_to_tuple_list(self.future_trajectory):
        #     pygame.draw.circle(
        #         surface,
        #         (0, 255, 0),
        #         (
        #             int(meters_to_pixels(x)),
        #             int(meters_to_pixels(y)),
        #         ),
        #         radius=5,
        #     )

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
        # breakpoint()
        N = np.random.randn(2, self.N_past) * self.sigma
        self.past_trajectory = self.past_trajectory + N
        # breakpoint()

        # Passing past trajectories to model to predict future trajectories
        X_ego_past = T(self.past_trajectory, self.pos)
        X_ego_future = self.model(torch.tensor(X_ego_past).float().unsqueeze(0))
        self.future_trajectory = T_inv(X_ego_future.squeeze(), self.pos)

        # # calculating past velocity
        # past_velocity = (self.past_trajectory[:, :-1] - self.past_trajectory[:, 1:]) / 0.12
        # past_velocity = np.append(past_velocity, ((self.past_trajectory[:, -1] - [x, y]) / 0.12).T[:, None], axis=1)
        # # predicting the future paths with the updated past trajectory
        # past_trajectory = torch.cat((torch.tensor(self.past_trajectory), torch.tensor(past_velocity)), dim=0)
        # X_ego_future = self.model(X_ego_past.unsqueeze(0))
        # self.future_trajectory = T_inv(X_ego_future.squeeze(), self.pos)[:2]

    # Adding event handlers for arrow keys to ajust noise
    def on_arrow_down(self):
        self.sigma = max(self.sigma - self.epsilon, 0)
        print("sigma gone down:", self.sigma)

    def on_arrow_up(self):
        self.sigma = min(self.sigma_max, self.sigma + self.epsilon)
        print("sigma gone up:", self.sigma)

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


""" Initializing the game + setup """
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((meters_to_pixels(WIDTH), meters_to_pixels(HEIGHT)))
    pygame.display.set_caption("Trajectory Prediction Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)
    agent = Agent(x=2, y=5, N_past=past_steps, N_future=future_steps)
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

        # calculating the speed of the agent
        speeds = (
            np.sum(
                (agent.past_trajectory[:, :-1] - agent.past_trajectory[:, 1:]) ** 2,
                axis=0,
            )
            ** 0.5
        )

        # rendering the speed and position on screen
        median_speed = np.median(speeds / 0.12)
        speed_text = f"Median speed(m/s): {median_speed:.5f}"
        pos_text = f"Position(m): ({agent.pos[0]:.2f}, {agent.pos[1]:.2f})"

        text_surface_speed = font.render(speed_text, True, (0, 0, 0))
        text_surface_pos = font.render(pos_text, True, (0, 0, 0))

        text_rect_speed = text_surface_speed.get_rect(
            bottomright=(meters_to_pixels(WIDTH) - 30, meters_to_pixels(HEIGHT) - 20)
        )
        text_rect_pos = text_surface_pos.get_rect(
            bottomright=(meters_to_pixels(WIDTH) - 30, meters_to_pixels(HEIGHT) - 50)
        )

        screen.blits(
            ((text_surface_speed, text_rect_speed), (text_surface_pos, text_rect_pos))
        )

        agent.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()
