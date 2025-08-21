import numpy as np
import pygame

##############################################################
##   Overview                                               ##
##   - helper functions for the simulator                   ##
##############################################################


""" Functions """
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


def T(X_past, X_current, offset=True, scale=True, scale_factor=0.5):
    """Transforms the data"""
    X_update = X_past.copy()
    if offset:
        X_update = X_update - np.array(X_current)[:, None]

    if scale:
        X_update *= scale_factor

    return X_update


def T_inv(X_future, X_current, offset=True, scale=True, scale_factor=0.5):
    """Reverses the transformation to display on screen"""
    # breakpoint()
    X_update = X_future.clone().detach().numpy()
    if scale:
        X_update /= scale_factor

    if offset:
        X_update = X_update + np.array(X_current)[:, None]

    return X_update


# Displaying text on screen
def display_text(
    agent,
    robot,
    font,
    screen,
    sampling_interval_ms,
    width,
    height,
    meters_to_pixels=meters_to_pixels,
):
    # calculating the speed of the agent
    speeds = np.linalg.norm(
        agent.past_trajectory[:, :-1] - agent.past_trajectory[:, 1:],
        axis=0,
    ) / sampling_interval_ms

    # rendering the speed and position on screen
    median_speed = np.median(speeds)
    speed_text = f"Median speed(m/s): {median_speed:.5f}"
    pos_text = f"Position(m): ({agent.pos[0]:.2f}, {agent.pos[1]:.2f})"
    robot_speed_text = f"Robot speed(m/s): {np.linalg.norm(robot.v):.5f}"
    robot_v_text = f"Robot linear vel(m/s): ({robot.v[0]:.3f}, {robot.v[1]:.3f})"
    robot_w_text = f"Robot angular vel(rad/s): {robot.w:.3f}"
    robot_pos_text = f"Robot pos(m): ({robot.pos[0]:.2f}, {robot.pos[1]:.2f})"
    robot_theta_text = f"Robot angle(rad): {robot.theta:.2f}"

    text_surface_speed = font.render(speed_text, True, (0, 0, 0))
    text_surface_pos = font.render(pos_text, True, (0, 0, 0))
    text_surface_robot_speed = font.render(robot_speed_text, True, (0, 0, 0))
    text_surface_robot_pos = font.render(robot_pos_text, True, (0, 0, 0))
    text_surface_robot_v = font.render(robot_v_text, True, (0, 0, 0))
    text_surface_robot_w = font.render(robot_w_text, True, (0, 0, 0))
    text_surface_robot_theta = font.render(robot_theta_text, True, (0, 0, 0))

    text_rect_speed = text_surface_speed.get_rect(
        bottomright=(meters_to_pixels(width) - 30, meters_to_pixels(height) - 20)
    )
    text_rect_pos = text_surface_pos.get_rect(
        bottomright=(meters_to_pixels(width) - 30, meters_to_pixels(height) - 50)
    )
    text_rect_robot_speed = text_surface_robot_speed.get_rect(
        bottomleft=(30, meters_to_pixels(height) - 20)
    )
    text_rect_robot_pos = text_surface_robot_pos.get_rect(
        bottomleft=(30, meters_to_pixels(height) - 50)
    )
    text_rect_robot_v = text_surface_robot_v.get_rect(
        bottomleft=(30, meters_to_pixels(height) - 80)
    )
    text_rect_robot_w = text_surface_robot_w.get_rect(
        bottomleft=(30, meters_to_pixels(height) - 110)
    )
    text_rect_robot_theta = text_surface_robot_theta.get_rect(
        bottomleft=(30, meters_to_pixels(height) - 140)
    )

    screen.blits(
        (
            (text_surface_speed, text_rect_speed),
            (text_surface_pos, text_rect_pos),
            (text_surface_robot_theta, text_rect_robot_theta),
            (text_surface_robot_pos, text_rect_robot_pos),
            (text_surface_robot_v, text_rect_robot_v),
            (text_surface_robot_w, text_rect_robot_w),
            (text_surface_robot_speed, text_rect_robot_speed),
        )
    )


# Drawing a circle radius around the agent when standing still
def draw_transparent_circle(surface, center, radius):
    target_rect = pygame.Rect(center, (0, 0)).inflate((radius * 2, radius * 2))
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.circle(shape_surf, (255, 0, 0, 75), (radius, radius), radius)
    surface.blit(shape_surf, target_rect)
