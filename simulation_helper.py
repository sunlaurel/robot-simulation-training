import torch
import numpy as np

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
