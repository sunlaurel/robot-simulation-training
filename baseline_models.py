import torch

# TODO: need to remove the double data upload from the delayed import
# TODO: fix the package management

def stand_still_model(X_past):
    from training import future_steps
    # breakpoint()
    if X_past.dim() == 3:
        return X_past[:, :, -1].unsqueeze(2).repeat(1, 1, future_steps)
    elif X_past.dim() == 2:
        return X_past[:, -1].unsqueeze(1).repeat(1, future_steps)
    else:
        raise RuntimeError(
            "Couldn't match the dimensions properly: go back to fix this in stand_still_model function"
        )


def maintain_velocity_model(X_past):
    from training import future_steps
    # breakpoint()
    N_future = future_steps
    if (
        X_past.dim() == 3
    ):  # handling the case when the model is taking in a batch of training/testing set
        X_future = torch.zeros(X_past.size(0), 2, N_future).float()
        V = (X_past[:, :, -1] - X_past[:, :, -2]) / 0.12
        X_cur = torch.tensor(X_past[:, :, -1])
        for step in range(N_future):
            X_cur = X_cur + V * 0.12
            X_future[:, :, step] = X_cur
        return X_future
    elif (
        X_past.dim() == 2
    ):  # handling the case when the model is simply predicting from a sample
        X_future = torch.zeros(2, N_future).float()
        V = (X_past[:, -1] - X_past[:, -2]) / 0.12
        X_cur = torch.tensor(X_past[:, -1])
        for step in range(N_future):
            X_cur = X_cur + V * 0.12
            X_future[:, step] = X_cur
        return X_future
    else:
        raise RuntimeError(
            "Couldn't handle the dimensions properly: please go back to fix in the implementation maintain_velocity_model function"
        )


baseline_model = lambda x: maintain_velocity_model(x)

stand_model = lambda x: stand_still_model(x)
