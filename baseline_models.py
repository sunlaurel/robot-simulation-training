import torch
from training import future_steps

save_path = "./best-weights/best_weight_noise.pth"


def stand_still_model(X_past):
    return X_past[:, -1].unsqueeze(1).repeat(1, 10)


def maintain_velocity_model(X_past, N_future=future_steps):
    # breakpoint()
    X_future = torch.zeros(2, N_future).float()
    V = (X_past[:, -1] - X_past[:, -2]) / 0.12
    X_cur = torch.tensor(X_past[:, -1])
    for step in range(N_future):
        X_cur = X_cur + V * 0.12
        X_future[:, step] = X_cur
    return X_future


baseline_model = lambda x: maintain_velocity_model(x)

stand_model = lambda x: stand_still_model(x)
