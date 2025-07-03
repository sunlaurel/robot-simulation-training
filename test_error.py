import utils
from training import loss_function
from train_helper import T_test
import torch
import re

save_path = "./best-weights/best_weight_noise_rotate_offset(0.1-sigma).pth"
match = re.search(r"\((\d+)-past\)", save_path)

# extracting number of past steps from the file name
if match:
    past_steps = int(match.group(1))
else:
    past_steps = 10

future_steps = 10
network = utils.models.MultiLayer(
    input_size=2 * past_steps,
    hidden_layer1=100,
    hidden_layer2=100,
    output_size=2 * future_steps,
)
network.load_state_dict(torch.load(save_path, weights_only=True))

_, testing_data = utils.data.GenTrainTestDatasets(
    csv_path="./training-data/crowd_data.csv",
    past_steps=past_steps,
    future_steps=future_steps,
)

test_generator = torch.utils.data.DataLoader(testing_data, batch_size=100, shuffle=False)
predicted_loss = 0
stand_loss = 0
baseline_loss = 0


def metric_test_loss(X_past, X_future, model, scale):
    metric_predicted = model(X_past.float() * scale) / scale
    loss = loss_function(X_future, metric_predicted.float())
    return loss


with torch.no_grad():
    # breakpoint()
    for input, expected in test_generator:
        input, expected = T_test(input, offset=True, scale=False, X_future=expected)
        stand_predicted = utils.models.stand_model(input.float())
        baseline_predicted = utils.models.constant_velocity_model(input.float())

        predicted_loss += metric_test_loss(input, expected, network, 1)
        stand_loss += loss_function(expected, stand_predicted.float())
        baseline_loss += loss_function(expected, baseline_predicted.float())

print(f"Predicted loss: {(predicted_loss / 100.0).item():.4f}")
print(f"Standing Model loss: {(stand_loss / 100.0).item():.4f}")
print(f"Baseline Model loss: {(baseline_loss / 100.0).item():.4f}")
