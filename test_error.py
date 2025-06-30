from utils import *
from training import testing_data, loss_function
from baseline_models import stand_model, baseline_model
from train_helper import T_test
import torch


save_path = "./best-weights/best_weight_offset.pth"
network = MultiLayer(2*10, 100, 100, 2*10)
network.load_state_dict(torch.load(save_path, weights_only=True))

# test_generator = torch.utils.data.DataLoader(testing_data, batch_size=100, shuffle=False)
test_generator = torch.utils.data.DataLoader(testing_data, batch_size=100, shuffle=False)

_, (input, expected) = next(enumerate(test_generator))
breakpoint()
input, expected = T_test(input, offset=True, scale=False, X_future=expected)

with torch.no_grad():
    predicted = network(input.float())
    stand_predicted = stand_model(input.float())
    baseline_predicted = baseline_model(input.float())

    predicted_loss = loss_function(expected, predicted.float())
    stand_loss = loss_function(expected, stand_predicted.float())
    baseline_loss = loss_function(expected, baseline_predicted.float())

print("Predicted loss:", predicted_loss)
print("Standing Model loss:", stand_loss)
print("Baseline Model loss:", baseline_loss)
