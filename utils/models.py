import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent
from torch.distributions import Normal
import torch
import json

########################################################
##   Overview                                         ##
##  - contains the models used for prediction         ##
##  - MultiLayerRobot is the model used specifically  ##
##    for the robot                                   ##
##  - also contains some baseline models (standing    ##
##    still and maintaining velocity)                 ##
########################################################

# Policy network - Predicts actions distribution from the current state
policy_size = 32
value_size = 128


class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, policy_size)
        self.fc2 = nn.Linear(policy_size, policy_size)
        self.fc3 = nn.Linear(policy_size, action_size)
        self.std_dev_v = nn.Parameter(
            torch.ones(action_size)
        )  # Trainable standard deviation (will be softplus transformed to ensure positivity)

    # Returns action distribution
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        x = torch.tanh(x)
        action_mean = x
        action_distrib = Independent(Normal(action_mean, F.softplus(self.std_dev_v)), 1)
        return action_distrib


# Value network - Predicts state value from the current state
class Value(nn.Module):
    def __init__(self, state_size):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(state_size, value_size)
        self.fc2 = nn.Linear(value_size, value_size)
        self.fc3 = nn.Linear(value_size, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


""" Multi Layer Perceptron (MLP) for predicting N steps into the future """
class MultiLayer(nn.Module):

    def __init__(
        self, input_size, hidden_layer1, hidden_layer2, output_size, scale_factor=1
    ):
        super().__init__()  # Inherited from the parent class nn.Module
        self.scale_factor = scale_factor
        self.linear1 = nn.Linear(
            input_size, hidden_layer1
        )  # Linear layer: input_size -> hidden_layer
        self.linear2 = nn.Linear(
            hidden_layer1, hidden_layer2
        )  # Linear layer: hidden_layer -> num_classes
        self.linear3 = nn.Linear(
            hidden_layer2, output_size
        )  # Linear layer: hidden_layer -> num_classes

    def forward(
        self, x, features=2
    ):  # Forward pass which defines how the layers relate the input x to the output
        x *= self.scale_factor
        batch_size = x.size(0)  # Store the batch size before flattening
        x = x.view(batch_size, -1)  # Flatten the input x
        x = F.relu(self.linear1(x))  # Linear transform, then relu
        out = self.linear3(x)
        out = out.view(
            batch_size, features, -1
        )  # Reshape the output to (batch_size, 2, future_time_steps)
        out /= self.scale_factor
        return out


""" Multi Layer Perceptron (MLP) for robot predicting future position to reach """
class MultiLayerRobot(nn.Module):

    def __init__(
        self, input_size, hidden_layer1, hidden_layer2, output_size, scale_factor=1
    ):
        super().__init__()  # Inherited from the parent class nn.Module
        self.scale_factor = scale_factor
        self.linear1 = nn.Linear(
            input_size, hidden_layer1
        )  # Linear layer: input_size -> hidden_layer
        self.linear2 = nn.Linear(
            hidden_layer1, hidden_layer2
        )  # Linear layer: hidden_layer -> num_classes
        self.linear3 = nn.Linear(
            hidden_layer2, output_size
        )  # Linear layer: hidden_layer -> num_classes

    def forward(
        self, x, features=2
    ):  # Forward pass which defines how the layers relate the input x to the output
        x *= self.scale_factor
        batch_size = x.size(0)  # Store the batch size before flattening
        x = x.view(batch_size, -1)  # Flatten the input x
        x = F.relu(self.linear1(x))  # Linear transform, then relu
        x = F.relu(self.linear2(x))
        out = self.linear3(x)
        out = out.view(
            batch_size, features
        )  # Reshape the output to (batch_size, 2)
        out /= self.scale_factor
        return out


""" Baseline models """
with open(file="./utils/config.json", mode="r", encoding="utf-8") as file:
    data = json.load(file)

future_steps = data["future-steps"]


def stand_still_model(X_past):
    # from training import future_steps
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
    # from training import future_steps
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


constant_velocity_model = lambda x: maintain_velocity_model(x)

stand_model = lambda x: stand_still_model(x)
