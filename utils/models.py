import torch.nn as nn
import torch.nn.functional as F


class MultiLayer(nn.Module):

    def __init__(
        self, input_size, hidden_layer1, hidden_layer2, output_size, scale_factor=1
    ):
        super().__init__()  # Inherited from the parent class nn.Module
        self.scale_factor = scale_factor
        # for debugging
        self.input_size = input_size
        self.output_size = output_size
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
        # breakpoint()
        batch_size = x.size(0)  # Store the batch size before flattening
        x = x.view(batch_size, -1)  # Flatten the input x
        x = F.relu(self.linear1(x))  # Linear transform, then relu
        x = F.relu(self.linear2(x))
        out = self.linear3(x)
        out = out.view(
            batch_size, features, -1
        )  # Reshape the output to (batch_size, 2, future_time_steps)
        out /= self.scale_factor
        return out


class MultiLayer2(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_layer1,
        hidden_layer2,
        hidden_layer3,
        hidden_layer4,
        output_size,
    ):
        super().__init__()  # Inherited from the parent class nn.Module
        self.linear1 = nn.Linear(input_size, hidden_layer1)
        self.linear2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.linear3 = nn.Linear(hidden_layer2, hidden_layer3)
        self.linear4 = nn.Linear(hidden_layer3, hidden_layer4)
        self.linear5 = nn.Linear(hidden_layer4, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        out = self.linear5(x)
        out = out.view(batch_size, 2, -1)
        return out
