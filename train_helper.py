import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from utils import models

###############################################################
##   Overview                                                ##
##   - helper functions for training the model to predict    ##
##     future trajectories of the agent                      ##
###############################################################


def T_train(
    X_past, X_future, angle=0, offset=True, scale=True, rotate=True, add_noise=True
):
    """Augmenting the data for training"""
    if offset:
        X_start = (X_past[:, 0, -1, None], X_past[:, 1, -1, None])
        X_past = torch.stack(
            (
                torch.tensor(X_past[:, 0] - X_start[0]),
                torch.tensor(X_past[:, 1] - X_start[1]),
            ),
            dim=1,
        )

        X_future = torch.stack(
            (
                torch.tensor(X_future[:, 0] - X_start[0]),
                torch.tensor(X_future[:, 1] - X_start[1]),
            ),
            dim=1,
        )

    if add_noise:
        sigma = 0.1
        N = np.random.rand(*X_past.shape) * sigma
        X_past = torch.tensor(X_past) + torch.tensor(N * 0.1, dtype=torch.float)

    if scale:
        X_past = X_past / 2
        X_future = X_future / 2

    if rotate:
        angle = random.uniform(-np.pi, np.pi)
        rotate = lambda x: torch.tensor(
            [[torch.cos(x), -torch.sin(x)], [torch.sin(x), torch.cos(x)]]
        )
        X_past = rotate(torch.tensor(angle)) @ X_past.float()
        X_future = rotate(torch.tensor(angle)) @ X_future.float()

    return X_past, X_future


def T_test(X_past, X_future, offset=True, scale=True):
    """Applying data augmentation to the testing data"""
    if offset:
        X_start = (X_past[:, 0, -1, None], X_past[:, 1, -1, None])
        X_past = torch.stack(
            (
                torch.tensor(X_past[:, 0] - X_start[0]),
                torch.tensor(X_past[:, 1] - X_start[1]),
            ),
            dim=1,
        )

        X_future = torch.stack(
            (
                torch.tensor(X_future[:, 0] - X_start[0]),
                torch.tensor(X_future[:, 1] - X_start[1]),
            ),
            dim=1,
    )

    if scale:
        X_past = X_past / 2
        X_future = X_future / 2

    return X_past, X_future


def train(
    network,
    data_generator,
    loss_function,
    optimizer,
    offset=False,
    scale=False,
    add_noise=False,
    rotate=False,
):
    network.train()  # updates any network layers that behave differently in training and execution
    avg_loss = 0
    num_batches = 0

    for i, (input_pos, target_pos) in enumerate(data_generator):
        # breakpoint()
        input_pos, target_pos = T_train(
            input_pos,
            target_pos,
            offset=offset,
            scale=scale,
            add_noise=add_noise,
            rotate=rotate,
        )

        optimizer.zero_grad()  # Gradients need to be reset each batch
        prediction = network(input_pos.float())
        # prediction = network(
        #     torch.cat((input_pos.float(), input_velocity.float()), dim=1)
        # )  # Forward pass: compute the next positions given previous positions
        # breakpoint()
        loss = loss_function(prediction, target_pos.float())
        # loss = loss_function(
        #     prediction, torch.cat((target_pos.float(), target_velocity.float()), dim=1)
        # )  # Compute the loss: difference between the output and correct result
        loss.backward()  # Backward pass: compute the gradients of the model with respect to the loss
        optimizer.step()
        avg_loss += loss.item()
        num_batches += 1
    return avg_loss / num_batches


def test(
    network,
    test_loader,
    loss_function,
    offset=False,
    scale=False,
):
    network.eval()  # updates any network layers that behave differently in training and execution
    test_loss = 0
    num_batches = 0

    # recording the loss for the baseline models
    stand_test_loss = 0
    baseline_test_loss = 0

    with torch.no_grad():
        for input_pos, target_pos in test_loader:
            input_pos, target_pos = T_test(
                input_pos, target_pos, offset=offset, scale=scale
            )
            output = network(input_pos.float())
            stand_output = models.stand_model(input_pos.float())
            baseline_output = models.constant_velocity_model(input_pos.float())
            # output = network(
            #     torch.cat((input_pos.float(), input_velocity.float()), dim=1)
            # )
            test_loss += loss_function(output, target_pos.float())
            stand_test_loss += loss_function(stand_output, target_pos.float())
            baseline_test_loss += loss_function(baseline_output, target_pos.float())
            # test_loss += loss_function(
            #     output, torch.cat((target_pos.float(), target_velocity.float()), dim=1)
            # ).item()
            num_batches += 1
    # return test_loss / num_batches
    return (
        test_loss / num_batches,
        stand_test_loss / num_batches,
        baseline_test_loss / num_batches,
    )


def logResults(
    epoch,
    num_epochs,
    train_loss,
    train_loss_history,
    test_loss,
    test_loss_history,
    stand_loss,
    stand_test_loss_history,
    baseline_loss,
    baseline_test_loss_history,
    epoch_counter,
    print_interval=1000,
):
    if epoch % print_interval == 0:
        print(
            "Epoch [%d/%d], Train Loss: %.4f, Test Loss: %.4f, Stand Loss: %.4f, Baseline Loss: %.4f"
            % (epoch + 1, num_epochs, train_loss, test_loss, stand_loss, baseline_loss)
        )
    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    stand_test_loss_history.append(stand_loss)
    baseline_test_loss_history.append(baseline_loss)
    epoch_counter.append(epoch)


def graphLoss(
    epoch_counter,
    train_loss_hist,
    test_loss_hist,
    stand_test_loss_hist,
    baseline_test_loss_hist,
    loss_name="Loss",
    start=0,
    graph_stand=False,
):
    fig = plt.figure()
    plt.plot(epoch_counter[start:], train_loss_hist[start:], color="blue")
    plt.plot(epoch_counter[start:], test_loss_hist[start:], color="red")
    plt.plot(epoch_counter[start:], baseline_test_loss_hist[start:], color="purple")
    if graph_stand:
        plt.plot(epoch_counter[start:], stand_test_loss_hist[start:], color="green")
        plt.legend(
            ["Train Loss", "Test Loss", "Baseline Loss", "Standing Loss"],
            loc="upper right",
        )
    else:
        plt.legend(["Train Loss", "Test Loss", "Baseline Loss"], loc="upper right")
    plt.xlabel("#Epochs")
    plt.ylabel(loss_name)
    plt.show()


def trainAndGraph(
    network,
    training_generator,
    testing_generator,
    loss_function,
    optimizer,
    num_epochs,
    future_steps,
    past_steps,
    logging_interval=1,
    offset=False,
    scale=False,
    add_noise=False,
    rotate=False,
):
    best_epoch = 0
    best_model_weights = None
    best_val_loss = float("inf")

    # Arrays to store training history
    test_loss_history = []
    epoch_counter = []
    train_loss_history = []

    # Arrays to store testing history of the baseline models
    stand_test_loss_history = []
    baseline_test_loss_history = []

    for epoch in range(num_epochs):
        avg_loss = train(
            network,
            training_generator,
            loss_function,
            optimizer,
            offset=offset,
            scale=scale,
            add_noise=add_noise,
            rotate=rotate,
        )

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_epoch = epoch
            best_model_weights = network.state_dict()  # Save weights in memory
            save_path = f"./weights/best-weights/best_weight{'_noise' if add_noise else ''}{'_rotate' if rotate else ''}{'_scale' if scale else ''}{'_offset' if offset else ''}{'(' + str(past_steps) + '-past)' if past_steps != 10 else ''}{'(0.1-sigma)' if add_noise else ''}.pth"

            torch.save(best_model_weights, save_path)  # Load weights on disk

        # test_loss = test(
        #     network, testing_generator, loss_function, offset=offset, scale=scale
        # )

        test_loss, stand_loss, baseline_loss = test(
            network, testing_generator, loss_function, offset=offset, scale=scale
        )

        # load it in and compare it to the other model
        logResults(
            epoch,
            num_epochs,
            avg_loss,
            train_loss_history,
            test_loss,
            test_loss_history,
            stand_loss,
            stand_test_loss_history,
            baseline_loss,
            baseline_test_loss_history,
            epoch_counter,
            logging_interval,
        )

    print("Best epoch:", best_epoch)
    graphLoss(
        epoch_counter,
        train_loss_history,
        test_loss_history,
        stand_test_loss_history,
        baseline_test_loss_history,
    )
