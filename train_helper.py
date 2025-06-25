import matplotlib.pyplot as plt
import torch

# helper functions
def train(network, data_generator, loss_function, optimizer):
    network.train()  # updates any network layers that behave differently in training and execution
    avg_loss = 0
    num_batches = 0

    for i, (input_pos, target_pos, input_velocity, target_velocity) in enumerate(data_generator):
        optimizer.zero_grad()  # Gradients need to be reset each batch
        prediction = network(
            torch.cat((input_pos.float(), input_velocity.float()), dim=1)
        )  # Forward pass: compute the next positions given previous positions
        
        loss = loss_function(
            prediction, torch.cat((target_pos.float(), target_velocity.float()), dim=1)
        )  # Compute the loss: difference between the output and correct result
        loss.backward()  # Backward pass: compute the gradients of the model with respect to the loss
        optimizer.step()
        avg_loss += loss.item()
        num_batches += 1
    return avg_loss / num_batches


def test(network, test_loader, loss_function):
    network.eval()  # updates any network layers that behave differently in training and execution
    test_loss = 0
    num_batches = 0
    with torch.no_grad():
        for input_pos, target_pos, input_velocity, target_velocity in test_loader:
            output = network(torch.cat((input_pos.float(), input_velocity.float()), dim=1))
            test_loss += loss_function(output, torch.cat((target_pos.float(), target_velocity.float()), dim=1)).item()
            num_batches += 1
    return test_loss / num_batches


def logResults(
    epoch,
    num_epochs,
    train_loss,
    train_loss_history,
    test_loss,
    test_loss_history,
    epoch_counter,
    print_interval=1000,
):
    if epoch % print_interval == 0:
        print(
            "Epoch [%d/%d], Train Loss: %.4f, Test Loss: %.4f"
            % (epoch + 1, num_epochs, train_loss, test_loss)
        )
    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    epoch_counter.append(epoch)


def graphLoss(
    epoch_counter, train_loss_hist, test_loss_hist, loss_name="Loss", start=0
):
    fig = plt.figure()
    plt.plot(epoch_counter[start:], train_loss_hist[start:], color="blue")
    plt.plot(epoch_counter[start:], test_loss_hist[start:], color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
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
    logging_interval=1,
):
    best_model_weights = None
    best_val_loss = float("inf")
    save_path = None

    # Arrays to store training history
    test_loss_history = []
    epoch_counter = []
    train_loss_history = []

    for epoch in range(num_epochs):
        avg_loss = train(network, training_generator, loss_function, optimizer)

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_model_weights = network.state_dict()  # Save weights in memory
            save_path = f"best_weights_epoch_{epoch+1}.pth"
            torch.save(best_model_weights, save_path)

        if best_model_weights is not None:
            # network.load_state_dict(best_model_weights)
            network.load_state_dict(torch.load(save_path, weights_only=True))

        test_loss = test(network, testing_generator, loss_function)
        logResults(
            epoch,
            num_epochs,
            avg_loss,
            train_loss_history,
            test_loss,
            test_loss_history,
            epoch_counter,
            logging_interval,
        )

    graphLoss(epoch_counter, train_loss_history, test_loss_history)
