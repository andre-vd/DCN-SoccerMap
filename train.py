import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, train_events, train_labels, train_outcomes, num_epochs=10, batch_size=32, lr=1e-4, name='Default'):
    """
    Trains a soccer event prediction model.
    
    Parameters:
        model: The model to be trained.
        train_events: Training data containing event inputs.
        train_labels: Ground truth labels with event positions.
        train_outcomes: Outcomes for each event (e.g., success/failure).
        num_epochs: Number of epochs to train the model.
        batch_size: Batch size for training.
        lr: Learning rate for the optimizer.

        Todo: Look into proper target logloss implementation
    """
    model.train()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = TensorDataset(train_events, train_labels, train_outcomes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels, outcomes in train_loader:
            inputs = inputs.to(torch.float32)
            outcomes = outcomes.to(torch.float32)

            optimizer.zero_grad()

            output_surface = model(inputs)

            pred_values = []
            for j in range(len(labels)):
                x, y = int(labels[j][0]), int(labels[j][1])
                x = min(max(x, 0), output_surface.size(2) - 1)
                y = min(max(y, 0), output_surface.size(3) - 1)
                pred_values.append(output_surface[j, 0, x, y])

            pred_values = torch.stack(pred_values)

            loss = criterion(pred_values, outcomes)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    print("Training Finished")
    total_params = sum(p.numel() for p in model.parameters())
    print("Parameters: ", total_params)
    torch.save(model.state_dict(), name)
    print("Model Saved as: ", name)