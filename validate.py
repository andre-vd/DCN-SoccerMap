# evaluate_model.py
import torch

def evaluate_model(model, data_loader, criterion):
    """
    Evaluates a soccer event prediction model.
    
    Parameters:
        model: The model to be evaluated.
        data_loader: DataLoader for the dataset to evaluate.
        criterion: Loss function to use.
        
    Returns:
        avg_loss: The average loss over the dataset.
        accuracy: The accuracy of the model predictions.
    """
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels, outcomes in data_loader:
            inputs = inputs.to(torch.float32)
            outcomes = outcomes.to(torch.float32)
            
            output_surface = model(inputs)
            
            pred_values = []
            for i in range(len(labels)):
                x, y = int(labels[i][0]), int(labels[i][1])
                x = min(max(x, 0), output_surface.size(2) - 1)
                y = min(max(y, 0), output_surface.size(3) - 1)
                pred_values.append(output_surface[i, 0, x, y])
            
            pred_values = torch.stack(pred_values)
        
            loss = criterion(pred_values, outcomes)
            total_loss += loss.item() * inputs.size(0)
            
            predictions = (pred_values > 0.5).float()  # Threshold at 0.5
            correct += (predictions == outcomes).sum().item()
            total_samples += outcomes.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy