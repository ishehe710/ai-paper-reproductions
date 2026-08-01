import torch

def evaluate(model, dataloader, loss_func, device):
    model.eval()

    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs, targets)
            loss = loss_func(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            total += targets.size(0)


           
    epoch_loss = running_loss / total

    return epoch_loss