'''
    Runs the entire training process. the model and dataset to train them
'''
from transformer.model.evaluate import evaluate

def train(model, num_epochs, train_loader, val_loader, loss_func, optimizer, scheduler, device):
    
    history = {'training': [], 'validation': []}
    for epoch in range(num_epochs):

        model.train()

        # training loop
        running_loss = 0.0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # forward
            outputs = model(inputs, targets)
            loss = loss_func(outputs, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # training metrics
            running_loss += loss.item() * inputs.size(0)
            total += targets.size(0)
        
        # Validation and tracking metrics phase
        train_loss = running_loss / total
        val_loss = evaluate(model, val_loader, loss_func, device)
        
        if scheduler is not None:
            scheduler.step()
        
        # print epoch metric info
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"\tTrain   Loss: {train_loss:.4f}")
        print(f"\tVal     Loss: {val_loss:.4f}")
        
        # log the metric data
        history['training'].append(train_loss)
        history['validation'].append(val_loss)
    
    return history
