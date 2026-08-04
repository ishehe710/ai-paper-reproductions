'''
    Runs the entire training process. the model and dataset to train them
'''
from transformer.model.evaluate import evaluate
import torch

def fit(model, num_epochs, train_loader, val_loader, loss_func, optimizer, scheduler, device):
    
    history = {'training': [], 'validation': []}
    for epoch in range(num_epochs):

        #print("cool 1:", epoch)

        model.train()

        # training loop
        running_loss = 0.0
        total = 0
        for inputs, targets in train_loader:
            
            #print("cool 2")
            inputs, targets = inputs.to(device), targets.to(device)

            # forward
            #print("cool 3")
            outputs = model(inputs, targets)
            outputs_t = torch.transpose(outputs, dim0=1, dim1=2)
            loss = loss_func(outputs_t, targets)
            
            
            #print(loss)
            #print(torch.isfinite(loss))
        

            # backward
            #print("cool 4")
            optimizer.zero_grad()
            #print("after zero_grad")
            
            
            
            loss.backward()
            #print("after backward")
            
            optimizer.step()
            #print("after step")
            
            # training metrics
            #print("cool 5")
            running_loss += loss.item() * inputs.size(0)
            total += targets.size(0)
        
        #print("before eval")
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
