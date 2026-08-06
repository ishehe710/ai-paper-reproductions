'''
    Runs the entire training process. the model and dataset to train them
'''
from transformer.model.evaluate import evaluate
import torch
from transformer.model.config import D_MODEL, WARMUP_STEPS

def fit(model, num_epochs, train_loader, val_loader, loss_func, optimizer, device, d_model=D_MODEL):
    
    history = {'training': [], 'validation': [], 'lr': []}
    
    global_step = 0  # Track total batches processed across all epochs
    lr_rate = 0
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
            
            global_step += 1
            lr_rate = (d_model ** -0.5) * min(global_step ** -0.5, global_step * (WARMUP_STEPS ** -1.5))
            history['lr'].append(lr_rate)


            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_rate  # Overwrites the value right before the update
            
            
                        
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
        
        
        
        # print epoch metric info
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"\tTrain   Loss: {train_loss:.4f}")
        print(f"\tVal     Loss: {val_loss:.4f}")
        print(f"\tLearing Rate: {lr_rate}")
        
        # log the metric data
        history['training'].append(train_loss)
        history['validation'].append(val_loss)
    
    return history
