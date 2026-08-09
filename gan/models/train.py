import torch 
from torch.optim import SGD
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from gan.models.generator import Generator
from gan.models.discriminator import Discriminator
from gan.models.preprocessing import batch
from gan.models.evaluate import evaluate
from gan.models.optimize import update_learning_rate, update_momentum

def fit(G_model, D_model, epochs, train_loader, latent_d, loss_discriminator,  loss_generator, val_loader, optimizer_D, optimizer_G, device, optimizer_choice="SGD"):
    
    G_model.train()
    D_model.train()
    
    
    # displaying examples
    fixed_z = torch.rand([32, latent_d], device=device)
    
    history = {'D': {'train': [], 'val': []}, 'G': {'train': [], 'val': []}}
    
    for epoch in range(epochs):
        
        train_loss_D = 0.0  
        train_loss_G = 0.0
    
        for images, _ in train_loader:
                        
            # train discriminator
            z = torch.rand([images.size(0), latent_d])
            
            fake_images = G_model(z)
            
            real_pred = D_model(images)
            fake_pred = D_model(fake_images.detach())
            
            loss_D = loss_discriminator(fake_pred, real_pred)
            
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            
            # train generator 1
            z = torch.randn([images.size(0), latent_d], device=device)
                        
            fake_images = G_model(z)
            
            fake_pred = D_model(fake_images)
            
            loss_G = loss_generator(fake_pred)
            
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            
            train_loss_D += loss_D.item()
            train_loss_G += loss_G.item()
            
            # train generator twice
            z = torch.randn([images.size(0), latent_d], device=device)
                        
            fake_images = G_model(z)
            
            fake_pred = D_model(fake_images)
            
            loss_G = loss_generator(fake_pred)
            
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            
            train_loss_D += loss_D.item()
            train_loss_G += loss_G.item()
            
        
        avg_train_loss_D = train_loss_D / len(train_loader)
        avg_train_loss_G = train_loss_G / len(train_loader)
        D_val_loss, G_val_loss = evaluate(
            D_model=D_model, 
            G_model=G_model,
            data_loader=val_loader, 
            loss_D_fn=loss_discriminator, 
            loss_G_fn=loss_generator, 
            latent_d=latent_d, 
            device=device
            )
        
        G_model.train()
        D_model.train()
        
        '''
            start    = 1
saturate = 250
initial  = 0.5
final    = 0.7
        
        '''
        
        if optimizer_choice == "SGD":
            update_learning_rate(optimizer_D)
            update_learning_rate(optimizer_G)
            update_momentum(
                optimizer_D,
                epoch + 1
            )
            update_momentum(
                optimizer_G,
                epoch + 1
            )
        
        
        
            current_lr_D = optimizer_D.param_groups[0]["lr"]
            current_lr_G = optimizer_G.param_groups[0]["lr"]

            current_momentum_D = optimizer_D.param_groups[0]["momentum"]
            current_momentum_G = optimizer_G.param_groups[0]["momentum"]
        
        # print epoch metric info
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"\tDiscriminator Train   Loss: {avg_train_loss_D:.4f}")
        print(f"\tGenerator Train       Loss: {avg_train_loss_G:.4f}")
        print(f"\tDiscriminator Val     Loss: {D_val_loss:.4f}")
        print(f"\tGenerator Val         Loss: {G_val_loss:.4f}")
        
        if optimizer_choice == "SGD":
            print(f"\tD Learning Rate:   {current_lr_D:.8f}")
            print(f"\tG Learning Rate:   {current_lr_G:.8f}")
            print(f"\tD Momentum:        {current_momentum_D:.4f}")
            print(f"\tG Momentum:        {current_momentum_G:.4f}")
        
        if epoch + 1 in [1, 5, 10, 15, 20, 22]:
            G_model.eval()

            with torch.no_grad():
                samples = G_model(fixed_z)
                samples = torch.reshape(samples, [32, 1, 28, 28])
                
                
                # 2. Compile the batch into a single grid image
                # 'nrow' sets how many images appear in each row
                img_grid = make_grid(samples, nrow=8, padding=2)

                # 3. Convert PyTorch tensor format to Matplotlib format
                # PyTorch uses:     [Channels, Height, Width]
                # Matplotlib uses:  [Height, Width, Channels]
                img_grid_np = img_grid.permute(1, 2, 0).numpy()

                # 4. Display the final grid image
                plt.figure(figsize=(8, 4))
                plt.imshow(img_grid_np)
                plt.axis("off")  # Hide the pixel rulers
                plt.tight_layout()
                plt.show()

                    

            G_model.train()
            
        history['D']['train'].append(avg_train_loss_D)
        history['G']['train'].append(avg_train_loss_G)
        history['D']['val'].append(D_val_loss)
        history['G']['val'].append(G_val_loss)



    return history
        
        
        
        
        


    
