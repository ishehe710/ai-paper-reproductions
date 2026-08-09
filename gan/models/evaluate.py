import torch

def evaluate(D_model, G_model, data_loader, loss_D_fn, loss_G_fn, latent_d, device):
    
    D_model.eval()
    G_model.eval()
    
    D_running_loss = 0.0
    G_running_loss = 0.0
    total = 0

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)


            # perform inference
            z = torch.rand([images.size(0), latent_d])
            
            fake_images = G_model(z)
            
            real_pred = D_model(images)
            fake_pred = D_model(fake_images)
            
            loss_D = loss_D_fn(fake_pred, real_pred)
            loss_G = loss_G_fn(fake_pred)
        
            
            D_running_loss += loss_D.item() * images.size(0)
            G_running_loss += loss_G.item() * images.size(0)
            total += images.size(0)

        
            
    epoch_loss = [D_running_loss / total, G_running_loss / total]

    return epoch_loss