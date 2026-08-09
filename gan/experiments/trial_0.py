import torch
from torch.optim import SGD

# in project imports
from gan.models.preprocessing import train_loader, val_loader, test_loader
from gan.models.config import NUM_PIECES, LEARN_RATE, NUM_EPOCHS, LATENT_D
from gan.models.discriminator import Discriminator
from gan.models.generator import Generator
from gan.models.train import fit
from gan.models.evaluate import evaluate

# setup trainning



D_model = Discriminator(num_pieces=NUM_PIECES)
G_model = Generator(0)

D_optimizer = SGD(D_model.parameters(), lr=LEARN_RATE)
G_optimizer = SGD(G_model.parameters(), lr=LEARN_RATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_model = G_model.to(device)
D_model = D_model.to(device)

# training
history = fit(
    G_model=G_model,
    D_model=D_model,
    epochs=NUM_EPOCHS,
    train_loader=train_loader,
    val_loader=val_loader,
    latent_d=LATENT_D,
    optimizer_D=D_optimizer,
    optimizer_G=G_optimizer,
    loss_generator=G_model.loss,
    loss_discriminator=D_model.loss,
    device=device
)

# test
test_result = evaluate(
    D_model=D_model,
    G_model=G_model,
    data_loader=test_loader,
    loss_D_fn=D_model.loss,
    loss_G_fn=G_model.loss,
    latent_d=LATENT_D,
    device=device
)

print("Test result:", test_result)