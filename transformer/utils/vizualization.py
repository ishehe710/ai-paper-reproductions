import matplotlib.pyplot as plt
# make plot for loss


# training and validation per epoch
def create_loss_plot(train_loss, val_loss):
    
    plt.plot(train_loss, label="training")
    plt.plot(val_loss, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Loss vs. Epoch Plot")
    plt.legend()
