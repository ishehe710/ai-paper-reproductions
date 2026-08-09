
# updates the learning rate according to the paper
def update_learning_rate(optimizer, decay_factor=1.000004, min_lr=1e-6):
    for param_group in optimizer.param_groups:
        param_group["lr"] = max(
            param_group["lr"] / decay_factor,
            min_lr
        )

# updates the momentum according to the paper     
def update_momentum(
    optimizer,
    epoch,
    initial_momentum=0.5,
    final_momentum=0.7,
    start=1,
    saturate=250
):
    
    if epoch <= start:
        momentum = initial_momentum

    elif epoch >= saturate:
        momentum = final_momentum

    else:
        progress = (epoch - start) / (saturate - start)
        momentum = (
            initial_momentum
            + progress * (final_momentum - initial_momentum)
        )

    for param_group in optimizer.param_groups:
        param_group["momentum"] = momentum