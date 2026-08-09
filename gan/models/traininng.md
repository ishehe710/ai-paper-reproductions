for real_images in train_loader:

    # -------------------------
    # Train Discriminator
    # -------------------------

    z = sample_noise(batch_size)

    fake_images = generator(z)

    real_pred = discriminator(real_images)
    fake_pred = discriminator(fake_images.detach())

    loss_D = ...

    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()


    # -------------------------
    # Train Generator
    # -------------------------

    z = sample_noise(batch_size)

    fake_images = generator(z)
    fake_pred = discriminator(fake_images)

    loss_G = ...

    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()