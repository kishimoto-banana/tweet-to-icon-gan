import time

import torch
from torch import nn


def train(generator, discriminator, dataloader, num_epochs=200, lr=0.0002, l1_coef=50, l2_coef=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion = nn.BCELoss()
    l2_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    generator.to(device)
    discriminator.to(device)

    # generator.train()
    # discriminator.train()

    torch.backends.cudnn.benchmark = True

    batch_size = dataloader.batch_size

    iteration = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_g_loss = 0
        epoch_d_loss = 0

        print(f"Epoch {epoch + 1}/{num_epochs}")
        for minibatch in dataloader:
            try:
                # discriminatorの学習
                images = minibatch["image"]
                text_embeddings = minibatch["text_embedding"]

                images = images.to(device)
                text_embeddings = text_embeddings.to(device)

                minibatch_size = images.size()[0]
                label_real = torch.full((minibatch_size,), 1).to(device)
                label_fake = torch.full((minibatch_size,), 0).to(device)
                smoothed_label_real = torch.FloatTensor(smooth_label(label_real.to("cpu").numpy(), -0.1)).to(device)

                out_real, _ = discriminator(images, text_embeddings)

                input_noise = torch.randn(minibatch_size, 100).to(device)
                input_noise = input_noise.view(input_noise.size(0), input_noise.size(1), 1, 1)
                fake_images = generator(text_embeddings, input_noise)
                out_fake, _ = discriminator(fake_images, text_embeddings)

                d_loss_fake = criterion(out_fake, label_fake)
                d_loss_real = criterion(out_real, smoothed_label_real)

                d_loss = d_loss_real + d_loss_fake

                g_optimizer.zero_grad()
                d_optimizer.zero_grad()

                d_loss.backward()
                d_optimizer.step()

                # generatorの学習
                input_noise = torch.randn(minibatch_size, 100).to(device)
                input_noise = input_noise.view(input_noise.size(0), input_noise.size(1), 1, 1)
                fake_images = generator(text_embeddings, input_noise)
                out_fake, activation_fake = discriminator(fake_images, text_embeddings)
                _, activation_real = discriminator(images, text_embeddings)

                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)

                g_loss = (
                    criterion(out_fake, label_real)
                    + l2_coef * l2_loss(activation_fake, activation_real.detach())
                    + l1_coef * l1_loss(fake_images, images)
                )

                g_optimizer.zero_grad()
                d_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()

                if iteration % 10 == 0:
                    print(f"epoch: {epoch}, iter: {iteration}, d_loss: {d_loss}, g_loss: {g_loss}")

                iteration += 1
            except Exception as e:
                import traceback

                print(f"error: {type(e)}\n" + traceback.format_exc())

        elapsed_time = time.time() - start_time
        print(f"epoch: {epoch}, epoch_d_loss: {epoch_d_loss / batch_size}, epoch_g_loss: {epoch_g_loss / batch_size}")
        print(f"elapsed time: {elapsed_time} s")

    return generator, discriminator


def smooth_label(tensor, offset):
    return tensor + offset
