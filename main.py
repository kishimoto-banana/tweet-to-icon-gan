import pickle

import torch
from lib.dataloader import ImageTransform, Text2ImageDataset, make_datapath_list
from lib.gan import Discriminator, Generator
from lib.trainer import train

if __name__ == "__main__":
    print("make model")
    generator = Generator()
    discriminator = Discriminator()

    print("make dataset")
    image_path_list = make_datapath_list("./data/icon")
    mean = (155.1961187682143, 147.82546655771634, 140.17598230655122)
    std = (81.30347665033898, 79.53520195424258, 82.12983734451537)
    img_size = 64
    batch_size = 64
    with open("./data/text_embedding.pkl", "rb") as f:
        text_embedding = pickle.load(f)
    epochs = 2
    dataset = Text2ImageDataset(
        image_file_list=image_path_list,
        text_embedding=text_embedding,
        transform=ImageTransform(resize=img_size, mean=mean, std=std),
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    trained_generator, trained_discriminator = train(generator, discriminator, dataloader, num_epochs=epochs)
    torch.save(trained_generator.state_dict(), "./models/generator.pth")
    torch.save(trained_discriminator.state_dict(), "./models/discriminator.pth")
