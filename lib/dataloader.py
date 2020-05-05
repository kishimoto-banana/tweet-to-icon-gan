import pathlib

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def make_datapath_list(target_dir):
    path_list = [str(p) for p in pathlib.Path(target_dir).iterdir() if p.is_file()]
    return path_list


class ImageTransform:
    def __init__(self, resize, mean, std):
        self.data_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(resize, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __call__(self, img):
        return self.data_transform(img)


class Text2ImageDataset(Dataset):
    def __init__(self, image_file_list, text_embedding, transform):
        self.image_file_list = image_file_list
        self.text_embedding = text_embedding
        self.transform = transform

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, idx):
        image_path = self.image_file_list[idx]
        img = Image.open(image_path).convert("RGB")
        img_trans = self.transform(img)

        text_embedding = torch.FloatTensor(self.text_embedding[idx, :])

        sample = {
            "image": img_trans,
            "text_embedding": text_embedding,
        }

        return sample
