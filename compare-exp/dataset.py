import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations as A
from albumentations import ToTensorV2
from torchvision import transforms


class HabitatDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.img_dir = os.path.join(root_dir, "JPEGImages")
        self.mask_dir = os.path.join(root_dir, "SegmentationClass")

        list_file = os.path.join(root_dir, f"{split}.txt")
        with open(list_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        img_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{image_id}.npg.npy")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.load(mask_path)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Return image ID during testing phase to save prediction results
        if self.split == 'test':
            return image, mask.long(), image_id
        else:
            return image, mask.long(), image_id


class HabitatDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=4, num_workers=8, img_size=1248):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size

        with open(os.path.join(self.root_dir, 'class_names.txt'), 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.num_classes = len(self.class_names)

    def setup(self, stage=None):
        train_transform = A.Compose([
            A.RandomCrop(self.img_size, self.img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)}, # Corresponds to shift_limit
                scale=(0.9, 1.1), # 1 - 0.2 = 0.8, 1 + 0.2 = 1.2 # Corresponds to scale_limit
                rotate=(-15, 15), # Corresponds to rotate_limit
                p=0.5,
                border_mode=cv2.BORDER_REFLECT
            ),
            A.OneOf([  # Randomly select one noise addition method
                A.ElasticTransform(alpha=1, sigma=50, p=0.3), # Fix: Remove alpha_affine
                A.ElasticTransform(alpha=0.5, sigma=60, approximate=True, p=0.1), # Alternative to GridDistortion (e.g., smoother deformation)
                A.ElasticTransform(alpha=2, sigma=40, approximate=True, p=0.3), # Alternative to PiecewiseAffine (e.g., more intense deformation)
            ], p=0.3),  # Elastic transformation etc. [[8]]
            A.OneOf([  # Randomly select one blur method
                A.Blur(blur_limit=3, p=0.1),
                A.MotionBlur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([  # Randomly select one color transformation method
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),  # Hue, saturation, brightness adjustment
            A.Resize(height=self.img_size, width=self.img_size),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ], bbox_params=None, keypoint_params=None)

        val_test_transform = A.Compose([
            A.CenterCrop(self.img_size, self.img_size),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ])

        if stage == 'fit' or stage is None:
            self.train_dataset = HabitatDataset(self.root_dir, 'train', transform=train_transform)
            self.val_dataset = HabitatDataset(self.root_dir, 'val', transform=val_test_transform)

        if stage == 'test' or stage is None:
            self.test_dataset = HabitatDataset(self.root_dir, 'test', transform=val_test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)


if __name__ == "__main__":
    data_module = HabitatDataModule(root_dir="../data/", batch_size=1, num_workers=2, img_size=1248)
    data_module.setup(stage="fit")
    loader = data_module.train_dataloader()
    for idx, batch in enumerate(loader):
        print(batch[0].shape)
        print(batch[0][0, :, 1, :])
        break
