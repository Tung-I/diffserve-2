import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch


class RealFakeImageDataset(Dataset):
    def __init__(self, real_dir, fake_dirs, clip_preprocess, max_images_per_prompt=4):
        self.samples = []
        self.transform = clip_preprocess

        # Load real images
        real_root = Path(real_dir)
        for prompt_dir in real_root.glob("prompt_*"):
            count = 0
            for img_path in sorted(prompt_dir.glob("*.jpg")):
                if count >= max_images_per_prompt * 4:
                    break
                self.samples.append((str(img_path), 1))
                count += 1

        # Load fake images
        for fake_dir in fake_dirs:
            fake_root = Path(fake_dir)
            for prompt_dir in fake_root.glob("prompt_*"):
                count = 0
                for img_path in sorted(prompt_dir.glob("*.jpg")):
                    if count >= max_images_per_prompt:
                        break
                    self.samples.append((str(img_path), 0))
                    count += 1

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = self.transform(Image.open(img_path).convert("RGB"))
        return image, torch.tensor(label, dtype=torch.long)
