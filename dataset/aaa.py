import os
import random
from typing import Callable, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image  # 用来加载图像
from tqdm import tqdm


class ImageInfo:
    def __init__(self, img_path: str, label: list):
        self.data = {'image_path': img_path, 'label': label}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self):
        return f"ImageInfo\n {self.data}"


class A007Dataset(Dataset):
    def __init__(self, txt_file: str, root_dir: str, transform: Optional[Callable] = None, seed: Optional[int] = 42):
        self.root_dir = root_dir
        self.transform = transform
        self.image_infos = list()

        if seed is not None:
            random.seed(seed)

        self._load_data(txt_file)

    def _load_data(self, txt_file: str):
        txt_path = os.path.join(self.root_dir, txt_file)
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"File {txt_path} not exists!")

        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                image_path = os.path.join(self.root_dir, 'Training', parts[0])
                label = [int(x) for x in parts[1]]
                self.image_infos.append(ImageInfo(image_path, label))

    def __getitem__(self, idx):
        results = self.image_infos[idx]

        # 加载图像
        img_path = results['image_path']
        img = Image.open(img_path).convert('RGB')

        # 如果有 transform，应用它
        if self.transform:
            img = self.transform(img)  # 注意，这里传递的是图像，不是整个 ImageInfo

        label_tensor = torch.tensor(results['label'], dtype=torch.long)  # 将标签转换为 Tensor
        return img, label_tensor

    def __len__(self):
        return len(self.image_infos)


def collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)


if __name__ == '__main__':
    import time
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像大小调整为224x224
        transforms.RandomCrop(224),  # 随机裁剪224x224区域
        transforms.ToTensor(),  # 将图像转换为 Tensor
        transforms.Normalize(mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
                             std=[58.395 / 255, 57.12 / 255, 57.375 / 255])  # 使用标准化
    ])

    dataset = A007Dataset(txt_file="train.txt",
                          root_dir="D:\\code\\A07\\dataset",
                          transform=transform,
                          seed=42)

    # 使用 DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # 测试加载时间
    time_start = time.time()
    for images, labels in dataloader:
        time_cost = time.time() - time_start
        print(time_cost)
        time_start = time.time()
