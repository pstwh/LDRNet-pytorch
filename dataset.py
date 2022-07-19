import numpy as np
import torch
from torch.utils.data import Dataset

from pathlib import Path
from PIL import Image
import albumentations as A


class DocumentDataset(Dataset):
    def __init__(self, img_folder, txt_path, transform=None):
        self.img_folder = Path(img_folder)
        self.txt_path = txt_path
        self.transform = transform

        with open(txt_path) as fr:
            self.lines = fr.readlines()
            self.lines = [line.strip() for line in self.lines]

    def __getitem__(self, index):
        line = self.lines[index]
        line = line.split(",")
        img_name = line[0]
        img = Image.open(self.img_folder / img_name)
        y = list(map(float, line[1:]))
        keypoints = np.array(y[:8]).astype(np.int64)
        keypoints = list(map(list, keypoints.reshape(-1, 2)))
        keypoints = [val + [i] for i, val in enumerate(keypoints)]

        if self.transform:
            img = self.transform(image=np.array(img), keypoints=keypoints)
            keypoints = img["keypoints"]
            img = img["image"]
            keypoints = [i[:-1] for i in sorted(keypoints, key=lambda x: x[2])]
            keypoints = list(np.array(keypoints).flatten())[:8]
            y[:8] = keypoints

        x = img
        y = torch.Tensor(y)

        return x, y

    def __len__(self):
        return len(self.lines)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # import torchvision.transforms as transforms

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # transform = transforms.Compose(
    #     [
    #         transforms.Resize((224, 224)),
    #         transforms.Normalize(mean=(0.5), std=(0.5)),
    #         transforms.ToTensor(),
    #     ]
    # )

    transform = A.Compose(
        [
            A.Resize(320, 320),
            A.Normalize(mean=(0.5), std=(0.5)),
            A.RandomCrop(224, 224),
            A.RandomBrightnessContrast(p=0.3),
            A.MotionBlur(3, p=0.5),
            A.ToGray(p=0.05),
            A.Perspective(p=0.5),
            A.Rotate(p=0.5, limit=[-90, 90]),
            A.CoarseDropout(
                max_holes=3,
                max_height=48,
                max_width=48,
                min_height=24,
                min_width=24,
                min_holes=2,
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    dataset = DocumentDataset(
        "LDR",
        "labels.txt",
        transform=transform,
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)

    for data in loader:
        print(data[0])
        print(data[1])
        break
