import torch
import torch.nn as nn

import numpy as np
import cv2

from dataset import DocumentDataset
from model import LDRNet
from loss import WeightedLocLoss, LineLoss

import warnings
from tqdm import tqdm


line_loss = LineLoss()
weighted_loc_loss = WeightedLocLoss()
cross_entropy_loss = nn.CrossEntropyLoss()

torch.manual_seed(0)


def train(
    model,
    train_dataloader,
    optimizer,
    scheduler,
    class_list=[1],
    epochs=500,
    device=torch.device("cuda"),
    val_dataloader=None,
):
    coord_size = 200
    using_weights = True
    using_line_loss = True

    size_per_line = int((coord_size - 8) / 4 / 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for epoch in range(1, epochs + 1):
            model.train()
            print("Epoch: " + str(epoch))
            batch_losses = []
            for step, data in tqdm(
                enumerate(train_dataloader), total=len(train_dataloader)
            ):
                x, y = data
                x = x.to(device)
                y = y.to(device)
                y[:, 0:8] = y[:, 0:8] / 224
                coord_start = y[:, 0:8]
                coord_end = torch.concat([y[:, 2:8], y[:, 0:2]], dim=1)
                coord_increment = (coord_end - coord_start) / (size_per_line + 1)
                new_coord = coord_start + coord_increment
                for index in range(1, size_per_line):
                    new_coord = torch.concat(
                        [new_coord, coord_start + (index + 1) * coord_increment], dim=1
                    )

                if using_weights:
                    weights_start = y[:, 8:16]
                    weights_end = torch.concat([y[:, 10:16], y[:, 8:10]], dim=1)
                    weights_increment = (weights_end - weights_start) / (
                        size_per_line + 1
                    )
                    new_weights = weights_start + weights_increment
                    for index in range(1, size_per_line):
                        new_weights = torch.concat(
                            [
                                new_weights,
                                weights_start + (index + 1) * weights_increment,
                            ],
                            dim=1,
                        )
                    y = torch.concat(
                        [
                            coord_start,
                            new_coord,
                            weights_start,
                            new_weights,
                            torch.unsqueeze(y[:, 8 * 2], dim=1),
                        ],
                        dim=1,
                    )
                else:
                    y = torch.concat([new_coord, y[:, 8]], dim=1)
                optimizer.zero_grad()
                corner_y_, border_y_, class_y_ = model(x)
                coord_y_ = torch.concat([corner_y_, border_y_], dim=1)
                coord_y = y[:, 0:coord_size]
                if using_weights:
                    weights = y[:, coord_size : coord_size * 2]
                    y_end = coord_size * 2
                else:
                    weights = torch.ones(4 * 2) / torch.ones(4 * 2)
                    y_end = coord_size
                y__end = coord_size

                losses = []
                total_loss = 0
                for class_size in class_list:
                    class_y = y[:, y_end]
                    y_end += 1
                    y__end += class_size + 1
                    class_y = class_y.to(device, dtype=torch.long)
                    class_loss = cross_entropy_loss(class_y_, class_y)
                    losses.append(class_loss)
                    total_loss += class_loss
                loc_loss = 50 * weighted_loc_loss(coord_y, coord_y_, weights)
                total_loss += loc_loss * 1

                if coord_size > 8:
                    total_slop_loss = 0
                    total_diff_loss = 0
                    for index in range(4):
                        line = coord_y_[:, index * 2 : index * 2 + 2]
                        for coord_index in range(size_per_line):
                            line = torch.concat(
                                [
                                    line,
                                    coord_y_[
                                        :,
                                        8
                                        + coord_index * 8
                                        + index * 2 : 8
                                        + coord_index * 8
                                        + index * 2
                                        + 2,
                                    ],
                                ],
                                dim=1,
                            )
                        line = torch.concat(
                            [
                                line,
                                coord_y_[
                                    :, (index * 2 + 2) % 8 : (index * 2 + 2 + 2) % 8
                                ],
                            ],
                            dim=1,
                        )
                        cur_slop_loss, cur_diff_loss = line_loss(line)
                        if using_weights:
                            total_slop_loss += cur_slop_loss * torch.mean(weights)
                            total_diff_loss += cur_diff_loss * torch.mean(weights)

                    if using_line_loss:
                        losses.append(total_slop_loss * 0.01)
                        losses.append(total_diff_loss * 0.01)
                        total_loss += total_slop_loss * 0.01
                        total_loss += total_diff_loss * 0.01
                    else:
                        losses.append(0)
                        losses.append(0)
                    total_loss += 0
                    total_loss += 0

                loss, loss_list, y_ = total_loss, losses, [coord_y_, class_y_]
                batch_loss_value = loss.item()
                loss.backward()
                optimizer.step()
                batch_losses.append(batch_loss_value)
            print("[INFO]: Training loss:", torch.Tensor(batch_losses).mean().item())
            scheduler.step()

            if val_dataloader is not None:
                validate(model, val_dataloader, class_list=class_list, device=device)


def validate(
    model,
    val_dataloader,
    class_list=[1],
    device=torch.device("cpu"),
):
    coord_size = 200
    using_weights = True
    using_line_loss = True

    size_per_line = int((coord_size - 8) / 4 / 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.eval()
        batch_losses = []
        with torch.no_grad():
            for step, data in tqdm(
                enumerate(val_dataloader), total=len(val_dataloader)
            ):
                x, y = data
                x = x.to(device)
                y = y.to(device)
                y[:, 0:8] = y[:, 0:8] / 224
                coord_start = y[:, 0:8]
                coord_end = torch.concat([y[:, 2:8], y[:, 0:2]], dim=1)
                coord_increment = (coord_end - coord_start) / (size_per_line + 1)
                new_coord = coord_start + coord_increment
                for index in range(1, size_per_line):
                    new_coord = torch.concat(
                        [new_coord, coord_start + (index + 1) * coord_increment], dim=1
                    )

                if using_weights:
                    weights_start = y[:, 8:16]
                    weights_end = torch.concat([y[:, 10:16], y[:, 8:10]], dim=1)
                    weights_increment = (weights_end - weights_start) / (
                        size_per_line + 1
                    )
                    new_weights = weights_start + weights_increment
                    for index in range(1, size_per_line):
                        new_weights = torch.concat(
                            [
                                new_weights,
                                weights_start + (index + 1) * weights_increment,
                            ],
                            dim=1,
                        )
                    y = torch.concat(
                        [
                            coord_start,
                            new_coord,
                            weights_start,
                            new_weights,
                            torch.unsqueeze(y[:, 8 * 2], dim=1),
                        ],
                        dim=1,
                    )
                else:
                    y = torch.concat([new_coord, y[:, 8]], dim=1)
                optimizer.zero_grad()
                corner_y_, border_y_, class_y_ = model(x)
                coord_y_ = torch.concat([corner_y_, border_y_], dim=1)
                coord_y = y[:, 0:coord_size]

                # Debug
                debug_ = corner_y_[0]
                debug_ = debug_[0:8]
                debug_ = [int(x * 224) for x in debug_]

                new_image = np.ascontiguousarray(
                    (torch.clone(x[0]).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
                    * 255,
                    dtype=np.uint8,
                )
                print(new_image.shape)

                cv2.circle(new_image, (debug_[0], debug_[1]), 3, (0, 0, 255), -1)
                cv2.circle(new_image, (debug_[2], debug_[3]), 3, (0, 255, 255), -1)
                cv2.circle(new_image, (debug_[4], debug_[5]), 3, (255, 0, 0), -1)
                cv2.circle(new_image, (debug_[6], debug_[7]), 3, (0, 255, 0), -1)
                cv2.imwrite("pred.jpg", new_image[:, :, ::-1])

                debug = coord_y[0]
                debug = debug[0:8]
                debug = [int(x * 224) for x in debug]

                new_image = np.ascontiguousarray(
                    (torch.clone(x[0]).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
                    * 255,
                    dtype=np.uint8,
                )
                cv2.circle(new_image, (debug[0], debug[1]), 3, (0, 0, 255), -1)
                cv2.circle(new_image, (debug[2], debug[3]), 3, (0, 255, 255), -1)
                cv2.circle(new_image, (debug[4], debug[5]), 3, (255, 0, 0), -1)
                cv2.circle(new_image, (debug[6], debug[7]), 3, (0, 255, 0), -1)
                cv2.imwrite("truth.jpg", new_image[:, :, ::-1])

                print([debug_, debug])

                if using_weights:
                    weights = y[:, coord_size : coord_size * 2]
                    y_end = coord_size * 2
                else:
                    weights = torch.ones(4 * 2) / torch.ones(4 * 2)
                    y_end = coord_size
                y__end = coord_size

                losses = []
                total_loss = 0
                for class_size in class_list:
                    class_y = y[:, y_end]
                    y_end += 1
                    y__end += class_size + 1
                    class_y = class_y.to(device, dtype=torch.long)
                    class_loss = cross_entropy_loss(class_y_, class_y)
                    losses.append(class_loss)
                    total_loss += class_loss
                loc_loss = weighted_loc_loss(coord_y, coord_y_, weights)
                total_loss += loc_loss * 1

                if coord_size > 8:
                    total_slop_loss = 0
                    total_diff_loss = 0
                    for index in range(4):
                        line = coord_y_[:, index * 2 : index * 2 + 2]
                        for coord_index in range(size_per_line):
                            line = torch.concat(
                                [
                                    line,
                                    coord_y_[
                                        :,
                                        8
                                        + coord_index * 8
                                        + index * 2 : 8
                                        + coord_index * 8
                                        + index * 2
                                        + 2,
                                    ],
                                ],
                                dim=1,
                            )
                        line = torch.concat(
                            [
                                line,
                                coord_y_[
                                    :, (index * 2 + 2) % 8 : (index * 2 + 2 + 2) % 8
                                ],
                            ],
                            dim=1,
                        )
                        cur_slop_loss, cur_diff_loss = line_loss(line)
                        if using_weights:
                            total_slop_loss += cur_slop_loss * torch.mean(weights)
                            total_diff_loss += cur_diff_loss * torch.mean(weights)

                    if using_line_loss:
                        losses.append(total_slop_loss * 0.01)
                        losses.append(total_diff_loss * 0.01)
                        total_loss += total_slop_loss * 0.01
                        total_loss += total_diff_loss * 0.01
                    else:
                        losses.append(0)
                        losses.append(0)
                    total_loss += 0
                    total_loss += 0

                loss, loss_list, y_ = total_loss, losses, [coord_y_, class_y_]
                batch_loss_value = loss.item()
                batch_losses.append(batch_loss_value)
        print("[INFO]: Validation loss:", torch.Tensor(batch_losses).mean().item())
        torch.save(model.state_dict(), "last.pth")


if __name__ == "__main__":
    from torch.utils.data import DataLoader, random_split
    from torch.optim.lr_scheduler import StepLR

    # import torchvision.transforms as transforms
    import torch.optim as optim

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transform = transforms.Compose(
    #     [
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=(0.5), std=(0.5)),
    #     ]
    # )

    # transform = A.Compose(
    #     [
    #         A.Resize(224, 224),
    #         A.Normalize(mean=(0.5), std=(0.5)),
    #         ToTensorV2(),
    #     ],
    #     keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    # )

    transform = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.5), std=(0.5)),
            A.RandomCrop(224, 224),
            A.RandomBrightnessContrast(p=0.2),
            A.MotionBlur(3, p=0.01),
            A.ToGray(p=0.01),
            A.Perspective(p=0.2),
            A.Rotate(p=0.34, limit=[-90, 90], border_mode=cv2.BORDER_CONSTANT, value=0),
            A.CoarseDropout(
                max_holes=3,
                max_height=64,
                max_width=64,
                min_height=24,
                min_width=24,
                min_holes=2,
                accept_keypoint_in_holes=True,
                p=0.34,
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

    size = len(dataset)
    train_size = int(size * 0.9)
    val_size = size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=128, shuffle=True, num_workers=4
    )

    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=64, shuffle=False, num_workers=4
    )

    model = LDRNet()
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"[INFO]: {total_params:,} total parameters.")
    print(f"[INFO]: {total_trainable_params:,} trainable parameters.")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

    train(
        model,
        train_loader,
        optimizer,
        scheduler,
        device=device,
        epochs=500,
        val_dataloader=val_dataloader,
    )
