import torch
import torch.nn as nn


class WeightedLocLoss(nn.Module):
    def __init__(self):
        super(WeightedLocLoss, self).__init__()

    def forward(self, y_true, y_pred, weights):
        return torch.mean(torch.square(y_pred - y_true) * weights)


class LineLoss(nn.Module):
    def __init__(self):
        super(LineLoss, self).__init__()

    def forward(self, line):
        line_x = line[:, 0::2]
        line_y = line[:, 1::2]
        x_diff = line_x[:, 1:] - line_x[:, 0:-1]
        y_diff = line_y[:, 1:] - line_y[:, 0:-1]
        x_diff_start = x_diff[:, 1:]
        x_diff_end = x_diff[:, 0:-1]
        y_diff_start = y_diff[:, 1:]
        y_diff_end = y_diff[:, 0:-1]
        similarity = (x_diff_start * x_diff_end + y_diff_start * y_diff_end) / (
            torch.sqrt(torch.square(x_diff_start) + torch.square(y_diff_start))
            * torch.sqrt(torch.square(x_diff_end) + torch.square(y_diff_end))
            + 0.0000000000001
        )
        slop_loss = torch.mean(1 - similarity)
        x_diff_loss = torch.mean(torch.square(x_diff[:, 1:] - x_diff[:, 0:-1]))
        y_diff_loss = torch.mean(torch.square(y_diff[:, 1:] - y_diff[:, 0:-1]))

        return slop_loss, x_diff_loss + y_diff_loss


if __name__ == "__main__":
    loss = WeightedLocLoss()
    res = loss(torch.ones((1, 8)), torch.zeros((1, 8)), torch.ones((1, 8)))
    print(res)
