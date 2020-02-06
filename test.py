# -*- coding: UTF-8 -*-
import numpy as np
import torch
from tqdm import tqdm

from dataset import get_test_data_loader
from model import ResidualBlock, ResNet
from utils.utils import LabeltoStr, StrtoLabel, device, logger


def main():
    model = ResNet(ResidualBlock).to(device)
    model.eval()
    model.load_state_dict(torch.load("model/best.pkl"))
    logger.info("Test: loaded model")

    test_dataloader = get_test_data_loader()

    correct = 0
    total = 0
    for i, (images, labels) in enumerate(tqdm(test_dataloader)):
        images = images.to(device)
        predict_label1, predict_label2 = model(images)
        predict_label = LabeltoStr(
            [
                np.argmax(predict_label1.to("cpu").data.numpy()[0]),
                np.argmax(predict_label2.to("cpu").data.numpy()[0]),
            ]
        )
        true_label = LabeltoStr(labels.data.numpy()[0])
        total += labels.size(0)
        if predict_label == true_label:
            correct += 1
    logger.info(
        f"Test finished! Accuracy on {total} test images: {100 * correct / total}%"
    )


if __name__ == "__main__":
    main()
