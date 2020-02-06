import numpy as np
import torch

from dataset import get_predict_data_loader
from model import ResidualBlock, ResNet
from utils.utils import LabeltoStr, StrtoLabel, logger


def main():
    model = ResNet(ResidualBlock).to("cpu")
    model.eval()
    model.load_state_dict(torch.load("model/best.pkl"))
    logger.info("Valid: loaded model")

    predict_dataloader = get_predict_data_loader()

    for i, (images, labels) in enumerate(predict_dataloader):
        predict_label1, predict_label2 = model(images)
        predict_label = LabeltoStr(
            [
                np.argmax(predict_label1.data.numpy()[0]),
                np.argmax(predict_label2.data.numpy()[0]),
            ]
        )
        true_label = LabeltoStr(labels.data.numpy()[0])
        logger.info(
            f"Test: {i}, Expect: {true_label}, Predict: {predict_label}, Result: {True if true_label == predict_label else False}"
        )


if __name__ == "__main__":
    main()
