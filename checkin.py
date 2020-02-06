import torch
import numpy as np
from model import ResidualBlock, ResNet
from utils.utils import LabeltoStr, StrtoLabel, logger
from PIL import Image
from dataset import transform

model = ResNet(ResidualBlock).to("cpu")
model.eval()
model.load_state_dict(torch.load("model/best.pkl"))


def detect(Img):
    i = transform(Img).unsqueeze(0)
    predict_label1, predict_label2 = model(i)
    predict_label = LabeltoStr(
        [
            np.argmax(predict_label1.data.numpy()[0]),
            np.argmax(predict_label2.data.numpy()[0]),
        ]
    )
    logger.info(f"Checkin: Detect result {predict_label}")
    return predict_label


if __name__ == "__main__":
    I = Image.open("data/valid/00_1.jpg")
    print(detect(I))
