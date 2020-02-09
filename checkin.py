import os
import sys

import numpy as np
import requests
import torch
from bs4 import BeautifulSoup
from PIL import Image

from dataset import transform
from model import ResidualBlock, ResNet
from utils.utils import LabeltoStr, device, logger

DK_URL = "https://dk.shmtu.edu.cn/"
CAPTCHA_URL = "https://cas.shmtu.edu.cn/cas/captcha"
CHECKIN_URL = DK_URL + "checkin"

model = ResNet(ResidualBlock)
model.eval()
model.load_state_dict(torch.load("model/best.pkl", map_location=device))


def detect(Img):
    logger.info(f"Detect: Detecting...")
    i = transform(Img).unsqueeze(0)
    predict_label1, predict_label2 = model(i)
    predict_label = LabeltoStr(
        [
            np.argmax(predict_label1.data.numpy()[0]),
            np.argmax(predict_label2.data.numpy()[0]),
        ]
    )
    logger.info(f"Detect: Result {predict_label}")
    return predict_label


# https://github.com/airstone42/daka/blob/master/main.py
def login():
    home = s.get(DK_URL)
    soup = BeautifulSoup(home.content, "lxml")
    captcha = s.get(CAPTCHA_URL, stream=True)
    valid_code = detect(Image.open(captcha.raw))
    execution = soup.find("input", attrs={"type": "hidden", "name": "execution"})
    data = {
        "username": USERNAME,
        "password": PASSWORD,
        "validateCode": valid_code,
        "execution": execution.get("value"),
        "_eventId": "submit",
        "geolocation": "",
    }
    post = s.post(home.url, data=data)
    logger.info("Login: Login...")
    return True if post.url == DK_URL else False


def checkin():
    data = {
        "xgh": USERNAME,
        "lon": "",
        "lat": "",
        "region": 1,
        "rylx": 4,
        "status": 0,
    }
    post = s.post(CHECKIN_URL, data=data)
    logger.info("Checkin: Checkin...")
    soup = BeautifulSoup(post.content, "lxml")
    return (
        True
        if "success" in str(soup.find("div", attrs={"class": "form-group"}))
        else False
    )


if __name__ == "__main__":
    USERNAME = sys.argv[1] if len(sys.argv) >= 2 else input("Input username:")
    PASSWORD = sys.argv[2] if len(sys.argv) >= 3 else input("Input password:")
    retry_count = 5
    s = requests.Session()
    s.headers.update({'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36'})
    for i in range(retry_count):
        result = login()
        if result:
            logger.info("Login: Successful!")
            break
        else:
            logger.warning(f"Login: {i} Try")
    for i in range(retry_count):
        result = checkin()
        if result:
            logger.info("Checkin: Successful!")
            break
        else:
            logger.warning(f"Checkin: {i} Try")
