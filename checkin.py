import os
import sys
import time

import numpy as np
import requests
import torch
from bs4 import BeautifulSoup
from PIL import Image

from dataset import transform
from model import ResidualBlock, ResNet
from utils.utils import LabeltoStr, device, logger


DK_URL = "https://dk.shmtu.edu.cn/"
CAS_URL = "https://cas.shmtu.edu.cn/"
CAPTCHA_URL = CAS_URL + "cas/captcha"
CHECKIN_URL = DK_URL + "checkin"

model = ResNet(ResidualBlock).to("cpu")
model.eval()
model.load_state_dict(torch.load("model/best.pkl", map_location=device))


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


# https://github.com/airstone42/daka/blob/master/main.py
def login():
    soup = BeautifulSoup(r.content, "lxml")
    captcha = s.get(CAPTCHA_URL, stream=True)
    valid_code = detect(Image.open(captcha.raw))
    execution = soup.find("input", attrs={"type": "hidden", "name": "execution"})[
        "value"
    ]
    data = {
        "username": USERNAME,
        "password": PASSWORD,
        "validateCode": valid_code,
        "execution": execution,
        "_eventId": "submit",
        "geolocation": "",
    }
    post = s.post(r.url, data=data)
    soup = BeautifulSoup(post.content, "lxml")
    logger.info("Checkin login: Login...")
    return (
        (False, post)
        if soup.find("div", attrs={"class": "alert alert-danger"})
        else (True, post)
    )


def checkin():
    if r.url != DK_URL:
        return False
    soup = BeautifulSoup(r.content, "lxml")
    check = str(soup.find("div", attrs={"class": "form-group"}))
    print(check)
    flag = False
    if "Health report have not been submitted today" in check:
        flag = False
    if "Health report already submitted" in check:
        flag = True
    if not flag:
        data = {
            "xgh": USERNAME,
            "region": 1,
            "rylx": 4,
            "status": 0,
        }
        post = s.post(CHECKIN_URL, data=data)
        logger.info("Checkin: Checkin...")
        soup = BeautifulSoup(post.content, "lxml")
        if "Health report already submitted" in str(
            soup.find("div", attrs={"class": "form-group"})
        ):
            flag = True
    if flag:
        logger.info("Checkin: Checkin successful!")
    return flag


if __name__ == "__main__":
    USERNAME = sys.argv[1] if len(sys.argv) >= 2 else input("Input username:")
    PASSWORD = sys.argv[2] if len(sys.argv) >= 3 else input("Input password:")
    retry_count = 5

    s = requests.Session()
    r = s.get(DK_URL)
    for i in range(retry_count):
        result, r = login()
        if result:
            break
        else:
            logger.warning(f"Checkin login: {i} Try")
        time.sleep(5)

    for i in range(retry_count):
        result = checkin()
        if result:
            break
        else:
            logger.warning(f"Checkin checkin: {i} Try")
        time.sleep(5)

