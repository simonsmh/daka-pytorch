import json
import os
import sys

import httpx
import numpy as np
import torch
import uvloop
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from bs4 import BeautifulSoup
from PIL import Image

from dataset import transform
from model import ResidualBlock, ResNet
from utils.utils import LabeltoStr, device, logger

DK_URL = "https://dk.shmtu.edu.cn/"
CAPTCHA_URL = "https://cas.shmtu.edu.cn/cas/captcha"
CHECKIN_URL = DK_URL + "checkin"
ARRSH_URL = DK_URL + "arrsh"

model = ResNet(ResidualBlock)
model.eval()
model.load_state_dict(torch.load("model/best.pkl", map_location=device))


async def user(username, password, region, retry_count=5):
    async def login(client, username, password):
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

        home = await client.get(DK_URL)
        soup = BeautifulSoup(home.content, "lxml")
        captcha = await client.get(CAPTCHA_URL)
        valid_code = detect(Image.open(captcha))
        execution = soup.find("input", attrs={"type": "hidden", "name": "execution"})
        data = {
            "username": username,
            "password": password,
            "validateCode": valid_code,
            "execution": execution.get("value"),
            "_eventId": "submit",
            "geolocation": "",
        }
        post = await client.post(home.url, data=data)
        logger.info(f"Login: {username} Login...")
        return True if post.url == DK_URL else False

    async def checkin(client, username, region):
        data = {
            "xgh": username,
            "lon": "",
            "lat": "",
            "region": region,
            "rylx": 4,
            "status": 0,
        }
        await client.post(CHECKIN_URL, data=data)
        logger.info(f"Checkin: {username} Checkin...")
        home = await client.get(DK_URL)
        soup = BeautifulSoup(home.content, "lxml")
        return (
            True
            if "success" in str(soup.find("div", attrs={"class": "form-group"}))
            else False
        )

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36"
    }
    async with httpx.AsyncClient(headers=headers, timeout=None, verify=False) as client:
        for i in range(retry_count):
            result = await login(client, username, password)
            if result:
                logger.info("Login: Successful!")
                break
            else:
                logger.warning(f"Login: {i} Try")
        for i in range(retry_count):
            result = await checkin(client, username, region)
            if result:
                logger.info("Checkin: Successful!")
                break
            else:
                logger.warning(f"Checkin: {i} Try")


def load_json(filename="config.json"):
    try:
        with open(filename, "r") as file:
            config = json.load(file)
    except FileNotFoundError:
        try:
            filename = f"{os.path.split(os.path.realpath(__file__))[0]}/{filename}"
            with open(filename, "r") as file:
                config = json.load(file)
        except FileNotFoundError:
            logger.exception(f"Cannot find {filename}.")
            sys.exit(1)
    logger.info(f"Json: Loaded {filename}")
    return config


if __name__ == "__main__":
    if len(sys.argv) >= 2 and os.path.exists(sys.argv[1]):
        configs = load_json(sys.argv[1])
    else:
        configs = {
            "USERS": [
                {
                    "USERNAME": input("学号: "),
                    "PASSWORD": input("密码: "),
                    "REGION": int(input("地区: ") or 1),
                }
            ]
        }
    loop = uvloop.new_event_loop()
    scheduler = AsyncIOScheduler(event_loop=loop)
    scheduler.start()
    for i, config in enumerate(configs.get("USERS", [])):
        job = scheduler.add_job(
            user,
            "cron",
            args=[
                config.get("USERNAME"),
                config.get("PASSWORD"),
                config.get("REGION", 1),
            ],
            name=config.get("USERNAME"),
            hour=0,
            minute=2 + i,
            jitter=30,
        )
        logger.info(job.next_run_time)
    loop.run_forever()
