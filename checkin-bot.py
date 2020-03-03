#!/usr/bin/env python
import datetime
import json
import os
import sys
from random import SystemRandom

import numpy as np
import requests
import torch
from bs4 import BeautifulSoup
from PIL import Image
from telegram.ext import CommandHandler, Updater
from telegram.ext.dispatcher import run_async

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


def login(s, username, password):
    home = s.get(DK_URL)
    soup = BeautifulSoup(home.content, "lxml")
    captcha = s.get(CAPTCHA_URL, stream=True)
    valid_code = detect(Image.open(captcha.raw))
    execution = soup.find("input", attrs={"type": "hidden", "name": "execution"})
    data = {
        "username": username,
        "password": password,
        "validateCode": valid_code,
        "execution": execution.get("value"),
        "_eventId": "submit",
        "geolocation": "",
    }
    post = s.post(home.url, data=data)
    logger.info(f"Login: {username} Login...")
    return True if post.url == DK_URL else False


def checkin(s, username, region):
    data = {
        "xgh": username,
        "lon": "",
        "lat": "",
        "region": region,
        "rylx": 4,
        "status": 0,
    }
    s.post(CHECKIN_URL, data=data)
    logger.info(f"Checkin: {username} Checkin...")
    home = s.get(DK_URL)
    soup = BeautifulSoup(home.content, "lxml")
    return (
        True
        if "success" in str(soup.find("div", attrs={"class": "form-group"}))
        else False
    )


@run_async
def checkin_queue(context):
    job = context.job
    username, password, region, chat = (
        job.context.get("username"),
        job.context.get("password"),
        job.context.get("region"),
        job.context.get("chat"),
    )
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36"
        }
    )
    retry_count = 5
    message = context.bot.send_message(
        chat, f"Job: Running for {username}", disable_notification=True
    )
    for i in range(retry_count):
        result = login(s, username, password)
        if result:
            append_text = f"Login: {username} Success!"
            logger.info(append_text)
            message = message.edit_text(f"{message.text}\n{append_text}")
            break
        else:
            append_text = f"Login: {username} Fail {i}"
            logger.warning(append_text)
            message = message.edit_text(f"{message.text}\n{append_text}")
    for i in range(retry_count):
        result = checkin(s, username, region)
        if result:
            append_text = f"Checkin: {username} Success!"
            logger.info(append_text)
            message = message.edit_text(f"{message.text}\n{append_text}")
            return
        else:
            append_text = f"Checkin: {username} Fail {i}"
            logger.warning(append_text)
            message = message.edit_text(f"{message.text}\n{append_text}")
    message.reply_text("Job failed! Planning to run in next hour.")
    context.job_queue.run_once(
        checkin_queue,
        SystemRandom().randint(1800, 3600),
        context={
            "username": username,
            "password": password,
            "region": region,
            "chat": chat,
        },
    )


@run_async
def start(update, context):
    message = update.message
    chat = message.forward_from_chat if message.forward_from_chat else message.chat
    jobs = [t.name for t in context.job_queue.jobs()]
    message.reply_markdown(
        f"Usage:\n/add <username> <password> \[region-num]\nregin-num: \n1 - 上海\n2 - 湖北\n3 - 其他中国地区\n5 - 国外\n/del <username>\nCHAT ID: `{chat.id}`\nCurrent Jobs: {jobs}"
    )
    logger.info(f"Start command: Current Jobs: {jobs}")


@run_async
def add(update, context):
    message = update.message
    chat = message.chat
    data = message.text.split(" ")
    if len(data) < 3:
        message.reply_text(
            "Usage:\n/add <username> <password> [region-num]\nregin-num: \n1 - 上海\n2 - 湖北\n3 - 其他中国地区\n5 - 国外\n"
        )
        return
    username, password = data[1], data[2]
    region = 1 if len(data) <= 3 else data[3]
    chat_id = chat.id if len(data) <= 4 else data[4]
    for job in context.job_queue.get_jobs_by_name(username):
        job.schedule_removal()
    jobs = [t.name for t in context.job_queue.jobs()]
    context.job_queue.run_daily(
        checkin_queue,
        datetime.time(
            0,
            min(3 + len(jobs), 59),
            SystemRandom().randrange(60),
            SystemRandom().randrange(1000000),
            datetime.timezone(datetime.timedelta(hours=8)),
        ),
        context={
            "username": username,
            "password": password,
            "region": region,
            "chat": chat.id,
        },
        name=username,
    )
    jobs.append(username)
    context.job_queue.run_once(
        checkin_queue,
        1,
        context={
            "username": username,
            "password": password,
            "region": region,
            "chat": chat_id,
        },
    )
    message.reply_text(
        f"Added successfully!\nusername: {username}\npassword: {password}\nCurrent Jobs: {jobs}"
    )
    logger.info(f"Added Jobs: {username}, Current Jobs: {jobs}")


@run_async
def delete(update, context):
    message = update.message
    chat = message.chat
    data = message.text.split(" ")
    if len(data) < 2:
        message.reply_text("Usage:\n/del <username>")
        return
    username = data[1]
    deleted_flag = False
    jobs = [t.name for t in context.job_queue.jobs()]
    for job in context.job_queue.get_jobs_by_name(username):
        if job.context.get("chat") == chat.id:
            deleted_flag = True
            job.schedule_removal()
            logger.info(f"Deleted Jobs: {username}, Current Jobs: {jobs}")
    if deleted_flag:
        message.reply_text(
            f"Deleted successfully!\nusername: {username}\nCurrent Jobs: {jobs}"
        )
    else:
        message.reply_text("You cannot delete it.")


@run_async
def error(update, context):
    logger.warning(f"Update {context} caused error {error}")


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
        config = load_json(sys.argv[1])
    else:
        config = load_json()
    TOKEN, CHAT = config.get("TOKEN"), config.get("CHAT")
    logger.info(f"Bot: Starting & Sending to {CHAT}")
    updater = Updater(
        TOKEN, use_context=True, request_kwargs=config.get("REQUEST_KWARGS")
    )
    updater.job_queue.run_daily(
        checkin_queue,
        datetime.time(0, 2, 0, 0, datetime.timezone(datetime.timedelta(hours=8))),
        context={
            "username": config.get("USERNAME"),
            "password": config.get("PASSWORD"),
            "chat": config.get("CHAT"),
            "region": config.get("REGION", 1),
        },
        name=config.get("USERNAME"),
    )
    updater.dispatcher.add_handler(CommandHandler("start", start))
    updater.dispatcher.add_handler(CommandHandler("add", add))
    updater.dispatcher.add_handler(CommandHandler("del", delete))
    updater.dispatcher.add_error_handler(error)
    updater.start_polling()
    updater.idle()
