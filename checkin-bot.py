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
        chat, f"任务: 正在为 {username} 运行...", disable_notification=True
    )
    for i in range(retry_count):
        try:
            if login(s, username, password):
                append_text = f"登录: {username} 成功!"
                logger.info(f"Login: {username} Success!")
                message = message.edit_text(f"{message.text}\n{append_text}")
                break
        except:
            continue
        append_text = f"登录: {username} 重试次数 {i}"
        logger.warning(f"Login: {username} Fail {i}")
        message = message.edit_text(f"{message.text}\n{append_text}")
    for i in range(retry_count):
        try:
            if checkin(s, username, region):
                append_text = f"打卡: {username} 成功!"
                logger.info(f"Checkin: {username} Success!")
                message = message.edit_text(f"{message.text}\n{append_text}")
                return
        except:
            continue
        append_text = f"打卡: {username} 重试次数 {i}"
        logger.warning(f"Checkin: {username} Fail {i}")
        message = message.edit_text(f"{message.text}\n{append_text}")
    message.reply_text("任务执行失败! 预计下个小时将继续执行。")
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
    logger.warning(f"Job: {username} fail -> run in next hour")


@run_async
def start(update, context):
    message = update.message
    jobs = [t.name for t in context.job_queue.jobs()]
    message.reply_text(
        "用法:\n"
        "添加数字平台账户:\n"
        "/add <学号> <密码> [地区]\n"
        "地区(默认上海):\n"
        "1 - 上海\n"
        "2 - 湖北\n"
        "3 - 其他中国地区\n"
        "5 - 国外\n"
        "移除数字平台账户:\n"
        "/del <学号>\n"
        "立即运行:\n"
        "/run [学号]\n"
        f"现在的任务列表: {jobs}"
    )
    logger.info(
        f"Start command: Current Jobs: {[t.context for t in context.job_queue.jobs()]}"
    )


@run_async
def add(update, context):
    message = update.message
    chat = message.chat
    data = message.text.split(" ")
    if len(data) < 3:
        message.reply_text(
            "用法:\n"
            "添加数字平台账户:\n"
            "/add <学号> <密码> \\[地区]\n"
            "地区:\n"
            "1 - 上海\n"
            "2 - 湖北\n"
            "3 - 其他中国地区\n"
            "5 - 国外"
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
            "chat": chat_id,
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
        f"添加成功!\n学号: {username}\n密码: {password}\n地区: {region}\n现在的任务列表: {jobs}"
    )
    logger.info(f"Added Jobs: {username}, Current Jobs: {jobs}")


@run_async
def delete(update, context):
    message = update.message
    chat = message.chat
    data = message.text.split(" ")
    if len(data) < 2:
        message.reply_text("用法:\n移除数字平台账户:\n/del <学号>")
        return
    username = data[1]
    deleted_flag = False
    jobs = [t.name for t in context.job_queue.jobs()]
    for job in context.job_queue.get_jobs_by_name(username):
        if job.context.get("chat") in [chat.id, ADMIN]:
            deleted_flag = True
            job.schedule_removal()
            logger.info(f"Deleted Jobs: {username}, Current Jobs: {jobs}")
    if deleted_flag:
        message.reply_text(f"删除成功!\n学号: {username}\n现在的任务列表: {jobs}")
    else:
        message.reply_text("您没有删除此账户的权限.")


@run_async
def run(update, context):
    message = update.message
    chat = message.chat
    data = message.text.split(" ")
    if len(data) > 1:
        if len(data[1]) == 12:
            jobs = [
                t.context
                for t in context.job_queue.jobs()
                if t.context.get("username") == data[1]
            ]
        elif data[1] == "all":
            jobs = [t.context for t in context.job_queue.jobs()]
        else:
            jobs = [
                t.context
                for t in context.job_queue.jobs()
                if t.context.get("chat") == data[1]
            ]
    else:
        jobs = [
            t.context
            for t in context.job_queue.jobs()
            if t.context.get("chat") == chat.id
        ]
    if jobs:
        for job in jobs:
            context.job_queue.run_once(
                checkin_queue,
                1,
                context={
                    "username": job.get("username"),
                    "password": job.get("password"),
                    "region": job.get("region"),
                    "chat": chat.id,
                },
            )
    else:
        message.reply_text("未找到账户，请先使用 /add 命令添加！\n用法:\n立即运行:\n/run [学号]")


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
    TOKEN, ADMIN = config.get("TOKEN"), config.get("ADMIN")
    logger.info(f"Bot: Starting & Sending to {ADMIN}")
    updater = Updater(
        TOKEN, use_context=True, request_kwargs=config.get("REQUEST_KWARGS")
    )
    updater.dispatcher.add_handler(CommandHandler("start", start))
    updater.dispatcher.add_handler(CommandHandler("add", add))
    updater.dispatcher.add_handler(CommandHandler("del", delete))
    updater.dispatcher.add_handler(CommandHandler("run", run))
    updater.dispatcher.add_error_handler(error)
    updater.start_polling()
    logger.info(f"Bot @{updater.bot.get_me().username} started.")
    updater.bot.set_my_commands(
        [["start", "使用说明"], ["add", "添加数字平台账户"], ["del", "移除数字平台账户"], ["run", "立即运行"]]
    )
    for i, conf in enumerate(config.get("USERS")):
        updater.job_queue.run_daily(
            checkin_queue,
            datetime.time(
                0,
                2 + i,
                SystemRandom().randrange(60),
                SystemRandom().randrange(1000000),
                datetime.timezone(datetime.timedelta(hours=8)),
            ),
            context={
                "username": conf.get("USERNAME"),
                "password": conf.get("PASSWORD"),
                "chat": conf.get("CHAT"),
                "region": conf.get("REGION", 1),
            },
            name=conf.get("USERNAME"),
        )
    updater.idle()
