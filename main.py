import os

import wandb
import numpy as np
import torch
import argparse
import wandb

from configs._base_.yaml_parser import load_config, Config
from utils.prepration import build_trainer
from metric.tensorboard_logger import MyLogger
from datetime import datetime
from loguru import logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Model Parameter Config")

    # 允许传递超参数
    parser.add_argument("--config", type=str, default='./configs/base_config.yaml', help="Path to config file")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--device", type=str, help="Training device (cuda/cpu)")
    parser.add_argument("--name", default="resnet18",choices=["resnet18", "DESSN"], type=str, help="Backbone name")
    parser.add_argument("--loss_type", default="focal",choices=["ce", "focal"], type=str, help="Configure loss")
    parser.add_argument("--epoch", default=20, type=int)

    args = parser.parse_args()

    # 读取 YAML 配置
    config_dict = load_config(args.config)
    config = Config(**config_dict)

    # 用命令行参数覆盖 YAML 参数
    if args.lr:
        config.train["lr"] = args.lr
    if args.batch_size:
        config.train["batch_size"] = args.batch_size
    if args.device:
        config.train["device"] = args.device
    if args.name:
        config.model["name"] = args.name
    if args.loss_type:
        config.train['loss_type'] = args.loss_type
    if args.epoch:
        config.train['epoch'] = args.epoch
    return config


if __name__ == '__main__':
    config = parse_args()
    current_time = datetime.now()
    time_str = current_time.strftime("%Y%m%d_%H%M")
    os.makedirs('tensorboard_log', exist_ok=True)
    logger.add('logs/{time}' + '-' + config.model["name"] + config.train['loss_type'] + '-' + config.data['dataset'] + '-' + config.train['optimizer'] + '.log',
               rotation='50 MB', level='DEBUG')

    logger.info(config)

    writer = MyLogger(logdir=os.path.join('tensorboard_log',time_str))

    ### GET TRAINER ###
    if config.model['name'] == 'resnet18':
        # Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
        trainer = build_trainer(config, writer=writer)
    else:
        raise NotImplementedError("Not Implement Model")

    ### START TRAIN ###
    trainer.start_train()




