import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from _dataset_.data_processing import CDDataset
from losses.cross_entropyloss import cross_entropy
from losses.mmIoULoss import mmIoULoss
from losses.focal_loss import FocalLoss, get_alpha, softmax_helper
from loguru import logger
from datetime import datetime
class Trainer:
    def __init__(self, model, config):
        cur_time = datetime.now()
        date_str = cur_time.strftime("%Y-%m-%d %H-%M-%S")
        date_str = date_str.replace(' ', '-')
        self.config = config
        self.device = self.config.train.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_set, self.val_set = self.load_data()
        self.train_loader, self.val_loader = self.get_dataloader()
        self.is_train = True

        self.losses = self.configure_loss()
        if isinstance(self.losses, nn.Module):
            self.losses = self.losses.to(self.device)

        self.optimizer = self.get_optimizer()
    def load_data(self):
        logger.info('Loading Data')
        train_set = CDDataset(root_dir=self.config.data['data_path'], split='train',
                                 img_size=self.config.model['img_size'],is_train=True,
                                 label_transform='norm')
        val_set = CDDataset(root_dir=self.config.data['data_path'], split='val',
                                 img_size=self.config.model['img_size'],is_train=False,
                                 label_transform='norm')
        return train_set, val_set


    def get_dataloader(self):
        logger.info('Get DataLoader')
        train_loader = DataLoader(self.train_set,
                                  batch_size=self.config.train['batch_size'],
                                  collate_fn=self.collate_func,
                                  pin_memory=True
                                  )
        val_loader = DataLoader(self.val_set,
                                batch_size=self.config.train['batch_size'],
                                collate_fn=self.collate_func,
                                pin_memory=True
                                )
        return train_loader, val_loader
    def collate_func(self, batch):
        """
        :param batch: Dataset 返回的Cunstomized的数据结构
        :return: 处理后的返回的应该是 list dict tensor这些常规类型
        """
        batchA, batchB, batchL, name = [], [] ,[] ,[]
        for index in range(len(batch)):
            batchA.append(batch[index]['A'])
            batchB.append(batch[index]['B'])
            batchL.append(batch[index]['L'])
            name.append(batch[index]['name'])
        batchA = torch.stack(batchA)
        batchB = torch.stack(batchB)
        batchL = torch.stack(batchL)

        return {
            "A": batchA,
            "B": batchB,
            "L": batchL,
            "name": name
        }
    def configure_loss(self):
        if self.config.train['loss_type'] == 'ce':
            logger.info('Using CrossEntropy Loss')
            return cross_entropy
        if self.config.train['loss_type'] == 'focal':
            logger.info('Using Focal Loss')
            alpha = get_alpha(self.train_set)
            return FocalLoss(apply_nonlin = softmax_helper, alpha = alpha, gamma = 2, smooth = 1e-5)
    def cal_loss(self, feature, label, *args, **kwargs):
        # for single loss
        if self.config.train['loss_type'] == 'ce':
            return cross_entropy(feature, label)
        if self.config.train['loss_type'] == 'focal':
            return self.losses(feature, label)

    def multi_loss_calculaing(self):
        pass
    def get_optimizer(self):
        """根据配置选择优化器"""
        optimizer_name = self.config.train["optimizer"]
        lr = self.config.train["lr"]
        weight_decay = self.config.train["weight_decay"]

        if optimizer_name == "sgd":
            logger.info('Using SGD Optimizer')
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=self.config.train.get("momentum", 0.9),
                weight_decay=weight_decay
            )
        elif optimizer_name == "adam":
            logger.info('Using Adam Optimizer')
            betas = self.config.train.get("betas")
            betas = [float(item) for item in betas]
            eps = float(self.config.train.get('eps'))

            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                betas=tuple(betas),
                eps=eps,
                weight_decay=weight_decay
            )
        elif optimizer_name == "adamw":
            logger.info('Using AdamW Optimizer')

            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                betas=tuple(self.config.train.get("betas", (0.9, 0.999))),
                eps=self.config.train.get("eps", 1e-8),
                weight_decay=weight_decay
            )
        elif optimizer_name == "rmsprop":
            logger.info('Using RMSprop Optimizer')

            optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=lr,
                momentum=self.config.train.get("momentum", 0.9),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        print(f"Using optimizer: {optimizer}")
        return optimizer

