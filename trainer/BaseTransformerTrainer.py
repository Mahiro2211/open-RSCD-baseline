import os

import h5py
import numpy as np
import torch
from loguru import logger
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from tqdm import tqdm

from metric.metric_tools import cm2score, get_confuse_matrix
from models.BaseTransformer import BASE_Transformer
from trainer.Trainer import Trainer

progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
)


class BASETranformer_Trainer(Trainer):
    def __init__(self, config, model, writer=None):
        super().__init__(model, config)
        self.model = model.to(self.device)
        self.cur_loss = 0
        self.run_log = writer if writer is not None else None
        self.max_result = {
            "acc": 0.0,
            "loss": 0.0,
            "miou": 0.0,
            "iou_0": 0.0,
            "mf1": 0.0,
            "F1_0": 0.0,
            "F1_1": 0.0,
            "recall_0": 0.0,
            "recall_1": 0.0,
            "iou_1": 0.0,
            "precision_0": 0.0,
            "precision_1": 0.0,
        }

    def save_chpt(self, epoch):
        os.makedirs("chpt", exist_ok=True)
        torch.save(
            {
                "epoch": epoch,  # 当前训练到第几轮（可选）
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            f"chpt/Epoch{epoch}_{self.config.model['name']}_checkpoint.pth",
        )

    def update_best_result(self, result_dict):
        logger.info(">>>>>>>>>>>>>BEST RESULT<<<<<<<<<<<<<<")
        logger.info(">>>>>>>>>>>>>BEST RESULT<<<<<<<<<<<<<<")
        for key, value in result_dict.items():
            if self.max_result[key] < result_dict[key]:
                self.max_result[key] = result_dict[key]
        logger.info(
            f"acc: {self.max_result['acc']}, miou: {self.max_result['miou']}, mf1: {self.max_result['mf1']}"
        )
        logger.info(
            f"iou_0: {self.max_result['iou_0']}, F1_0: {self.max_result['F1_0']}, F1_1: {self.max_result['F1_1']}"
        )
        logger.info(
            f"recall_0: {self.max_result['recall_0']}, recall_1: {self.max_result['recall_1']}"
        )
        logger.info(
            f"precision_0: {self.max_result['precision_0']}, precision_1: {self.max_result['precision_1']}"
        )
        logger.info(">>>>>>>>>>>>>BEST RESULT<<<<<<<<<<<<<<")
        logger.info(">>>>>>>>>>>>>BEST RESULT<<<<<<<<<<<<<<")

    def start_train(self):
        logger.info("Start Training BASETranformer Model!")
        tot_epoch = self.config.train["num_epochs"]

        for epoch in range(1, tot_epoch + 1):
            result_dict = self.train_one_epoch(epoch)
            logger.info(f"Epoch: {epoch} -- Train Loss: {result_dict['loss']}")
            if self.run_log is not None:
                self.run_log.log_dict(
                    {
                        "acc/train": result_dict["acc"],
                        "loss/train": result_dict["loss"],
                        "miou/train": result_dict["miou"],
                        "mf1/train": result_dict["mf1"],
                        "F1_0/train": result_dict["F1_0"],
                        "F1_1/train": result_dict["F1_1"],
                        "recall_0/train": result_dict["recall_0"],
                        "recall_1/train": result_dict["recall_1"],
                        "precision_0/train": result_dict["precision_0"],
                        "precision_1/train": result_dict["precision_1"],
                    },
                    step=epoch,
                )
            if epoch % self.config.train["eval_interval"] == 0:
                self.is_train = False
                logger.info("Starting Validate....")
                self.validate(is_train=self.is_train, epoch=epoch)

    def train_one_epoch(self, epoch):
        self.model.train()
        logger.info(f"Training Epoch {epoch} ......")
        self.cur_loss = 0
        with tqdm(total=len(self.train_loader), ncols=100) as pbar:
            tot_tr_pred, tot_tr_label = [], []
            for batch_idx, batch in enumerate(self.train_loader):
                # 半精度训练
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    img_A, img_B, label = (
                        batch["A"].to(self.device),
                        batch["B"].to(self.device),
                        batch["L"].to(self.device),
                    )

                    pred = self.model(img_A, img_B)

                    tot_tr_pred.append(pred[0].detach().cpu())
                    tot_tr_label.append(label.detach().cpu())
                    loss = self.cal_loss(feature=pred[0], label=label)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(
                    set_to_none=True
                )  # set_to_none=True here can modestly improve performance (见pytorch官方文档)

                self.cur_loss += loss.item()
                pbar.update(1)

        tot_tr_pred = torch.cat(tot_tr_pred, dim=0).cpu()
        tot_tr_label = torch.cat(tot_tr_label, dim=0).cpu()
        result_dict = self.get_all_metric(tot_tr_label, tot_tr_pred, epoch)
        result_dict["loss"] = self.cur_loss / len(self.train_set)
        return result_dict

    def validate(self, is_train, epoch):
        ds_len = len(self.train_set) if is_train == True else len(self.val_set)
        dl_for_val = self.train_loader if is_train == True else self.val_loader
        tot_pred = []
        tot_label = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in progress.track(
                enumerate(dl_for_val), description="Validating"
            ):
                img_A = batch["A"].to(self.device)
                img_B = batch["B"].to(self.device)
                label = batch["L"].to(self.device)

                # 不需要 unsqueeze，batch_size=1 的话已经是 [1, C, H, W]
                output = self.model(img_A, img_B)

                tot_pred.append(output[0].detach())  # 避免不必要的 autograd
                tot_label.append(label)

        tot_pred = torch.cat(tot_pred, dim=0).cpu()
        tot_label = torch.cat(tot_label, dim=0).cpu()

        M = "train" if is_train else "val"
        logger.info(f"Calculating {M} Metric")
        result_dict = self.get_all_metric(tot_label, tot_pred, epoch)

        # pdb.set_trace()
        if self.config.train["save_result"] == True and is_train == False:
            result_root = "./result"
            ds_name = self.config.data["dataset"]
            message = "train" if self.is_train else "val"
            os.makedirs("./result", exist_ok=True)

            with h5py.File(
                os.path.join(f"{result_root}/{ds_name}_{message}_{str(epoch)}_.h5"), "w"
            ) as h5f:
                h5f.create_dataset("pred", data=tot_pred.numpy())
                h5f.create_dataset("label", data=tot_label.numpy())
                h5f["desc"] = "Trained with Resnet18"
            if self.run_log is not None:
                self.run_log.log_dict(
                    {
                        "acc/val": result_dict["acc"],
                        "miou/val": result_dict["miou"],
                        "mf1/val": result_dict["mf1"],
                        "F1_0/val": result_dict["F1_0"],
                        "F1_1/val": result_dict["F1_1"],
                        "recall_0/val": result_dict["recall_0"],
                        "recall_1/val": result_dict["recall_1"],
                        "precision_0/val": result_dict["precision_0"],
                        "precision_1/val": result_dict["precision_1"],
                    },
                    step=epoch,
                )
        if is_train == False:
            self.update_best_result(result_dict)
        self.is_train = True
        self.model.train()

    def get_all_metric(self, label, pred, epoch):
        cm = get_confuse_matrix(
            self.config.train["num_classes"],
            label.detach().cpu().numpy(),
            np.argmax(pred.detach().cpu().numpy(), axis=1),
        )
        result_dict = cm2score(cm)
        if self.is_train:
            logger.info(">>>>>>>>>>>>>TRAIN METRIC<<<<<<<<<<<<<<")
        else:
            logger.info(">>>>>>>>>>>>>VAL METRIC<<<<<<<<<<<<<<")
        logger.info(
            f"Epoch: {epoch}, acc: {result_dict['acc']}, miou: {result_dict['miou']}, mf1: {result_dict['mf1']}"
        )
        logger.info(
            f"iou_0: {result_dict['iou_0']}, F1_0: {result_dict['F1_0']}, F1_1: {result_dict['F1_1']}"
        )
        logger.info(
            f"recall_0: {result_dict['recall_0']}, recall_1: {result_dict['recall_1']}"
        )
        logger.info(
            f"precision_0: {result_dict['precision_0']}, precision_1: {result_dict['precision_1']}"
        )
        return result_dict


if __name__ == "__main__":
    from loguru import logger

    from main import parse_args

    config = parse_args()
    logger.add(
        "logs/{time}"
        + config.data["dataset"]
        + "-"
        + config.train["optimizer"]
        + ".log",
        rotation="50 MB",
        level="DEBUG",
    )

    net = BASE_Transformer(
        input_nc=3,
        output_nc=2,
        token_len=4,
        resnet_stages_num=4,
        with_pos="learned",
        enc_depth=1,
        dec_depth=8,
    )

    # print(net)
    trainer = BASETranformer_Trainer(model=net, config=config)
    trainer.run_log().close()
    # trainer.start_train()
    # trainer.validate(False, 1)
