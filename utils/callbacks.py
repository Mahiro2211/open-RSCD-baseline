import rich
from lightning.pytorch.callbacks import Callback, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
import pdb



class CustomRichProgressBar(RichProgressBar):
    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        if hasattr(pl_module, 'current_loss'):
            items["Loss"] = f"{pl_module.current_loss:.4f}"
        return items


class BeforeTrain(Callback):

    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")

class InitLossFunction(Callback):

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # pl_module.config.loss_name ==
        pass

class InitOptAndScheduler(Callback):
    def configure_optimizers(self):
        """自动配置优化器和学习率调度器"""
        optimizer_mapping = {
            "adam": Adam,
            "sgd": SGD,
            "rmsprop": RMSprop
        }

        optimizer_class = optimizer_mapping.get(self.config.optimizer_name)
        rich.print(f'Use {self.config.optimizer_name} optmizer')

        if not optimizer_class:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_name}")

        # 构造优化器参数
        optimizer_kwargs = {
            "lr": self.config.learning_rate,
            "weight_decay": self.config.weight_decay
        }
        if self.config.optimizer_name == "sgd":
            optimizer_kwargs["momentum"] = self.config.momentum
        else:
            optimizer_kwargs["betas"] = self.config.betas
            optimizer_kwargs["eps"] = self.config.eps

        optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)

        # 选择学习率调度器
        scheduler_mapping = {
            "steplr": StepLR(optimizer, step_size=self.config.step_size, gamma=self.config.gamma),
            "reducelronplateau": ReduceLROnPlateau(optimizer, mode="min", factor=self.config.gamma,
                                                   patience=self.config.patience, min_lr=self.config.min_lr),
            "cosineannealing": CosineAnnealingLR(optimizer, T_max=self.config.T_max, eta_min=self.config.min_lr)
        }

        scheduler = scheduler_mapping.get(self.config.scheduler_name)
        if not scheduler:
            return optimizer  # 若未指定 scheduler，则仅返回优化器

        if self.config.scheduler_name == "reducelronplateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train_loss",  # 监视 loss 进行调整
                    "interval": "epoch",
                    "frequency": 1
                }
            }

        return optimizer, scheduler  # 其他 scheduler 直接返回列表格式