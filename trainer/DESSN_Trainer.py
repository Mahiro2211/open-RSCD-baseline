import torch

from trainer.Trainer import Trainer
from models.DESSN import Net
from loguru import logger


class DESSN_Trainer(Trainer):
    def __init__(self, config, model=Net(in_ch=3, out_ch=2)):
        super().__init__(model, config)
        self.model = model.to(self.device)
        self.cur_loss = 0

    def start_train(self):
        logger.info("Start Training DESSN Model!")
        tot_epoch = self.config.train["num_epochs"]

        for epoch in range(1, tot_epoch + 1):
            result_dict = self.train_one_epoch(epoch)
            logger.info(f'Epoch: {epoch} -- Train Loss: {result_dict["loss"]}')
            if epoch % self.config.train['eval_interval'] == 0:
                logger.info(f'Starting Validate....')
                self.validate()
    def train_one_epoch(self, epoch):
        self.model.train()
        self.logger.info(f'Training Epoch {epoch} ......')
        self.cur_loss = 0
        # torch.cuda.empty_cache()
        for batch_idx, batch in enumerate(self.train_loader):
            # 半精度训练
            with torch.autocast(device_type=self.device, dtype=torch.float16):
                img_A, img_B, label = batch['A'].to(self.device), batch['B'].to(self.device), batch['L'].to(self.device)

                pred = self.model(img_A, img_B)

                loss = self.cal_loss(feature=pred, label=label)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True) # set_to_none=True here can modestly improve performance (见pytorch官方文档)
            self.cur_loss += loss.item()

        return {
            'loss': self.cur_loss / len(self.train_set)
        }
    def validate(self, is_train):
        self.model.eval()

        ds_len = len(self.train_set) if is_train == True else len(self.val_set)
        ds_for_validate = self.train_set if is_train == True else self.val_set
        tot_pred = []
        tot_label = []

        for batch_idx, batch in enumerate(ds_for_validate):
            img_A, imgB, label = batch['A'], batch['B'], batch['L']
            tot_pred.append(self.model(img_A.unsqueeze(0), imgB.unsqueeze(0)))
            tot_label.append(label)
        tot_pred = torch.stack(tot_pred)
        tot_label = torch.stack(label)

if __name__ == '__main__':
    from main import parse_args
    torch.set_float32_matmul_precision('medium')
    config = parse_args()
    trainer = DESSN_Trainer(config)

    # print(trainer.losses)
    trainer.start_train()