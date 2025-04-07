from torch.utils.tensorboard import SummaryWriter
class MyLogger:
    def __init__(self, logdir='runs/exp1'):
        self.writer = SummaryWriter(log_dir=logdir)

    def log(self, metric_name, value, step):
        self.writer.add_scalar(metric_name, value, step)

    def log_dict(self, metrics_dict: dict, step: int):
        for key, value in metrics_dict.items():
            self.writer.add_scalar(key, value, step)
    def close(self):
        self.writer.close()
