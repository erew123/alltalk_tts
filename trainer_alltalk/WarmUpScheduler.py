from torch.optim.lr_scheduler import LRScheduler

class WarmUpScheduler(LRScheduler):
    def __init__(self, optimizer, total_epochs, warmup_lr, target_lr, after_scheduler=None):
        self.total_epochs = total_epochs
        self.warmup_epochs = int(total_epochs * 0.05)
        if self.warmup_epochs <= 0:
            self.warmup_epochs = 1
        self.warmup_lr = warmup_lr
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        print(f"[WARMUP] {self.warmup_epochs} epochs")
        super(WarmUpScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            progress = self.last_epoch / self.warmup_epochs
            return [self.warmup_lr + progress * (self.target_lr - self.warmup_lr) for _ in self.base_lrs]
        if self.after_scheduler:
            return self.after_scheduler.get_last_lr()
        return [base_lr for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is not None and epoch < self.warmup_epochs:
            self.last_epoch = epoch
        elif self.after_scheduler and epoch is not None:
            if epoch < self.warmup_epochs:
                self.last_epoch = epoch
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
                self.last_epoch = epoch
        else:
            super(WarmUpScheduler, self).step(epoch)
        if self.after_scheduler and self.last_epoch >= self.warmup_epochs:
            self.after_scheduler.step(epoch - self.warmup_epochs if epoch is not None else None)