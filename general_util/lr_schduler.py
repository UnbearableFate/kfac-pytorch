import torch.optim as optim

class WarmupScheduler(optim.lr_scheduler._LRScheduler):
    """ Linear Warmup Scheduler for the first few epochs """
    def __init__(self, optimizer, warmup_epochs, base_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            scale = (self.last_epoch + 1) / self.warmup_epochs
            return [scale * self.base_lr for _ in self.base_lrs]
        return self.base_lrs  # Default to optimizer's lr after warmup

def get_scheduler(optimizer, method):
    if method == "KFAC":
        milestones = [35, 75, 90]  # Decay points for K-FAC
    elif method == "SGD":
        milestones = [100, 150]  # Decay points for SGD
    else:
        raise ValueError("Unknown method: Choose 'KFAC' or 'SGD'.")

    warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs=5, base_lr=optimizer.param_groups[0]['lr'])
    decay_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    return warmup_scheduler, decay_scheduler