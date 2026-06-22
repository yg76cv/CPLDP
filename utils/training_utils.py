import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class DegradationAwareController:
    def __init__(self, config, num_degradation_types=5):
        self.config = config
        self.num_types = num_degradation_types
        self.type_counts = torch.zeros(num_degradation_types)
        self.type_losses = torch.zeros(num_degradation_types)
        self.type_psnr = torch.zeros(num_degradation_types)
        self.lr_factors = torch.ones(num_degradation_types)
        self.relative_lr_alpha = getattr(config, "relative_lr_alpha", 1.0)
        self.relative_lr_eps = getattr(config, "relative_lr_eps", 1e-6)

    def update_statistics(self, deg_types, losses, psnrs):
        for i in range(self.num_types):
            mask = deg_types == i
            count = mask.sum().item()
            if count > 0:
                self.type_counts[i] += count
                self.type_losses[i] += losses[mask].sum().item()
                self.type_psnr[i] += psnrs[mask].sum().item()

        valid_mask = self.type_counts > 0
        if valid_mask.any():
            avg_psnr_per_type = torch.zeros_like(self.type_psnr)
            avg_psnr_per_type[valid_mask] = self.type_psnr[valid_mask] / self.type_counts[valid_mask]
            global_avg_psnr = avg_psnr_per_type[valid_mask].mean().item()
        else:
            avg_psnr_per_type = torch.zeros_like(self.type_psnr)
            global_avg_psnr = 0.0

        for i in range(self.num_types):
            if not valid_mask[i]:
                continue
            avg_psnr = avg_psnr_per_type[i].item()
            rel_gap = (global_avg_psnr - avg_psnr) / (global_avg_psnr + self.relative_lr_eps)
            lr_factor = 1.0 + self.relative_lr_alpha * rel_gap
            self.lr_factors[i] = min(2.0, max(0.5, lr_factor))
    def get_dynamic_params(self, deg_types):
        batch_params = {
            "lr_factor": torch.ones(len(deg_types))
        }

        for i, deg_type in enumerate(deg_types):
            batch_params["lr_factor"][i] = self.lr_factors[deg_type]

        return batch_params

    def get_adaptive_augmentation(self, deg_types):
        aug_probs = {
            "flip_prob": 0.5,
            "rotate_prob": 0.5,
            "color_jitter": 0.3
        }

        if 3 in deg_types:
            dark_count = (deg_types == 3).sum().item()
            aug_probs["flip_prob"] = max(0.2, 0.5 - dark_count / len(deg_types) * 0.3)

        if 4 in deg_types:
            haze_count = (deg_types == 4).sum().item()
            aug_probs["color_jitter"] = min(0.7, 0.3 + haze_count / len(deg_types) * 0.4)

        return aug_probs

class EpochPSNRLRScheduler:
    def __init__(self, optimizer, init_lr, factor=0.5, patience=3, min_lr=1e-6, threshold=0.02):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.best_psnr = 0.0
        self.no_improve_epochs = 0
        self.current_lr = init_lr

        for pg in self.optimizer.param_groups:
            pg["lr"] = init_lr

    @property
    def lr(self):
        return self.current_lr

    def set_optimizer_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def step(self, epoch_psnr: float):
        if epoch_psnr > self.best_psnr + self.threshold:
            self.best_psnr = epoch_psnr
            self.no_improve_epochs = 0
        else:
            self.no_improve_epochs += 1

        if self.no_improve_epochs >= self.patience:
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.no_improve_epochs = 0

        self.set_optimizer_lr(self.current_lr)
        return self.current_lr
