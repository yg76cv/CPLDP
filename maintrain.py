
import argparse, os, shutil
from collections import OrderedDict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import clip, pyiqa
from utils import backbone, clip_score, dataloader_prompt_add
from utils import dataloader_images as dataloader_sharp
from utils.prompt_model import Prompts
from utils.training_losses import ColorLoss, L_brightness, L_contrast, compute_enhance_loss
from utils.training_utils import DegradationAwareController, EpochPSNRLRScheduler, weights_init

task_name = "experiment"
dstpath = "./outputs/" + task_name + "/train_scripts"
os.makedirs(dstpath, exist_ok=True)
shutil.copy(__file__, dstpath)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")
model.to(device)
for para in model.parameters():
    para.requires_grad = False

def strip_module_prefix(state_dict):
    return OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state_dict.items())

def make_loader(dataset, batch_size, num_workers):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

def train(config):
    U_net = backbone.UNet(3, 3, condition_dim=512).to(device)
    U_net = torch.nn.DataParallel(U_net)
    iqa_metric = pyiqa.create_metric("psnr", device=device)

    if config.load_pretrain_prompt:
        learn_prompt = Prompts(model, config.length_prompt).to(device)
        learn_prompt.load_state_dict(strip_module_prefix(torch.load(config.prompt_pretrain_dir, map_location=device)))
        config.num_clip_pretrained_iters = 0
        torch.save(learn_prompt.state_dict(), config.prompt_snapshots_folder + "pretrained_prompt.pth")
    else:
        if config.num_clip_pretrained_iters < 3000:
            print("WARNING: num_clip_pretrained_iters < 3000, reset to 8000.")
            config.num_clip_pretrained_iters = 8000
        learn_prompt = Prompts(model, config.length_prompt).to(device)

    learn_prompt = torch.nn.DataParallel(learn_prompt)
    U_net.module.apply(weights_init)
    if config.load_pretrain:
        U_net.module.load_state_dict(strip_module_prefix(torch.load(config.pretrain_dir, map_location=device)), strict=True)
        torch.save(U_net.state_dict(), config.train_snapshots_folder + "pretrained_network.pth")

    train_dataset = dataloader_sharp.underwater_loader(config.underwater_images_path, config.reference_path, config.detype, config.data_path)
    train_loader = make_loader(train_dataset, config.train_batch_size, config.num_workers)
    prompt_train_dataset = dataloader_prompt_add.underwater_loader(config.underwater_images_path, config.reference_path, config.detype, config.data_path)
    prompt_train_loader = make_loader(prompt_train_dataset, config.prompt_batch_size, config.num_workers)
    if hasattr(prompt_train_dataset, "sample_ids"):
        counts = torch.zeros(6, dtype=torch.float32)
        for s in prompt_train_dataset.sample_ids:
            try:
                t = int(s.get("de_type", -1))
            except Exception:
                t = -1
            if 0 <= t < 6:
                counts[t] += 1.0
        if counts.sum() > 0:
            w = counts.sum() / (counts + 1e-6)
            w = w / (w.mean() + 1e-6)
        else:
            w = torch.ones(6, dtype=torch.float32)
        w[5] *= float(config.stage1_ref_ce_boost)
        stage1_class_weights = w.to(device)
    else:
        stage1_class_weights = torch.tensor([1, 1, 1, 1, 1, float(config.stage1_ref_ce_boost)], dtype=torch.float32, device=device)

    L_clip2 = clip_score.L_clip_from_feature(model).to(device)
    L_bright, L_contrast_loss, L_MSE, L_color = L_brightness(), L_contrast(), nn.MSELoss(), ColorLoss()
    deg_controller = DegradationAwareController(config)
    adam_betas = (config.adam_beta1, config.adam_beta2)
    train_optimizer = torch.optim.Adam(U_net.parameters(), lr=config.train_lr, betas=adam_betas, weight_decay=config.weight_decay)
    prompt_optimizer = torch.optim.Adam(learn_prompt.parameters(), lr=config.prompt_lr, betas=adam_betas, weight_decay=config.weight_decay)
    psnr_lr_scheduler = EpochPSNRLRScheduler(train_optimizer, init_lr=config.train_lr, factor=0.5, patience=3, min_lr=config.eta_min, threshold=0.02)

    total_iteration = 0
    stage2_total_iteration = 0
    max_score_psnr = 32.0
    pr_last_few_iter = 0
    score_psnr = [0.0] * 30

    clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    img_resize = transforms.Resize((224, 224))

    def set_optimizer_lr(optimizer, lr):
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    print("Start training ...")
    for epoch in range(1, config.num_epochs + 1):

        if total_iteration < config.num_clip_pretrained_iters:
            U_net.eval()
            for p in U_net.parameters():
                p.requires_grad_(False)
            for p in learn_prompt.parameters():
                p.requires_grad_(True)

            for iteration, (img_underwater_p, label) in enumerate(prompt_train_loader):
                if total_iteration >= config.num_clip_pretrained_iters:
                    break

                img_underwater_p = img_underwater_p.to(device)
                label = label.to(device)

                feat = img_underwater_p.to(device)
                label = label.to(device)

                outputs_dict = learn_prompt({"tensor": feat, "flag": 0})
                logits6 = outputs_dict["degradation"][:, :6]

                ref_feat = outputs_dict["reference"]
                if ref_feat.dim() == 3:
                    ref_feat = ref_feat.mean(dim=0)
                ref_feat = F.normalize(ref_feat, dim=-1)

                ce = F.cross_entropy(logits6, label, weight=stage1_class_weights)

                mask = (label == 5)
                if mask.any():
                    img_norm = F.normalize(feat[mask], dim=-1)
                    sim = (img_norm @ ref_feat.t()).squeeze(1)
                    ref_align = (1 - sim).mean()
                else:
                    ref_align = torch.zeros((), device=device)

                loss = ce + float(config.stage1_ref_align_w) * ref_align

                prompt_optimizer.zero_grad()
                loss.backward()
                prompt_optimizer.step()

                if (total_iteration + 1) % config.prompt_snapshot_iter == 0:
                    torch.save(learn_prompt.state_dict(), config.prompt_snapshots_folder + f"iter_{total_iteration+1}.pth")
                total_iteration += 1
            torch.save(learn_prompt.state_dict(), config.prompt_snapshots_folder + "prompt_after_stage1.pth")
            continue

        learn_prompt.eval()
        for p in learn_prompt.parameters():
            p.requires_grad_(False)
        for p in U_net.parameters():
            p.requires_grad_(True)
        U_net.train()

        with torch.no_grad():
            deg_prompts = [
                p.parameter.squeeze(0) if p.parameter.dim() == 3 else p.parameter
                for p in learn_prompt.module.embedding_prompts[:5]
            ]
            deg_prompts = torch.stack(deg_prompts)
            prompt_module = learn_prompt.module
            tokenized_prompts = prompt_module.tokenize_templates(5, device)
            deg_text_features = prompt_module.text_encoder(deg_prompts, tokenized_prompts)
            ref_tokenized = prompt_module.tokenize_templates(1, device)
            ref_prompt = prompt_module.ref_embedding
            if ref_prompt.dim() == 3 and ref_prompt.size(0) > 1:
                ref_prompt = ref_prompt[0].unsqueeze(0)
            elif ref_prompt.dim() == 2:
                ref_prompt = ref_prompt.unsqueeze(0)
            ref_text_features = learn_prompt.module.text_encoder(ref_prompt, ref_tokenized)

        epoch_psnr_sum = 0.0
        epoch_psnr_cnt = 0
        epoch_base_lr = psnr_lr_scheduler.lr

        for iteration, (img_underwater, img_label, deg_type) in enumerate(train_loader):
            img_underwater = img_underwater.to(device)
            img_label = img_label.to(device)
            deg_type = deg_type.to(device)

            dynamic_params = deg_controller.get_dynamic_params(deg_type.cpu().numpy())
            batch_lr_factor = float(dynamic_params["lr_factor"].float().mean().item())
            batch_lr_factor = max(0.5, min(2.0, batch_lr_factor))
            effective_lr = max(config.eta_min, epoch_base_lr * batch_lr_factor)
            set_optimizer_lr(train_optimizer, effective_lr)

            with torch.no_grad():
                img_clip = img_resize(img_underwater)
                img_clip = clip_normalizer(img_clip)
                image_features = model.encode_image(img_clip)

                outputs = image_features @ deg_text_features.T
                deg_type_preds = torch.argmax(outputs, dim=1)
                condition_vectors = deg_text_features[deg_type_preds]

            enhanced_map = U_net(img_underwater, condition=condition_vectors)
            sample_mse = ((enhanced_map - img_label) ** 2).mean(dim=[1, 2, 3])
            Le = sample_mse.mean()
            color_loss = L_color(enhanced_map)
            bright_loss = L_bright(enhanced_map, img_underwater)
            contrast_loss_val = L_contrast_loss(enhanced_map)
            ref_sim = L_clip2(enhanced_map, ref_text_features)
            color_mask = (deg_type < 3).float()
            light_mask = (deg_type == 3).float()
            haze_mask = (deg_type == 4).float()

            Lt_color = (sample_mse * color_mask).sum() / (color_mask.sum() + 1e-6)
            Lt_light = (sample_mse * light_mask).sum() / (light_mask.sum() + 1e-6)
            Lt_haze = (sample_mse * haze_mask).sum() / (haze_mask.sum() + 1e-6)

            omega_L = torch.stack([torch.clamp(Lt_color, min=0.3, max=2.0), torch.clamp(Lt_light, min=0.3, max=2.0), torch.clamp(Lt_haze, min=0.3, max=2.0)])
            loss, _ = compute_enhance_loss(Le, deg_type, color_loss, bright_loss, contrast_loss_val, ref_sim, omega_L, lambda_re=0.9)

            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()

            with torch.no_grad():
                batch_psnr = torch.mean(iqa_metric(enhanced_map.clamp(0, 1), img_label))
                epoch_psnr_sum += batch_psnr.item()
                epoch_psnr_cnt += 1

                score_psnr[pr_last_few_iter] = batch_psnr
                pr_last_few_iter = (pr_last_few_iter + 1) % len(score_psnr)
                sample_losses = F.mse_loss(enhanced_map, img_label, reduction="none").mean(dim=[1, 2, 3])
                sample_psnr = 10 * torch.log10(1.0 / sample_losses)
                deg_controller.update_statistics(deg_type.cpu(), sample_losses.cpu(), sample_psnr.cpu())

            avg_psnr_window = sum(score_psnr) / len(score_psnr)
            if avg_psnr_window > max_score_psnr and ((total_iteration + 1) % config.best_save_iter) == 0:
                max_score_psnr = avg_psnr_window
                torch.save(U_net.state_dict(), config.train_snapshots_folder + f"best_model_iter_{total_iteration+1}.pth")
            total_iteration += 1
            stage2_total_iteration += 1

        if epoch_psnr_cnt > 0:
            epoch_avg_psnr = epoch_psnr_sum / epoch_psnr_cnt
            current_lr = psnr_lr_scheduler.step(epoch_avg_psnr)
            print(f"[Epoch {epoch}] mean PSNR={epoch_avg_psnr:.4f}, next_epoch_base_lr={current_lr:.7f}, stage2_iters={stage2_total_iteration}")
        else:

            set_optimizer_lr(train_optimizer, psnr_lr_scheduler.lr)

def str2bool(value):
    value = str(value).lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")

def build_parser():
    parser = argparse.ArgumentParser(description="Train the relative-PSNR degradation-aware underwater enhancement model.")
    parser.add_argument('-b', '--underwater_images_path', type=str, default="./data/raw/")
    parser.add_argument('-r', '--reference_path', type=str, default='./data/reference/')
    parser.add_argument('--detype', type=str, nargs='+', default=['green', 'blue', 'green_blue', 'dark', 'haze'], help='degradation types used for training')
    parser.add_argument('--data_path', type=str, default="./data/splits/")
    parser.add_argument('--length_prompt', type=int, default=16)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--prompt_batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=8000)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--train_lr', type=float, default=0.00002)
    parser.add_argument('--prompt_lr', type=float, default=0.000005)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.99)
    parser.add_argument('--eta_min', type=float, default=5e-6)
    parser.add_argument('--relative_lr_alpha', type=float, default=1.0, help='relative-difficulty lr scaling coefficient alpha')
    parser.add_argument('--relative_lr_eps', type=float, default=1e-6, help='epsilon for relative-difficulty lr scaling')
    parser.add_argument('--num_clip_pretrained_iters', type=int, default=10000)
    parser.add_argument('--best_save_iter', type=int, default=20)
    parser.add_argument('--prompt_snapshot_iter', type=int, default=100)
    parser.add_argument('--train_snapshots_folder', type=str, default="./outputs/" + task_name + "/snapshots_train/")
    parser.add_argument('--prompt_snapshots_folder', type=str, default="./outputs/" + task_name + "/snapshots_prompt/")
    parser.add_argument('--load_pretrain', type=str2bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default='./checkpoints/pretrained_network.pth')
    parser.add_argument('--load_pretrain_prompt', type=str2bool, default=False)
    parser.add_argument('--prompt_pretrain_dir', type=str, default='./checkpoints/prompt_after_stage1.pth')
    parser.add_argument('--stage1_ref_align_w', type=float, default=200.0, help='Stage1: reference class CLIP feature alignment weight')
    parser.add_argument('--stage1_ref_ce_boost', type=float, default=2.0, help='Stage1: additional weight for reference class in CE')
    return parser

def main():
    config = build_parser().parse_args()
    os.makedirs(config.train_snapshots_folder, exist_ok=True)
    os.makedirs(config.prompt_snapshots_folder, exist_ok=True)
    train(config)

if __name__ == "__main__":
    main()
